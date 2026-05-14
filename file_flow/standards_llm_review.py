#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在上一步 ``*_work.json``（或填答后的 JSON）含 ``document_types`` 与各字段 ``content`` 之后，
读取 **评审标准清单** JSON（与 ``out/standards_example.json`` 一致：顶层数组），
对其中**每一项**调用大模型：将 ``category``、``subcategory``、``content`` 拼成「分类上下文」，
对照条目的 ``standard`` 字段作答；将模型输出写入该条目的 ``review_answer``。

最终写出**结果 JSON**：在输入工作 JSON 的完整拷贝上增加 ``standards_review`` 对象
（含 ``items``：原字段 + ``review_answer``，以及 ``standards_path`` 等元数据），便于下游可视化。

用法（在包含 ``file_flow`` 包的上级目录执行）::

    python -m file_flow.standards_llm_review -i out/某案_work.json -o out/某案_review.json
    python -m file_flow.standards_llm_review --config pipeline.json
    python -m file_flow.standards_llm_review --dry-run -i ... -o ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FILE_FLOW_DIR = Path(__file__).resolve().parent

from .llm_openai import (  # noqa: E402
    LlmEnvConfig,
    call_openai_compatible_chat,
    configure_logging,
    is_http_endpoint_url,
    load_llm_config_for_file_flow,
)
from .naming import (
    review_json_filename_for_base,
    stem_base_from_stage_stem,
)
from .pipeline_merge import (
    file_flow_root,
    load_merged_pipeline_config,
    resolve_pipeline_disk_path,
)
from .step_dotenv import ensure_step_dotenv_loaded  # noqa: E402

ensure_step_dotenv_loaded(_FILE_FLOW_DIR)

_LOG = logging.getLogger(__name__)

# 每条请求中「案卷摘录汇总」上限，避免撑爆上下文
_MAX_DIGEST_CHARS = 100_000

DEFAULT_STANDARDS_REVIEW_SYSTEM = (
    "你是行政执法案卷评审助手。用户会提供："
    "（1）分类上下文：由类别 category、子类 subcategory、相关说明 content 组成；"
    "（2）可选的案卷摘录汇总，来自上一步按 schema 从案卷抽取的各字段 content；"
    "（3）须对照回答的评审标准 standard（可为条文、检查项或具体问题）。"
    "请结合分类上下文与摘录（若提供）作出专业判断：先给出简要结论，再说明依据；"
    "不得编造材料中不存在的事实。若信息不足，须明确说明「无法判断」及原因。"
    "使用简体中文，条理清晰。"
)


def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def _merged_str(merged: dict[str, Any] | None, key: str) -> str | None:
    if not merged:
        return None
    v = merged.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def load_standards_review_system_prompt(merged: dict[str, Any] | None = None) -> str:
    """本环节专用 system：环境变量 / pipeline 的 file_flow_standards_review_system_prompt。"""
    m = merged or {}
    return (
        _env_first("FILE_FLOW_STANDARDS_REVIEW_SYSTEM_PROMPT")
        or _merged_str(m, "file_flow_standards_review_system_prompt")
        or DEFAULT_STANDARDS_REVIEW_SYSTEM
    )


def _truthy_merged(merged: dict[str, Any] | None, key: str, default: bool = True) -> bool:
    if not merged:
        return default
    v = merged.get(key)
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _digest_from_work(work: dict[str, Any], max_chars: int = _MAX_DIGEST_CHARS) -> str:
    """从 ``document_types`` 各 field 的 ``content`` 拼成案卷摘录汇总。"""
    chunks: list[str] = []
    docs = work.get("document_types")
    if not isinstance(docs, list):
        return ""
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_name = str(doc.get("document_name", "") or doc.get("document_type", "")).strip() or "（文书）"
        fields = doc.get("fields")
        if not isinstance(fields, list):
            continue
        for field in fields:
            if not isinstance(field, dict):
                continue
            fn = str(field.get("field_name", "")).strip() or "（字段）"
            raw = field.get("content")
            c = str(raw).strip() if raw is not None else ""
            if c:
                chunks.append(f"【{doc_name} / {fn}】\n{c}")
    text = "\n\n".join(chunks)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n（摘录汇总已截断至前 {max_chars} 字符，原约 {len(text)} 字符）"


def build_standards_review_user_prompt(
    row: dict[str, Any],
    digest: str,
    *,
    attach_digest: bool,
) -> str:
    """
    构造单条标准项的用户消息：分类上下文（category + subcategory + content）+ 可选摘录汇总
    + standard（须作答的评审标准表述）。
    """
    cat = str(row.get("category", "")).strip()
    sub = str(row.get("subcategory", "")).strip()
    cell = str(row.get("content", "")).strip()
    std = str(row.get("standard", "")).strip()

    meta_lines: list[str] = []
    for k in ("score", "penalty", "number"):
        if k not in row:
            continue
        val = row.get(k)
        if val is None or val == "":
            continue
        meta_lines.append(f"- {k}: {val}")

    blocks: list[str] = [
        "【分类上下文】（请将类别、子类、相关说明一并理解）",
        f"类别（category）：{cat or '（无）'}",
        f"子类（subcategory）：{sub or '（无）'}",
        f"相关说明（content）：{cell or '（无）'}",
    ]
    if meta_lines:
        blocks.append("")
        blocks.append("【本条元数据】")
        blocks.extend(meta_lines)

    if attach_digest and digest.strip():
        blocks.extend(["", "【案卷摘录汇总】（来自上一步 schema 各字段 content，供对照）", digest])

    blocks.extend(
        [
            "",
            "【须对照作答的评审标准（standard）】",
            std or "（未配置 standard）",
            "",
            "请仅针对上述「评审标准（standard）」组织答案；若与案卷摘录矛盾，以摘录为准并说明。",
        ]
    )
    return "\n".join(blocks)


def run_standards_llm_review_on_data(
    work: dict[str, Any],
    standards_rows: list[dict[str, Any]],
    standards_path: Path,
    base_cfg: LlmEnvConfig,
    merged: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    返回新 dict：``work`` 深拷贝 + ``standards_review``（每条含 ``review_answer``）。
    """
    out: dict[str, Any] = json.loads(json.dumps(work, ensure_ascii=False))
    sys_prompt = load_standards_review_system_prompt(merged)
    cfg = replace(base_cfg, system_prompt=sys_prompt)
    attach = _truthy_merged(merged, "file_flow_review_attach_schema_digest", default=True)
    digest = _digest_from_work(out) if attach else ""

    items_out: list[dict[str, Any]] = []
    total = len(standards_rows)
    for idx, row in enumerate(standards_rows, start=1):
        if not isinstance(row, dict):
            continue
        one = json.loads(json.dumps(row, ensure_ascii=False))
        user = build_standards_review_user_prompt(one, digest, attach_digest=attach)
        _LOG.info(
            "[环节:标准评审] (%s/%s) category=%s subcategory=%s 用户消息字符数=%s",
            idx,
            total,
            str(one.get("category", ""))[:40],
            str(one.get("subcategory", ""))[:40],
            len(user),
        )
        if dry_run:
            one["review_answer"] = "[dry-run 未调用大模型]"
        else:
            try:
                text = call_openai_compatible_chat(cfg, user)
            except RuntimeError:
                _LOG.exception("[环节:标准评审] 第 %s 条 API 失败", idx)
                raise
            one["review_answer"] = text.strip()
        items_out.append(one)

    out["standards_review"] = {
        "standards_path": str(standards_path.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items_out,
        "digest_attached": attach,
    }
    meta = out.get("_file_flow_meta")
    if isinstance(meta, dict):
        meta = dict(meta)
        meta["standards_review_path"] = str(standards_path.resolve())
        meta["standards_review_count"] = len(items_out)
        out["_file_flow_meta"] = meta
    return out


def _resolve_path(raw: Path, cwd: Path) -> Path:
    if raw.is_absolute():
        return raw.resolve()
    for base in (cwd, _FILE_FLOW_DIR):
        hit = (base / raw).resolve()
        if hit.is_file():
            return hit
    return (_FILE_FLOW_DIR / raw).resolve()


def run_standards_review(
    merged: dict[str, Any],
    *,
    workspace: Path,
    cwd: Path,
    work_input: Path | None = None,
    standards_path: Path | None = None,
    output_path: Path | None = None,
    dry_run: bool = False,
    log_level: str | None = None,
    log_file: Path | None = None,
) -> int:
    """CLI/编排入口：读工作 JSON + 标准清单，写 ``*_review.json``。"""
    configure_logging(level=log_level, log_file=log_file)

    in_raw = work_input
    if in_raw is None:
        for key in ("file_flow_review_work_input", "file_flow_llm_output", "file_flow_llm_input"):
            v = merged.get(key)
            if isinstance(v, str) and v.strip():
                in_raw = Path(v.strip())
                break
    if in_raw is None:
        print(
            "错误: 未指定工作 JSON，请设置 file_flow_review_work_input / file_flow_llm_output / file_flow_llm_input 或使用 -i",
            file=sys.stderr,
        )
        return 1

    st_raw = standards_path
    if st_raw is None:
        ms = merged.get("file_flow_standards_json")
        if isinstance(ms, str) and ms.strip():
            st_raw = Path(ms.strip())
        else:
            st_raw = workspace / "out" / "standards_example.json"

    work_path = _resolve_path(Path(in_raw), cwd)
    standards_disk = _resolve_path(Path(st_raw), cwd)

    out_raw = output_path
    if out_raw is None:
        mo = merged.get("file_flow_review_result_output")
        if isinstance(mo, str) and mo.strip():
            out_raw = Path(mo.strip())
        else:
            base = stem_base_from_stage_stem(work_path.stem, merged)
            out_raw = work_path.with_name(review_json_filename_for_base(base, merged))
    out_path = Path(out_raw)
    out_path = out_path.resolve() if out_path.is_absolute() else (cwd / out_path).resolve()

    if not work_path.is_file():
        print(f"错误: 找不到工作 JSON: {work_path}", file=sys.stderr)
        return 1
    if not standards_disk.is_file():
        print(f"错误: 找不到评审标准 JSON: {standards_disk}", file=sys.stderr)
        return 1

    try:
        work = json.loads(work_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"错误: 无法解析工作 JSON: {e}", file=sys.stderr)
        return 1
    if not isinstance(work, dict):
        print("错误: 工作 JSON 根须为对象", file=sys.stderr)
        return 1

    try:
        std_raw = standards_disk.read_text(encoding="utf-8")
        standards_data = json.loads(std_raw)
    except (OSError, json.JSONDecodeError) as e:
        print(f"错误: 无法解析标准 JSON: {e}", file=sys.stderr)
        return 1
    if not isinstance(standards_data, list):
        print("错误: 评审标准 JSON 顶层须为数组（见 standards_example.json）", file=sys.stderr)
        return 1
    standards_rows = [x for x in standards_data if isinstance(x, dict)]

    base_cfg = load_llm_config_for_file_flow(merged)
    dry = dry_run
    if not dry and (
        not base_cfg.api_base
        or not base_cfg.model
        or not is_http_endpoint_url((base_cfg.api_base or "").strip())
    ):
        print("警告: 大模型未配置有效 URL/模型，改为 dry-run。", file=sys.stderr)
        dry = True

    try:
        result = run_standards_llm_review_on_data(
            work,
            standards_rows,
            standards_disk,
            base_cfg,
            merged,
            dry_run=dry,
        )
    except RuntimeError as e:
        _LOG.exception("[环节:标准评审] 失败: %s", e)
        print(f"错误: {e}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _LOG.info("[环节:写文件] path=%s", out_path)
    print(f"已写出: {out_path.resolve()}")
    return 0


def _resolve_pipeline_cli(cfg_arg: Path | None) -> Path | None:
    return resolve_pipeline_disk_path(file_flow_root(), cfg_arg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按 standards 清单对案卷摘录逐项调用大模型评审，写出完整 review JSON")
    ap.add_argument("--config", type=Path, default=None, help="管线 JSON；默认使用 file_flow 目录下的 pipeline.json")
    ap.add_argument("-i", "--work-input", type=Path, default=None, help="上一步工作 JSON（*_work.json 等）")
    ap.add_argument(
        "-s",
        "--standards",
        type=Path,
        default=None,
        help="评审标准清单 JSON；默认同 pipeline 的 file_flow_standards_json 或 out/standards_example.json",
    )
    ap.add_argument("-o", "--output", type=Path, default=None, help="结果 JSON；默认同目录 {stem}_review.json")
    ap.add_argument("--dry-run", action="store_true", help="不请求 API，review_answer 写占位")
    ap.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="日志级别",
    )
    ap.add_argument("--log-file", type=Path, default=None, help="追加日志文件 UTF-8")
    ns = ap.parse_args(argv)

    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(_FILE_FLOW_DIR)
    configure_logging(level=ns.log_level, log_file=ns.log_file)
    if dotenv_missing:
        _LOG.warning("[环节:环境] 未安装 python-dotenv，已跳过 .env")
    elif env_loaded:
        _LOG.info("[环节:环境] 已加载环境文件 %s 个", len(env_loaded))

    cfg_disk = _resolve_pipeline_cli(ns.config)
    merged = load_merged_pipeline_config(cfg_disk if cfg_disk is not None and cfg_disk.is_file() else None)

    return run_standards_review(
        merged,
        workspace=_FILE_FLOW_DIR,
        cwd=Path.cwd(),
        work_input=ns.work_input,
        standards_path=ns.standards,
        output_path=ns.output,
        dry_run=ns.dry_run,
        log_level=ns.log_level,
        log_file=ns.log_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
