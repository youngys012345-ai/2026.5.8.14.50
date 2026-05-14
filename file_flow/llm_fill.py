#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件模式第二步：读取工作 JSON（``document_types`` / ``fields`` 结构，与 ``schema_example.json`` 一致），
将文书名、字段名、``content`` 摘录与评审要点拼成用户消息，调用大模型，将回复写入字段对象下的 ``answer``。

**仅依赖**仓库根的 ``pipeline_config.py`` + ``pipeline.json``（可选 ``--config``）以及本目录下的
``llm_openai.py``、``pipeline_merge.py``、``step_dotenv.py``。

评审要点优先取 ``related_review_items``（列表多行拼接）；若无则使用 ``description``。

``-i`` / ``-o`` 可省略：此时使用 ``pipeline.json`` 的 ``file_flow_llm_input``、``file_flow_llm_output``。

用法::

    python file_flow/llm_fill.py -i file_flow/out/某案_work.json -o file_flow/out/某案_answered.json
    python file_flow/llm_fill.py --config pipeline.json
    python file_flow/llm_fill.py --dry-run -i ... -o ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from file_flow.llm_openai import (  # noqa: E402
    LlmEnvConfig,
    call_openai_compatible_chat,
    configure_logging,
    is_http_endpoint_url,
    load_llm_config_for_file_flow,
)
from file_flow.pipeline_merge import load_merged_pipeline_config  # noqa: E402
from file_flow.step_dotenv import ensure_step_dotenv_loaded  # noqa: E402

ensure_step_dotenv_loaded(_ROOT)

_LOG = logging.getLogger(__name__)

_FIRST_LLM_LOG_MAX_CHARS = 20000


def _preview_for_log(text: str, max_chars: int = _FIRST_LLM_LOG_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…（共 {len(text)} 字符，已按 max={max_chars} 截断）"


def _field_review_prompt(field_obj: dict[str, Any]) -> str:
    """评审/问答用说明：``related_review_items`` 优先，否则 ``description``。"""
    items = field_obj.get("related_review_items")
    if isinstance(items, list):
        lines = [str(x).strip() for x in items if str(x).strip()]
        if lines:
            return "\n".join(lines)
    d = field_obj.get("description")
    if isinstance(d, str) and d.strip():
        return d.strip()
    if d is not None:
        return str(d).strip()
    return ""


def build_user_prompt(
    document_name: str,
    field_name: str,
    extracted_content: str,
    review_prompt: str,
) -> str:
    """拼接用户消息：摘录 + 评审要点。"""
    c = extracted_content.strip() if extracted_content else "（抽取内容为空）"
    q = review_prompt.strip() if review_prompt else "（未配置 related_review_items / description，请结合摘录作一般性说明）"
    return (
        f"【文书名称】{document_name}\n\n"
        f"【字段名称】{field_name}\n\n"
        f"【与案卷相关的抽取内容】\n{c}\n\n"
        f"【评审要点】\n{q}\n\n"
        "请根据「抽取内容」对照「评审要点」作答。若内容与要点无关，请说明。"
    )


def fill_answers(
    root: dict[str, Any],
    cfg: LlmEnvConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """遍历 ``document_types`` → ``fields``，写入 ``answer``。"""
    out: dict[str, Any] = json.loads(json.dumps(root, ensure_ascii=False))
    meta = out.pop("_file_flow_meta", None)

    doc_types = out.get("document_types")
    if not isinstance(doc_types, list) or not doc_types:
        _LOG.warning("[环节:跳过] 根节点缺少非空 document_types，未执行填答")
        if isinstance(meta, dict):
            out["_file_flow_meta"] = meta
        return out

    total_calls = 0
    for doc in doc_types:
        if not isinstance(doc, dict):
            continue
        fields = doc.get("fields")
        if not isinstance(fields, list):
            continue
        total_calls += sum(1 for fo in fields if isinstance(fo, dict))

    _LOG.info("[环节:栏目标注] 预计调用次数=%s（每个 field 一次）", total_calls)

    first_llm_request_logged = False
    first_llm_response_logged = False

    done = 0
    for doc in doc_types:
        if not isinstance(doc, dict):
            continue
        doc_name = str(doc.get("document_name", "")).strip() or "（未命名文书）"
        fields = doc.get("fields")
        if not isinstance(fields, list):
            _LOG.warning("[环节:跳过] 文书「%s」缺少 fields 数组", doc_name)
            continue

        for field_obj in fields:
            if not isinstance(field_obj, dict):
                continue
            field_name = str(field_obj.get("field_name", "")).strip() or "（未命名字段）"
            done += 1
            content = field_obj.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            review = _field_review_prompt(field_obj)
            if not content.strip():
                _LOG.warning(
                    "[环节:上下文为空] 文书「%s」字段「%s」content 为空，仍将仅按评审要点尝试作答",
                    doc_name,
                    field_name,
                )
            user = build_user_prompt(doc_name, field_name, content, review)
            _LOG.info(
                "[环节:栏目] (%s/%s) 文书=%s 字段=%s 用户消息字符数=%s",
                done,
                total_calls,
                doc_name,
                field_name,
                len(user),
            )
            if dry_run:
                field_obj["answer"] = "[dry-run 未调用大模型]"
                continue
            if not first_llm_request_logged:
                _LOG.info(
                    "[环节:首条LLM请求] system 提示（%s 字符）:\n%s",
                    len(cfg.system_prompt),
                    _preview_for_log(cfg.system_prompt),
                )
                _LOG.info(
                    "[环节:首条LLM请求] user 消息（%s 字符）:\n%s",
                    len(user),
                    _preview_for_log(user),
                )
                first_llm_request_logged = True
            text = call_openai_compatible_chat(cfg, user)
            if not first_llm_response_logged:
                _LOG.info(
                    "[环节:首条LLM响应] 助手回复（%s 字符）:\n%s",
                    len(text),
                    _preview_for_log(text),
                )
                first_llm_response_logged = True
            field_obj["answer"] = text

    if isinstance(meta, dict):
        out["_file_flow_meta"] = meta
    return out


def _resolve_path(raw: Path, cwd: Path) -> Path:
    if raw.is_absolute():
        return raw.resolve()
    for base in (cwd, _ROOT):
        hit = (base / raw).resolve()
        if hit.is_file():
            return hit
    return (_ROOT / raw).resolve()


def run_llm_fill(
    merged: dict[str, Any],
    *,
    workspace: Path,
    cwd: Path,
    input_path: Path | None = None,
    output_path: Path | None = None,
    dry_run: bool = False,
    log_level: str | None = None,
    log_file: Path | None = None,
) -> int:
    """可编程入口：对工作 JSON 填答 ``answer``。"""
    configure_logging(level=log_level, log_file=log_file)

    in_raw = input_path
    if in_raw is None:
        mi = merged.get("file_flow_llm_input")
        if not isinstance(mi, str) or not mi.strip():
            print("错误: 未指定 input_path，或在 pipeline.json 中设置 file_flow_llm_input", file=sys.stderr)
            return 1
        in_raw = Path(mi.strip())
    out_raw = output_path
    if out_raw is None:
        mo = merged.get("file_flow_llm_output")
        if not isinstance(mo, str) or not mo.strip():
            print("错误: 未指定 output_path，或在 pipeline.json 中设置 file_flow_llm_output", file=sys.stderr)
            return 1
        out_raw = Path(mo.strip())

    in_path = _resolve_path(Path(in_raw), cwd)
    out_path = Path(out_raw)
    out_path = out_path.resolve() if out_path.is_absolute() else (cwd / out_path).resolve()

    if not in_path.is_file():
        print(f"错误: 找不到输入: {in_path}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取或解析 JSON: {e}", file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print("错误: JSON 根须为对象", file=sys.stderr)
        return 1

    cfg = load_llm_config_for_file_flow(merged)
    dry = dry_run
    if not dry and (
        not cfg.api_base
        or not cfg.model
        or not is_http_endpoint_url((cfg.api_base or "").strip())
    ):
        _LOG.warning(
            "[环节:配置] 缺少有效的 LLM API URL（http(s) 完整 POST）或 LLM_MODEL，自动 dry-run"
        )
        print(
            "警告: 未配置完整的大模型 endpoint URL 或模型名，仅执行 dry-run。",
            file=sys.stderr,
        )
        dry = True

    try:
        filled = fill_answers(data, cfg, dry_run=dry)
    except RuntimeError as e:
        _LOG.exception("[环节:失败] %s", e)
        return 1

    out_path.write_text(json.dumps(filled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _LOG.info("[环节:写文件] path=%s", out_path.resolve())
    print(f"已写出: {out_path.resolve()}")
    return 0


def _resolve_pipeline_cli(cfg_arg: Path | None) -> Path | None:
    from file_flow.pipeline_merge import repo_root, resolve_pipeline_disk_path  # noqa: PLC0415

    return resolve_pipeline_disk_path(repo_root(), cfg_arg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="工作 JSON 字段级大模型填答（document_types schema）")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="管线 JSON；默认优先 file_flow/pipeline.json",
    )
    ap.add_argument("-i", "--input", type=Path, default=None, help="*_work.json；可改用 pipeline 的 file_flow_llm_input")
    ap.add_argument("-o", "--output", type=Path, default=None, help="写出路径；可改用 pipeline 的 file_flow_llm_output")
    ap.add_argument("--dry-run", action="store_true", help="不请求 API，answer 写占位句")
    ap.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="日志级别；默认 INFO 或环境变量 REVIEW_STANDARD_LLM_LOG_LEVEL",
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="追加写入日志文件（UTF-8）",
    )
    ns = ap.parse_args(argv)

    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(_ROOT)
    configure_logging(level=ns.log_level, log_file=ns.log_file)
    if dotenv_missing:
        _LOG.warning(
            "[环节:环境] 未安装 python-dotenv，已跳过 .env；请 pip install python-dotenv"
        )
    elif env_loaded:
        _LOG.info("[环节:环境] 已加载环节变量文件 %s 个", len(env_loaded))

    cfg_disk = _resolve_pipeline_cli(ns.config)
    merged = load_merged_pipeline_config(cfg_disk if cfg_disk is not None and cfg_disk.is_file() else None)

    return run_llm_fill(
        merged,
        workspace=_ROOT,
        cwd=Path.cwd(),
        input_path=ns.input,
        output_path=ns.output,
        dry_run=ns.dry_run,
        log_level=ns.log_level,
        log_file=ns.log_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
