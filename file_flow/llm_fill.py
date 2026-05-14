#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件模式第二步（**非** schema 全文摘录）：读取工作 JSON（``document_types`` / ``fields`` 结构），
在 **各字段已有 ``content``（上一步摘录或人工填写）** 的前提下，按固定规则拼接用户消息并调用大模型，
将回复写入字段对象下的 ``answer``。

**与第一步的分工**：``schema_llm_extract`` / ``pdf_prepare --llm-extract`` 只负责从 PDF 全文按字段说明写入 ``content``；
本模块 **才** 引入 ``standards.json``：与 ``standards.json`` 按下标对齐，把每条 ``standard`` 作为【评审问题】。

**拼接规则**（与 ``standards.json`` 按下标对齐）：

1. 自顶向下遍历 ``document_types`` → 各文书 ``fields`` 的顺序，为每个字段生成一条请求。
2. 用户消息包含：``【文书名称】``、``【字段名称】``（field_name）、``【字段说明】``（description）、
   ``【与案卷相关的抽取内容】``（content）、``【评审问题】``。
3. ``【评审问题】`` 取 ``standards.json`` **顶层数组**中与该字段**同下标**的元素的 ``standard`` 字段字符串；
   若 standards 条数不足该下标，则评审问题为空并在日志中提示。

**仅依赖**本包内 ``pipeline_config.py``、``pipeline.json``（可选 ``--config``）以及
``llm_openai.py``、``pipeline_merge.py``、``step_dotenv.py``。

``-i`` / ``-o`` 可省略：此时使用 ``pipeline.json`` 的 ``file_flow_llm_input``、``file_flow_llm_output``。

用法（在包含 ``file_flow`` 包的上级目录执行）::

    python -m file_flow.llm_fill -i out/某案_work.json -o out/某案_answered.json
    python -m file_flow.llm_fill --config pipeline.json
    python -m file_flow.llm_fill --dry-run -i ... -o ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
from .pipeline_merge import (
    file_flow_root,
    load_merged_pipeline_config,
    resolve_pipeline_disk_path,
)
from .step_dotenv import ensure_step_dotenv_loaded  # noqa: E402

ensure_step_dotenv_loaded(_FILE_FLOW_DIR)

_LOG = logging.getLogger(__name__)

# 日志中 system、助手回复、各小块标题等单行预览上限（不展开全文摘录）
_LOG_SHORT_PREVIEW = 400


def _clip_one_line(text: str, max_chars: int = _LOG_SHORT_PREVIEW) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def summarize_fill_user_prompt_for_log(user_text: str) -> str:
    """
    仅用于日志：输出 user prompt 各结构化块；「与案卷相关的抽取内容」只报字符数，不展开正文。
    """
    lines: list[str] = ["[环节:user prompt 结构摘要]"]
    segments: tuple[tuple[str, str | None, bool], ...] = (
        ("【文书名称】", "【字段名称】", False),
        ("【字段名称】", "【字段说明】", False),
        ("【字段说明】", "【与案卷相关的抽取内容】", False),
        ("【与案卷相关的抽取内容】", "【评审问题】", True),
        ("【评审问题】", "请根据", False),
    )
    for start_tag, end_tag, omit_body in segments:
        i = user_text.find(start_tag)
        if i == -1:
            continue
        body_start = i + len(start_tag)
        if end_tag:
            j = user_text.find(end_tag, body_start)
            if j == -1:
                body = user_text[body_start:].strip()
            else:
                body = user_text[body_start:j].strip()
        else:
            body = user_text[body_start:].strip()
        if omit_body:
            lines.append(f"  {start_tag} 字符数={len(body)}（正文不在日志中展开）")
        else:
            lines.append(f"  {start_tag} {_clip_one_line(body, 200)}")
    if "请根据" in user_text:
        lines.append("  【固定结尾】请根据「抽取内容」对照「评审问题」…（已省略全文）")
    if len(lines) == 1:
        lines.append(f"  （未能按块解析，总字符数={len(user_text)}）")
    return "\n".join(lines)


def load_standards_standard_by_index(
    merged: dict[str, Any],
    *,
    workspace: Path,
    cwd: Path,
) -> tuple[str, ...]:
    """
    读取 ``file_flow_standards_json``（或默认 ``out/standards_example.json``），
    返回顶层数组各元素 ``standard`` 字段按顺序组成的元组（缺省或非对象则空串）。
    """
    st_raw = merged.get("file_flow_standards_json")
    if isinstance(st_raw, str) and st_raw.strip():
        p = Path(st_raw.strip())
    else:
        p = workspace / "out" / "standards_example.json"
    disk = _resolve_path_standards(p, cwd)
    if not disk.is_file():
        _LOG.warning("[环节:配置] 未找到 standards JSON，评审问题将均为空: %s", disk)
        return ()
    try:
        data = json.loads(disk.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _LOG.warning("[环节:配置] 无法解析 standards JSON %s: %s，评审问题将均为空", disk, e)
        return ()
    if not isinstance(data, list):
        _LOG.warning("[环节:配置] standards JSON 顶层须为数组: %s", disk)
        return ()
    out: list[str] = []
    for x in data:
        if isinstance(x, dict):
            out.append(str(x.get("standard", "")).strip())
        else:
            out.append("")
    return tuple(out)


def _resolve_path_standards(raw: Path, cwd: Path) -> Path:
    if raw.is_absolute():
        return raw.resolve()
    for base in (cwd, _FILE_FLOW_DIR):
        hit = (base / raw).resolve()
        if hit.is_file():
            return hit
    return (_FILE_FLOW_DIR / raw).resolve()


def build_user_prompt(
    document_name: str,
    field_name: str,
    field_description: str,
    extracted_content: str,
    review_question: str,
) -> str:
    """按约定块拼接用户消息。"""
    dn = (document_name or "").strip() or "（未命名文书）"
    fn = (field_name or "").strip() or "（未命名字段）"
    fd = (field_description or "").strip() or "（无）"
    c = extracted_content.strip() if extracted_content else "（抽取内容为空）"
    rq = (review_question or "").strip() or "（无对应评审问题：请检查 standards.json 与字段顺序是否对齐）"
    return (
        f"【文书名称】{dn}\n\n"
        f"【字段名称】{fn}\n\n"
        f"【字段说明】{fd}\n\n"
        f"【与案卷相关的抽取内容】\n{c}\n\n"
        f"【评审问题】\n{rq}\n\n"
        "请根据「抽取内容」对照「评审问题」作答。若内容与问题无关，请说明。"
    )


def fill_answers(
    root: dict[str, Any],
    cfg: LlmEnvConfig,
    *,
    standards_standards: tuple[str, ...] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    """遍历 ``document_types`` → ``fields``，按下标取 ``standards_standards[i]`` 作为【评审问题】，写入 ``answer``。"""
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

    _LOG.info(
        "[环节:栏目标注] 预计调用次数=%s（每字段一次；standards 条数=%s）",
        total_calls,
        len(standards_standards),
    )

    first_llm_request_logged = False
    first_llm_response_logged = False

    done = 0
    field_index = 0
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
            fd_raw = field_obj.get("description")
            field_desc = (
                fd_raw.strip()
                if isinstance(fd_raw, str)
                else (str(fd_raw).strip() if fd_raw is not None else "")
            )
            content = field_obj.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            rq = standards_standards[field_index] if field_index < len(standards_standards) else ""
            if field_index >= len(standards_standards):
                _LOG.warning(
                    "[环节:对齐] 字段序号 %s 超出 standards 条数 %s，评审问题为空（文书=%s 字段=%s）",
                    field_index,
                    len(standards_standards),
                    doc_name,
                    field_name,
                )
            done += 1
            field_index += 1
            if not content.strip():
                _LOG.warning(
                    "[环节:上下文为空] 文书「%s」字段「%s」content 为空，仍将按评审问题尝试作答",
                    doc_name,
                    field_name,
                )
            user = build_user_prompt(doc_name, field_name, field_desc, content, rq)
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
                    "[环节:首条LLM请求] system 提示（%s 字符）摘要:\n%s",
                    len(cfg.system_prompt),
                    _clip_one_line(cfg.system_prompt, _LOG_SHORT_PREVIEW),
                )
                _LOG.info(
                    "[环节:首条LLM请求] user 消息总字符数=%s\n%s",
                    len(user),
                    summarize_fill_user_prompt_for_log(user),
                )
                first_llm_request_logged = True
            text = call_openai_compatible_chat(cfg, user)
            if not first_llm_response_logged:
                _LOG.info(
                    "[环节:首条LLM响应] 助手回复（%s 字符）摘要:\n%s",
                    len(text),
                    _clip_one_line(text, _LOG_SHORT_PREVIEW),
                )
                first_llm_response_logged = True
            field_obj["answer"] = text

    if isinstance(meta, dict):
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

    standards_seq = load_standards_standard_by_index(merged, workspace=workspace, cwd=cwd)

    try:
        filled = fill_answers(data, cfg, standards_standards=standards_seq, dry_run=dry)
    except RuntimeError as e:
        _LOG.exception("[环节:失败] %s", e)
        return 1

    out_path.write_text(json.dumps(filled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _LOG.info("[环节:写文件] path=%s", out_path.resolve())
    print(f"已写出: {out_path.resolve()}")
    return 0


def _resolve_pipeline_cli(cfg_arg: Path | None) -> Path | None:
    return resolve_pipeline_disk_path(file_flow_root(), cfg_arg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="工作 JSON 字段级大模型填答（document_types schema）")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="管线 JSON；默认使用 file_flow 目录下的 pipeline.json",
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

    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(_FILE_FLOW_DIR)
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
        workspace=_FILE_FLOW_DIR,
        cwd=Path.cwd(),
        input_path=ns.input,
        output_path=ns.output,
        dry_run=ns.dry_run,
        log_level=ns.log_level,
        log_file=ns.log_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
