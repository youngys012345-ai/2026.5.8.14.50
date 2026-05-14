#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 **document_types** 结构（与 ``out/schema_example.json`` 一致）调用大模型，
从 PDF 全文中为各 ``fields`` 项摘录匹配内容，写入字段对象下的 ``content``。

**本环节只做「从 PDF 正文按 schema 索引做信息摘录」**，不读取 ``standards.json``、不在 user 消息中拼接
任何「评审问题 / standard」。清单级评审见 ``standards_llm_review``（读取已填 ``content`` 的整份工作 JSON）。

可由 ``pdf_prepare`` 在 ``file_flow_llm_extract=true`` 且步骤列表**不含** ``schema_llm_extract`` 时**内联**调用；
也可作为 ``pipeline_merge`` 的独立步骤 ``schema_llm_extract`` 运行（见 ``run_schema_llm_extract``、``python -m file_flow.schema_llm_extract``）。

1. **公共上下文**：每项文书将 ``document_name`` 与文书级 ``description`` 拼成固定前缀。
2. **逐字段一次调用**：每条字段仅依据 **字段名称（field_name）** 与 **字段说明（description）** 作为抽取目标，
   与全文一并交给模型；**不**使用 ``case_sources`` / ``related_review_items`` 等扩展分支。
3. **写回**：将模型返回正文经 ``parse_llm_plain_excerpt``（仅首尾空白 trim，**不**去除 Markdown 代码围栏）
   写入该字段 ``content``。

大模型 **system** 使用 ``load_schema_extract_system_prompt``（或环境变量 ``FILE_FLOW_SCHEMA_EXTRACT_SYSTEM_PROMPT``），
与 ``FILE_FLOW_SYSTEM_PROMPT`` / ``file_flow_llm_system_prompt``（历史字段级填答用）相互独立。

根对象**必须**包含非空的 ``document_types`` 数组；否则 ``pdf_prepare`` 会在加载阶段报错退出。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .llm_openai import (
    LlmEnvConfig,
    call_openai_compatible_chat,
)

_LOG = logging.getLogger(__name__)

_FILE_FLOW_DIR = Path(__file__).resolve().parent


def _log_clip(s: str, max_chars: int = 200) -> str:
    t = (s or "").replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def summarize_schema_extract_user_prompt_for_log(user_text: str) -> str:
    """
    仅用于日志：按实际构造的 user 串解析各块；「【PDF 全文】」之后只报字符数，不展开正文。
    """
    lines: list[str] = [
        f"[环节:schema 抽取 user prompt 结构摘要] 用户消息总字符数={len(user_text)}",
    ]
    fn_mark = "【字段名称】"
    fd_mark = "【字段说明】"
    pdf_mark = "【PDF 全文】"

    i_fn = user_text.find(fn_mark)
    if i_fn == -1:
        lines.append("  （未能解析：缺少【字段名称】）")
        return "\n".join(lines)

    prefix = user_text[:i_fn].strip()
    lines.append(f"  ├─ 文书级前缀（【字段名称】之前）: {_log_clip(prefix, 240)}")

    start_fn = i_fn + len(fn_mark)
    i_fd = user_text.find(fd_mark, start_fn)
    if i_fd == -1:
        lines.append("  ├─ （缺少【字段说明】）")
        return "\n".join(lines)
    body_fn = user_text[start_fn:i_fd].strip()
    lines.append(f"  ├─ 【字段名称】 {_log_clip(body_fn, 160)}")

    start_fd = i_fd + len(fd_mark)
    i_pdf = user_text.find(pdf_mark, start_fd)
    if i_pdf == -1:
        lines.append("  ├─ （缺少【PDF 全文】）")
        return "\n".join(lines)
    body_fd = user_text[start_fd:i_pdf].strip()
    lines.append(f"  ├─ 【字段说明】 {_log_clip(body_fd, 240)}")

    start_pdf = i_pdf + len(pdf_mark)
    body_pdf = user_text[start_pdf:].strip()
    lines.append(f"  └─ 【PDF 全文】之后正文 字符数={len(body_pdf)}（不在日志中展开）")
    return "\n".join(lines)


def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


# 单次请求中全文上限，避免超出常见上下文窗口（可按需调大）
_MAX_FULLTEXT_CHARS = 9999999

DEFAULT_SCHEMA_EXTRACT_SYSTEM = (
    "你是案卷材料**信息抽取**模型（本步只做摘录，不做评审、不答题、不下结论）。\n"
    "输入中会给出 schema 定义的抽取目标索引：文书名称、文书说明、字段名称、字段说明，以及从 PDF 解析得到的正文。\n"
    "你的唯一任务：仅根据上述索引在正文中**检索并摘录**与索引语义直接相关的**全部**原文片段"
    "（可多条、可含表格/列表的转写文字；保持与原文一致的表述）。\n"
    "严禁在本步进行「是否符合规范」「是否通过评审」等判断；严禁回答评审类问题；这些由后续环节处理。\n"
    "若正文不存在相关内容，只输出空字符串，不得编造。\n"
    "只输出摘录正文本身：不要输出 JSON，不要写前言/结语/小标题式的任务复述。"
)


def _merged_str(merged: dict[str, Any] | None, key: str) -> str | None:
    if not merged:
        return None
    v = merged.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def load_schema_extract_system_prompt(merged: dict[str, Any] | None = None) -> str:
    """
    schema 摘录环节的**指令文本**（并入单条 ``user`` 消息首部；不单独发 Chat Completions 的 ``system``）。
    来源：环境变量 ``FILE_FLOW_SCHEMA_EXTRACT_SYSTEM_PROMPT`` / ``pipeline.json`` 的
    ``file_flow_schema_extract_system_prompt`` / 内置默认。
    """
    m = merged or {}
    return (
        _env_first("FILE_FLOW_SCHEMA_EXTRACT_SYSTEM_PROMPT")
        or _merged_str(m, "file_flow_schema_extract_system_prompt")
        or DEFAULT_SCHEMA_EXTRACT_SYSTEM
    )


def _clip_full_text(full_text: str) -> str:
    if len(full_text) <= _MAX_FULLTEXT_CHARS:
        return full_text
    return (
        full_text[:_MAX_FULLTEXT_CHARS]
        + f"\n\n（全文已截断至前 {_MAX_FULLTEXT_CHARS} 字符，原共 {len(full_text)} 字符）"
    )


def parse_llm_plain_excerpt(raw: str) -> str:
    """模型返回的正文：仅首尾 strip，不改动内部 Markdown（含代码围栏）。"""
    return (raw or "").strip()


def build_public_context(document_name: str, document_description: str) -> str:
    """文书级抽取目标：schema 中文书名称 + 文书级 description（与 JSON 字段一致）。"""
    name = (document_name or "").strip() or "（未命名文书）"
    desc = (document_description or "").strip()
    if desc:
        return f"【文书名称】{name}\n【文书说明】{desc}"
    return f"【文书名称】{name}\n【文书说明】（无）"


def build_field_extract_user_prompt(
    public_context: str,
    field_name: str,
    field_description: str,
    full_text: str,
) -> str:
    """
    单字段抽取：显式列出 schema 抽取目标索引 + PDF 正文；指令为「摘录」而非「作答」。
    """
    fn = (field_name or "").strip() or "（未命名字段）"
    fd = (field_description or "").strip() or "（无）"
    return (
        "以下为 schema 定义的**抽取目标索引**（请仅据此在文末 PDF 正文中做客观摘录；"
        "本步不做评审、不回答评审问题、不下合规结论）：\n\n"
        f"{public_context}\n"
        f"【字段名称】{fn}\n"
        f"【字段说明】{fd}\n\n"
        "【抽取任务】\n"
        "在【PDF 全文】中检索并摘录**所有**与上述文书名称、文书说明、字段名称、字段说明在语义上直接相关的原文。"
        "可输出多条摘录，条间可用换行分隔；不要评价材料好坏。\n\n"
        "【输出格式】\n"
        "仅输出摘录到的正文；若全文无任何匹配内容，只输出一个空字符串，不要输出「无」「未找到」等说明性句子。"
        "不要输出 JSON，不要复述本任务书全文。\n\n"
        f"【PDF 全文】\n{_clip_full_text(full_text)}"
    )


def build_schema_extract_full_user_prompt(
    merged: dict[str, Any] | None,
    public_context: str,
    field_name: str,
    field_description: str,
    full_text: str,
) -> str:
    """环节指令 + 结构化抽取请求 + PDF 全文，合并为**一条**将发给模型的 user 正文。"""
    head = load_schema_extract_system_prompt(merged).strip()
    body = build_field_extract_user_prompt(public_context, field_name, field_description, full_text)
    if not head:
        return body
    return f"{head}\n\n---\n\n{body}"


def _field_display_name(field_obj: dict[str, Any]) -> str:
    fn = field_obj.get("field_name")
    if isinstance(fn, str) and fn.strip():
        return fn.strip()
    return "（未命名字段）"


def enrich_work_json_with_llm_schema_extract(
    work: dict[str, Any],
    pdf_full_text: str,
    base_cfg: LlmEnvConfig,
    merged: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    深拷贝 ``work``，按「公共上下文 + 字段名称 + 字段说明」每字段调用大模型一次，将摘录写入 ``content``。

    ``base_cfg`` 仅提供连接参数；``LlmEnvConfig.system_prompt`` 应为空，环节指令由
    ``build_schema_extract_full_user_prompt`` 并入单条 ``user``。
    """
    out: dict[str, Any] = json.loads(json.dumps(work, ensure_ascii=False))
    cfg = replace(base_cfg, system_prompt="")

    docs = out.get("document_types")
    if not isinstance(docs, list) or not docs:
        _LOG.error("[环节:抽取] 缺少非空 document_types，已跳过抽取")
        return out

    total = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        fields = doc.get("fields")
        if not isinstance(fields, list):
            continue
        total += sum(1 for f in fields if isinstance(f, dict))

    done = 0
    _LOG.info("[环节:抽取] 预计 LLM 调用次数=%s（每字段一次，依据 field_name + description）", total)

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_name = str(doc.get("document_name", "")).strip()
        doc_desc = str(doc.get("description", "")).strip()
        public = build_public_context(doc_name, doc_desc)
        fields = doc.get("fields")
        if not isinstance(fields, list):
            _LOG.warning("[环节:抽取] 文书「%s」缺少 fields 数组，已跳过", doc_name or "?")
            continue

        for field in fields:
            if not isinstance(field, dict):
                continue
            label = _field_display_name(field)
            desc = str(field.get("description", "")).strip()
            done += 1
            full_user = build_schema_extract_full_user_prompt(merged, public, label, desc, pdf_full_text)
            _LOG.info(
                "[环节:抽取] (%s/%s) 文书=%s 字段=%s\n%s",
                done,
                total,
                doc_name or "?",
                label,
                summarize_schema_extract_user_prompt_for_log(full_user),
            )
            if dry_run:
                field["content"] = "[dry-run 未调用大模型抽取]"
            else:
                try:
                    raw_reply = call_openai_compatible_chat(cfg, full_user)
                except RuntimeError:
                    _LOG.exception("[环节:抽取] 文书「%s」字段「%s」API 失败", doc_name, label)
                    raise
                field["content"] = parse_llm_plain_excerpt(raw_reply)

    return out


def fulltext_path_for_work_json(work_path: Path, merged: dict[str, Any]) -> Path:
    """
    与 ``pdf_prepare`` 约定一致：若工作文件为 ``{{案卷名}}{{suffix_work}}.json``，
    则全文为同目录 ``{{案卷名}}_fulltext.txt``。
    """
    from .naming import stem_base_from_stage_stem

    base = stem_base_from_stage_stem(work_path.stem, merged)
    return work_path.parent / f"{base}_fulltext.txt"


def _resolve_disk_path(raw: Path, cwd: Path, workspace: Path | None = None) -> Path:
    if raw.is_absolute():
        return raw.resolve()
    bases: list[Path] = [cwd, _FILE_FLOW_DIR]
    if workspace is not None:
        bases.insert(0, workspace)
    for base in bases:
        hit = (base / raw).resolve()
        if hit.is_file():
            return hit
    return (_FILE_FLOW_DIR / raw).resolve()


def run_schema_llm_extract(
    merged: dict[str, Any],
    *,
    workspace: Path,
    cwd: Path,
    work_input: Path | None = None,
    output_path: Path | None = None,
    dry_run: bool = False,
    log_level: str | None = None,
    log_file: Path | None = None,
) -> int:
    """
    对已存在的 ``*_work.json`` 与同目录 ``*_fulltext.txt`` 调用大模型，按 schema 写入各字段 ``content``。

    供管线步骤 ``schema_llm_extract`` 使用；与 ``pdf_prepare`` 内联摘录二选一（由 ``pipeline_merge`` 控制）。
    ``file_flow_llm_extract`` 为关时直接返回 0（不写回、不调 API）。
    """
    from .llm_openai import build_llm_env_config, configure_logging, is_http_endpoint_url
    from .pdf_prepare import resolve_llm_extract_enabled

    configure_logging(level=log_level, log_file=log_file)

    if not resolve_llm_extract_enabled(merged, None):
        _LOG.info("[环节:schema_llm_extract] file_flow_llm_extract 已关闭，跳过")
        return 0

    in_raw = work_input
    if in_raw is None:
        v = merged.get("file_flow_schema_extract_work_input")
        if isinstance(v, str) and v.strip():
            in_raw = Path(v.strip())
    if in_raw is None:
        print(
            "错误: 未指定工作 JSON，请设置 file_flow_schema_extract_work_input 或使用 -i/--work-input",
            file=sys.stderr,
        )
        return 1

    work_path = _resolve_disk_path(Path(in_raw), cwd, workspace)
    if not work_path.is_file():
        print(f"错误: 找不到工作 JSON: {work_path}", file=sys.stderr)
        return 1

    ft_path = fulltext_path_for_work_json(work_path, merged)
    if not ft_path.is_file():
        print(f"错误: 找不到全文文件（应与 work 同目录）: {ft_path}", file=sys.stderr)
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
        text = ft_path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"错误: 无法读取全文: {e}", file=sys.stderr)
        return 1

    base_cfg = build_llm_env_config(merged, "")
    dry = dry_run
    if not dry and (
        not base_cfg.api_base
        or not base_cfg.model
        or not is_http_endpoint_url((base_cfg.api_base or "").strip())
    ):
        print("警告: 大模型抽取未配置有效 URL/模型，改为 dry-run 占位。", file=sys.stderr)
        dry = True

    try:
        work = enrich_work_json_with_llm_schema_extract(work, text, base_cfg, merged, dry_run=dry)
    except RuntimeError as e:
        _LOG.exception("[环节:schema_llm_extract] 失败: %s", e)
        print(f"错误: {e}", file=sys.stderr)
        return 1

    meta = work.get("_file_flow_meta")
    if isinstance(meta, dict):
        meta = dict(meta)
    else:
        meta = {}
    meta["内容填充模式"] = "llm_schema_extract_standalone_dry_run" if dry else "llm_schema_extract_standalone"
    meta["file_flow_schema_extract_fulltext_file"] = ft_path.name
    work["_file_flow_meta"] = meta

    out_raw = output_path
    if out_raw is None:
        out_raw = work_path
    out_path = Path(out_raw)
    out_path = out_path.resolve() if out_path.is_absolute() else (cwd / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(work, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _LOG.info("[环节:写文件] schema 摘录已写回 path=%s", out_path)
    print(f"已写出: {out_path.resolve()}")
    return 0


def _resolve_pipeline_cli(cfg_arg: Path | None) -> Path | None:
    from .pipeline_merge import file_flow_root, resolve_pipeline_disk_path

    return resolve_pipeline_disk_path(file_flow_root(), cfg_arg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="对已生成的 *_work.json 与同目录全文做大模型 schema 摘录")
    ap.add_argument("--config", type=Path, default=None, help="管线 JSON；默认 file_flow/pipeline.json")
    ap.add_argument("-i", "--work-input", type=Path, default=None, help="*_work.json 路径")
    ap.add_argument("-o", "--output", type=Path, default=None, help="输出路径；默认覆盖输入 work 文件")
    ap.add_argument("--dry-run", action="store_true", help="不调 API，content 写占位")
    ap.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="日志级别",
    )
    ap.add_argument("--log-file", type=Path, default=None, help="追加日志 UTF-8")
    ns = ap.parse_args(argv)

    from .llm_openai import configure_logging
    from .pipeline_merge import load_merged_pipeline_config
    from .step_dotenv import ensure_step_dotenv_loaded

    configure_logging(level=ns.log_level, log_file=ns.log_file)

    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(_FILE_FLOW_DIR)
    if dotenv_missing:
        _LOG.warning("[环节:环境] 未安装 python-dotenv，已跳过 .env")
    elif env_loaded:
        _LOG.info("[环节:环境] 已加载环境文件 %s 个", len(env_loaded))

    cfg_disk = _resolve_pipeline_cli(ns.config)
    merged = load_merged_pipeline_config(cfg_disk if cfg_disk is not None and cfg_disk.is_file() else None)

    return run_schema_llm_extract(
        merged,
        workspace=_FILE_FLOW_DIR,
        cwd=Path.cwd(),
        work_input=ns.work_input,
        output_path=ns.output,
        dry_run=ns.dry_run,
        log_level=ns.log_level,
        log_file=ns.log_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
