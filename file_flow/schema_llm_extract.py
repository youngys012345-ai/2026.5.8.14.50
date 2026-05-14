#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 **document_types** 结构（与 ``out/schema_example.json`` 一致）调用大模型，
从 PDF 全文中为各 ``fields`` 项摘录匹配内容，写入字段对象下的 ``content``。

**本环节只做「从 PDF 正文按 schema 索引做信息摘录」**，不读取 ``standards.json``、不在 user 消息中拼接
任何「评审问题 / standard」。清单级评审见 ``standards_llm_review``（读取已填 ``content`` 的整份工作 JSON）。

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

import json
import logging
import os
from dataclasses import replace
from typing import Any

from .llm_openai import (
    LlmEnvConfig,
    call_openai_compatible_chat,
)

_LOG = logging.getLogger(__name__)


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
