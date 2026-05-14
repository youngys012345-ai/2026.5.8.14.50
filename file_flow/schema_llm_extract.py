#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 **document_types** 结构（与 ``file_flow/out/schema_example.json`` 一致）调用大模型，
从 PDF 全文中为各 ``fields`` 项摘录匹配内容，写入字段对象下的 ``content``。

1. **公共上下文**：每项文书将 ``document_name`` 与文书级 ``description`` 拼成固定前缀。
2. **逐条要点调用**：字段上可选 ``case_sources`` / ``case_source``（元素含 ``description``）；
   若未配置，则使用该字段的 ``description`` 作为唯一一条抽取要点。
3. **写回**：多条要点时合并为一段文本写入 ``content``（纯 schema 扩展键，与示例中
   ``field_name`` / ``description`` 等并列）。

根对象**必须**包含非空的 ``document_types`` 数组；否则 ``pdf_prepare`` 会在加载阶段报错退出。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import replace
from typing import Any

from file_flow.llm_openai import (
    LlmEnvConfig,
    call_openai_compatible_chat,
)

_LOG = logging.getLogger(__name__)


def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


# 单次请求中全文上限，避免超出常见上下文窗口（可按需调大）
_MAX_FULLTEXT_CHARS = 9999999

DEFAULT_SCHEMA_EXTRACT_SYSTEM = (
    "你是行政执法案卷信息抽取助手。用户会提供：文书类公共上下文、本条需要对照的抽取要点说明、"
    "以及从 PDF 解析得到的全文。"
    "请**仅从全文**中摘录与「抽取要点」直接相关的原文片段（可含多条、可含表格转写文字）；"
    "若文中找不到相关内容，必须只输出空字符串，不要编造。"
    "只输出抽取要点本身的内容：不要输出 JSON、不要 Markdown 代码围栏、不要前后解释。"
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
    """抽取环节专用 system：环境变量 / pipeline 的 file_flow_schema_extract_system_prompt。"""
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
    """从模型回复中取出纯摘录：去掉常见 Markdown 代码围栏。"""
    text = raw.strip()
    m = re.search(r"```(?:[a-z]*)?\s*([\s\S]*?)\s*```", text, flags=re.I)
    if m:
        text = m.group(1).strip()
    return text.strip()


def build_public_context(document_name: str, document_description: str) -> str:
    """文书级公共上下文：文书名称 + 文书说明。"""
    name = (document_name or "").strip() or "（未命名文书）"
    desc = (document_description or "").strip()
    if desc:
        return f"【文书名称】{name}\n【文书说明】{desc}"
    return f"【文书名称】{name}\n【文书说明】（无）"


def build_case_extract_user_prompt(
    public_context: str,
    case_description: str,
    field_label: str,
    full_text: str,
) -> str:
    """单条抽取要点：公共上下文 + 本条要点 + 全文。"""
    case = (case_description or "").strip() or "（未配置抽取要点）"
    return (
        f"{public_context}\n\n"
        f"【字段名称】{field_label}\n\n"
        f"【本条抽取要点】\n{case}\n\n"
        f"【PDF 全文】\n{_clip_full_text(full_text)}\n\n"
        "请只输出从【PDF 全文】中与「本条抽取要点」相关的原文摘录；无则输出空字符串。"
    )


def _normalize_case_sources(field_obj: dict[str, Any]) -> list[dict[str, Any]]:
    """
    得到抽取要点列表；元素为含 ``description`` 的字典。
    优先 ``case_sources`` / ``case_source``；否则使用字段 ``description``。
    """
    raw = field_obj.get("case_sources")
    if raw is None:
        raw = field_obj.get("case_source")
    out: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, dict) and str(x.get("description", "")).strip():
                out.append(dict(x))
    elif isinstance(raw, dict) and str(raw.get("description", "")).strip():
        out.append(dict(raw))
    if out:
        return out
    desc = str(field_obj.get("description", "")).strip()
    if desc:
        return [{"description": desc}]
    return [
        {
            "description": (
                "（未配置 description 与 case_sources，请结合全文与 field_name 尽量摘录相关原文）"
            )
        }
    ]


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
    深拷贝 ``work``，按「公共上下文 + 每条抽取要点」调用大模型，将摘录写入各字段 ``content``。
    """
    out: dict[str, Any] = json.loads(json.dumps(work, ensure_ascii=False))
    extract_sys = load_schema_extract_system_prompt(merged)
    cfg = replace(base_cfg, system_prompt=extract_sys)

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
        for field in fields:
            if isinstance(field, dict):
                total += max(1, len(_normalize_case_sources(field)))

    done = 0
    _LOG.info("[环节:抽取] 预计 LLM 调用次数=%s（每条抽取要点一次）", total)

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
            sources = _normalize_case_sources(field)
            parts: list[str] = []
            for idx, cs in enumerate(sources, start=1):
                done += 1
                desc = str(cs.get("description", "")).strip()
                user = build_case_extract_user_prompt(public, desc, label, pdf_full_text)
                _LOG.info(
                    "[环节:抽取] (%s/%s) 文书=%s 字段=%s 要点序号=%s 用户消息字符数=%s",
                    done,
                    total,
                    doc_name or "?",
                    label,
                    idx,
                    len(user),
                )
                if dry_run:
                    excerpt = "[dry-run 未调用大模型抽取]"
                else:
                    try:
                        raw_reply = call_openai_compatible_chat(cfg, user)
                    except RuntimeError:
                        _LOG.exception("[环节:抽取] 文书「%s」字段「%s」API 失败", doc_name, label)
                        raise
                    excerpt = parse_llm_plain_excerpt(raw_reply)
                if len(sources) > 1:
                    parts.append(f"【要点{idx}】\n{excerpt}")
                else:
                    parts.append(excerpt)
            merged_text = "\n\n".join(parts) if len(sources) > 1 else (parts[0] if parts else "")
            field["content"] = merged_text

    return out
