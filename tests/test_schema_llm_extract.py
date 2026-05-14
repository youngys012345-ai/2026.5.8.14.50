# -*- coding: utf-8 -*-
"""file_flow/schema_llm_extract：摘录与 document_types 装配单测。"""

from __future__ import annotations

from file_flow.llm_openai import LlmEnvConfig
from file_flow.pdf_prepare import apply_full_text_to_all_contents, build_work_json_skeleton
from file_flow.schema_llm_extract import (
    build_field_extract_user_prompt,
    build_public_context,
    enrich_work_json_with_llm_schema_extract,
    parse_llm_plain_excerpt,
    summarize_schema_extract_user_prompt_for_log,
)


def test_summarize_schema_user_prompt_structure_only() -> None:
    pub = build_public_context("甲文书", "文书说明一行")
    u = build_field_extract_user_prompt(pub, "案由", "填写案由", "X" * 5000)
    s = summarize_schema_extract_user_prompt_for_log(u)
    assert "结构摘要" in s
    assert "5000" in s or "字符数=" in s
    assert "【PDF 全文】" in s
    assert "X" * 100 not in s


def test_parse_llm_plain_excerpt_keeps_markdown_fence() -> None:
    raw = "```\n摘录一段\n```"
    assert parse_llm_plain_excerpt(raw) == raw


def test_build_public_context_and_field_extract_prompt() -> None:
    pub = build_public_context("立案表", "说明文字")
    assert "立案表" in pub and "说明文字" in pub
    u = build_field_extract_user_prompt(pub, "案由", "填写立案案由", "PDF全文示例")
    assert "案由" in u and "填写立案案由" in u and "PDF全文示例" in u


def test_nested_schema_skeleton() -> None:
    schema = {
        "schema_version": "1",
        "document_types": [
            {
                "document_name": "文书甲",
                "description": "文书级说明",
                "fields": [{"field_name": "栏目一", "description": "字段说明"}],
            }
        ],
    }
    w = build_work_json_skeleton(schema)
    f0 = w["document_types"][0]["fields"][0]
    assert f0["content"] == ""
    assert f0.get("answer") == ""


def test_skeleton_then_fulltext() -> None:
    schema = {
        "schema_version": "1",
        "document_types": [
            {
                "document_name": "文书甲",
                "description": "d",
                "fields": [{"field_name": "栏目一", "description": "x"}],
            }
        ],
    }
    w = build_work_json_skeleton(schema)
    assert w["document_types"][0]["fields"][0]["content"] == ""
    apply_full_text_to_all_contents(w, "全文")
    assert w["document_types"][0]["fields"][0]["content"] == "全文"


def test_enrich_one_call_per_field_dry_run() -> None:
    work = {
        "document_types": [
            {
                "document_name": "D",
                "description": "文书说明",
                "fields": [
                    {
                        "field_name": "F",
                        "description": "字段说明",
                    }
                ],
            }
        ]
    }
    cfg = LlmEnvConfig(
        api_base="http://localhost/v1/chat/completions",
        api_keys=(),
        model="dummy",
        timeout_sec=1.0,
        system_prompt="x",
    )
    out = enrich_work_json_with_llm_schema_extract(work, "全文正文", cfg, None, dry_run=True)
    c = out["document_types"][0]["fields"][0]["content"]
    assert "dry-run" in c


def test_enrich_replaces_cfg_system_with_schema_extract_prompt(monkeypatch) -> None:
    """即使传入 llm_fill 风格的评审 system，实际请求也必须换成 schema 摘录专用 system。"""
    captured: dict[str, str] = {}

    def fake_chat(cfg, user_text: str) -> str:
        captured["system"] = cfg.system_prompt
        captured["user_has_review_block"] = "【评审问题】" in user_text
        return "摘录"

    monkeypatch.setattr(
        "file_flow.schema_llm_extract.call_openai_compatible_chat",
        fake_chat,
    )
    work = {
        "document_types": [
            {
                "document_name": "文书",
                "description": "",
                "fields": [{"field_name": "栏A", "description": "da"}],
            }
        ]
    }
    cfg = LlmEnvConfig(
        api_base="http://localhost/v1/chat/completions",
        api_keys=("k",),
        model="m",
        timeout_sec=1.0,
        system_prompt="你是评审助手（错误地传入的 system，应被替换）",
    )
    enrich_work_json_with_llm_schema_extract(work, "全文X", cfg, None, dry_run=False)
    assert "信息抽取" in captured["system"] or "摘录" in captured["system"]
    assert "错误的评审" not in captured["system"]
    assert captured["user_has_review_block"] is False


    calls: list[str] = []

    def fake_chat(_cfg, user_text: str) -> str:
        calls.append(user_text)
        return "摘录结果"

    monkeypatch.setattr(
        "file_flow.schema_llm_extract.call_openai_compatible_chat",
        fake_chat,
    )
    work = {
        "document_types": [
            {
                "document_name": "文书",
                "description": "",
                "fields": [
                    {"field_name": "栏A", "description": "da"},
                    {"field_name": "栏B", "description": "db"},
                ],
            }
        ]
    }
    cfg = LlmEnvConfig(
        api_base="http://localhost/v1/chat/completions",
        api_keys=("k",),
        model="m",
        timeout_sec=1.0,
        system_prompt="sys",
    )
    out = enrich_work_json_with_llm_schema_extract(work, "全文X", cfg, None, dry_run=False)
    assert len(calls) == 2
    assert "栏A" in calls[0] and "da" in calls[0]
    assert "栏B" in calls[1] and "db" in calls[1]
    assert out["document_types"][0]["fields"][0]["content"] == "摘录结果"
    assert out["document_types"][0]["fields"][1]["content"] == "摘录结果"
