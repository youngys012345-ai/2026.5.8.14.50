# -*- coding: utf-8 -*-
"""file_flow/schema_llm_extract：纯文本摘录与 document_types 装配单测。"""

from __future__ import annotations

from file_flow.llm_openai import LlmEnvConfig
from file_flow.pdf_prepare import apply_full_text_to_all_contents, build_work_json_skeleton
from file_flow.schema_llm_extract import (
    build_case_extract_user_prompt,
    build_public_context,
    enrich_work_json_with_llm_schema_extract,
    parse_llm_plain_excerpt,
)


def test_parse_llm_plain_excerpt_strips_fence() -> None:
    assert parse_llm_plain_excerpt("```\n摘录一段\n```") == "摘录一段"


def test_build_public_context_and_user_prompt() -> None:
    pub = build_public_context("立案表", "说明文字")
    assert "立案表" in pub and "说明文字" in pub
    u = build_case_extract_user_prompt(pub, "要点甲", "案由", "PDF全文示例")
    assert "要点甲" in u and "PDF全文示例" in u


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


def test_enrich_nested_dry_run_writes_content() -> None:
    work = {
        "document_types": [
            {
                "document_name": "D",
                "description": "文书说明",
                "fields": [
                    {
                        "field_name": "F",
                        "case_sources": [{"description": "要点1"}, {"description": "要点2"}],
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


def test_enrich_document_types_calls_llm_per_field(monkeypatch) -> None:
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
    assert out["document_types"][0]["fields"][0]["content"] == "摘录结果"
    assert out["document_types"][0]["fields"][1]["content"] == "摘录结果"
