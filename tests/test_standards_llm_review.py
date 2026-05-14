# -*- coding: utf-8 -*-
"""file_flow/standards_llm_review 提示词与 dry-run 合并单测。"""

from __future__ import annotations

from pathlib import Path

from file_flow.llm_openai import LlmEnvConfig
from file_flow.standards_llm_review import (
    build_standards_review_user_prompt,
    run_standards_llm_review_on_data,
)


def test_build_user_prompt_has_context_and_standard() -> None:
    row = {
        "category": "程序",
        "subcategory": "立案",
        "content": "说明文字",
        "standard": "是否符合时限要求？",
        "score": "1",
        "penalty": "0",
        "number": True,
    }
    u = build_standards_review_user_prompt(row, "摘录A", attach_digest=True)
    assert "程序" in u and "立案" in u and "说明文字" in u
    assert "是否符合时限" in u
    assert "摘录A" in u


def test_run_on_data_dry_run_adds_standards_review() -> None:
    work = {
        "schema_version": "1",
        "document_types": [
            {
                "document_name": "D",
                "fields": [{"field_name": "f", "content": "摘录"}],
            }
        ],
    }
    standards = [
        {"category": "c", "subcategory": "s", "content": "", "standard": "问？", "penalty": "p"},
    ]
    cfg = LlmEnvConfig(
        api_base="http://localhost/v1/chat/completions",
        api_keys=(),
        model="m",
        timeout_sec=1.0,
        system_prompt="sys",
    )
    out = run_standards_llm_review_on_data(
        work,
        standards,
        Path("standards.json"),
        cfg,
        {"file_flow_review_attach_schema_digest": True},
        dry_run=True,
    )
    assert "document_types" in out
    assert "standards_review" in out
    assert out["standards_review"]["items"][0]["review_answer"] == "[dry-run 未调用大模型]"
    assert "digest_attached" in out["standards_review"]


def test_render_html_includes_standards_section() -> None:
    from file_flow.render_html import render_review_html

    html = render_review_html(
        {
            "document_types": [],
            "standards_review": {
                "standards_path": "/x/s.json",
                "items": [
                    {
                        "category": "a",
                        "subcategory": "b",
                        "content": "c",
                        "standard": "d?",
                        "review_answer": "结论",
                    }
                ],
            },
        },
        title="t",
    )
    assert "按 standards 清单评审" in html
    assert "结论" in html
