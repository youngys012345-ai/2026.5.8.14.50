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
    work_json = '{"document_types": [{"document_name": "D", "fields": [{"field_name": "f", "content": "摘录A"}]}]}'
    u = build_standards_review_user_prompt(row, work_json, attach_work_json=True)
    assert "程序" in u and "立案" in u and "说明文字" in u
    assert "是否符合时限" in u
    assert "摘录A" in u
    assert "案卷工作 JSON" in u


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
    assert out["standards_review"]["work_json_attached"] is True
    assert out["standards_review"]["digest_attached"] is True


def test_render_html_shows_field_description_under_title() -> None:
    from file_flow.render_html import render_review_html

    html = render_review_html(
        {
            "document_types": [
                {
                    "document_name": "测试文书",
                    "fields": [
                        {
                            "field_name": "field_key",
                            "description": "这是中文说明",
                            "content": "摘录正文",
                        }
                    ],
                }
            ],
        },
        title="t",
    )
    assert "field_key" in html
    assert "字段说明" in html
    assert "这是中文说明" in html
    assert "评审要点" in html
    assert "填答" in html


def test_render_html_standards_review_end_section_requirement_and_result() -> None:
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
    assert "清单评审：评审要求与评审结果" in html
    assert "结论" in html
    assert "评审要求" in html
    assert "评审结果" in html
    assert "评审标准（须对照）" not in html
    assert "评审结论" not in html


def test_render_html_with_standards_review_field_area_has_no_answer_column() -> None:
    """含 standards_review 时字段区不渲染右侧评审要点 / 填答。"""
    from file_flow.render_html import render_review_html

    html = render_review_html(
        {
            "document_types": [
                {
                    "document_name": "文书甲",
                    "fields": [
                        {
                            "field_name": "f1",
                            "description": "说明",
                            "content": "抽取甲",
                            "related_review_items": ["要点A"],
                            "answer": "不应展示",
                        }
                    ],
                }
            ],
            "standards_review": {
                "standards_path": "/p/s.json",
                "items": [{"standard": "问", "review_answer": "答"}],
            },
        },
        title="t",
    )
    assert "与 schema 对应的抽取结果" in html
    assert "抽取甲" in html
    assert "评审要点" not in html
    assert "填答" not in html
    assert "不应展示" not in html
    assert "清单评审：评审要求与评审结果" in html
    assert "答" in html
