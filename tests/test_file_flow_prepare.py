# -*- coding: utf-8 -*-
"""file_flow/pdf_prepare 装配逻辑单测（document_types schema）。"""

from __future__ import annotations

import json
from pathlib import Path

from file_flow.pdf_prepare import build_work_json_from_schema_and_text, is_document_types_schema


def _minimal_document_types_schema() -> dict:
    return {
        "schema_version": "1",
        "standard_version": "1",
        "case_type": "t",
        "description": "root",
        "document_types": [
            {
                "document_name": "文书甲",
                "document_type": "t1",
                "description": "doc",
                "required": True,
                "related_review_items": [],
                "fields": [
                    {
                        "field_name": "栏目一",
                        "description": "fd",
                        "data_type": "string",
                        "required": True,
                        "related_review_items": [],
                    }
                ],
            }
        ],
        "covered_review_items": [],
        "required_documents": [],
        "created_at": "",
    }


def test_build_work_json_sets_content_and_answer_placeholders() -> None:
    schema = _minimal_document_types_schema()
    out = build_work_json_from_schema_and_text(schema, "全文占位")
    f0 = out["document_types"][0]["fields"][0]
    assert f0["content"] == "全文占位"
    assert f0.get("answer") == ""


def test_schema_example_json_loads() -> None:
    p = Path(__file__).resolve().parent.parent / "file_flow" / "out" / "schema_example.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert is_document_types_schema(data)
    out = build_work_json_from_schema_and_text(data, "x")
    assert "document_types" in out
    any_field = out["document_types"][0]["fields"][0]
    assert "content" in any_field and "answer" in any_field
