# -*- coding: utf-8 -*-
"""file_flow/pdf_prepare 装配逻辑单测。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from file_flow.pdf_prepare import build_work_json_from_schema_and_text


def test_build_work_json_sets_question_answer_and_content() -> None:
    schema = {
        "文书甲": {
            "是否必须": "必须",
            "字段": {
                "栏目一": {"要求": ["a", "b"], "内容": ""},
            },
        }
    }
    out = build_work_json_from_schema_and_text(schema, "全文占位")
    assert out["文书甲"]["字段"]["栏目一"]["问题"] == "a\nb"
    assert out["文书甲"]["字段"]["栏目一"]["内容"] == "全文占位"
    assert out["文书甲"]["字段"]["栏目一"]["回答"] == ""


def test_result_json_schema_loads() -> None:
    p = Path(__file__).resolve().parent.parent / "result_json" / "评审标准.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    out = build_work_json_from_schema_and_text(data, "x")
    first_key = next(iter(out))
    fields = out[first_key]["字段"]
    any_field = next(iter(fields.values()))
    assert "问题" in any_field and "回答" in any_field
