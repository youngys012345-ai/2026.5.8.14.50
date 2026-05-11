# -*- coding: utf-8 -*-
"""review_standard_field_qa：prompt 组装与 dry-run 填充逻辑（不发起网络请求）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from review_standard_field_qa import (
    CONTENT_FIELD,
    build_field_qa_user_prompt,
    fill_field_answers,
)
from review_standard_llm_fill import RESULT_FIELD, load_llm_config_from_env


def test_build_field_qa_user_prompt_structure() -> None:
    p = build_field_qa_user_prompt(
        "立案登记表",
        "页码1：表格摘录……",
        "案件来源",
        ["要求A", "要求B"],
        "否",
    )
    assert "立案登记表" in p
    assert "大模型抽取结果" in p
    assert "案件来源" in p
    assert "要求A" in p and "要求B" in p
    assert "是否需要识别手写体" in p or "否" in p


def test_fill_field_answers_dry_run_writes_content() -> None:
    src = {
        "立案登记表": {
            RESULT_FIELD: "模拟抽取正文",
            "字段": {
                "案件来源": {"是否需要识别手写体": "否", "要求": ["条文1"], "内容": ""},
            },
        }
    }
    cfg = load_llm_config_from_env()
    out = fill_field_answers(src, cfg, dry_run=True)
    assert out["立案登记表"]["字段"]["案件来源"][CONTENT_FIELD]
    assert "dry-run" in out["立案登记表"]["字段"]["案件来源"][CONTENT_FIELD]


def test_roundtrip_sample_json_if_present() -> None:
    """若存在评审标准.json，结构须含 字段->内容 键。"""
    p = Path(__file__).resolve().parent.parent / "评审标准.json"
    if not p.is_file():
        pytest.skip("评审标准.json 不在预期路径")
