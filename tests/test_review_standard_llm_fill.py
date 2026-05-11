# -*- coding: utf-8 -*-
"""review_standard_llm_fill：prompt 拼接与环境配置解析的单元测试（不发起网络请求）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_standard_llm_fill import (
    RESULT_FIELD,
    build_table_extraction_user_prompt,
    fill_review_standard_json,
    load_llm_config_from_env,
)


def test_build_table_extraction_user_prompt_contains_pdf_and_task() -> None:
    p = build_table_extraction_user_prompt("PAGE1\n表格A", "立案登记表")
    assert "PAGE1" in p
    assert "PDF 抽取结果" in p
    assert "立案登记表" in p
    assert "页码和表格内容" in p


def test_build_table_extraction_user_prompt_no_prior_llm_block() -> None:
    """不显式包含前几轮模型摘要区块，仅 PDF + 本轮任务。"""
    p = build_table_extraction_user_prompt("FULL", "现场笔录")
    assert "FULL" in p
    assert "现场笔录" in p
    assert "此前各一级字段" not in p


def test_fill_review_standard_json_dry_run_adds_result_field() -> None:
    src = {
        "立案登记表": {"是否必须": "必须", "字段": {}},
        "现场笔录": {"是否必须": "非必须", "字段": {}},
    }
    cfg = load_llm_config_from_env()
    out = fill_review_standard_json(src, "mdbody", cfg, dry_run=True)
    assert RESULT_FIELD in out["立案登记表"]
    assert RESULT_FIELD in out["现场笔录"]
    assert "dry-run" in out["立案登记表"][RESULT_FIELD]


def test_load_llm_config_from_env_reads_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.com")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    c = load_llm_config_from_env()
    assert c.api_base == "https://example.com"
    assert c.api_key == "k"
    assert c.model == "m"


def test_iter_roundtrip_评审标准_json_format() -> None:
    """根目录 评审标准.json 可被解析且一级块为对象。"""
    p = Path(__file__).resolve().parent.parent / "评审标准.json"
    if not p.is_file():
        pytest.skip("评审标准.json 不在预期路径")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data.get("立案登记表"), dict)
    assert "字段" in data["立案登记表"]
