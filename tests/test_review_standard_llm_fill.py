# -*- coding: utf-8 -*-
"""review_standard_llm_fill：prompt 拼接与环境配置解析的单元测试（不发起网络请求）。"""

from __future__ import annotations

import json
from pathlib import Path

from unittest.mock import MagicMock, patch

import pytest

from review_standard_llm_fill import (
    RESULT_FIELD,
    LlmEnvConfig,
    build_table_extraction_user_prompt,
    call_openai_compatible_chat,
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
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.com/v1/chat/completions")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    c = load_llm_config_from_env()
    assert c.api_base == "https://example.com/v1/chat/completions"
    assert c.api_key == "k"
    assert c.model == "m"


def test_load_llm_config_from_env_llm_api_base_wins_over_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_API_BASE 优先于 OPENAI_API_BASE。"""
    monkeypatch.setenv("LLM_API_BASE", "https://primary/v1/chat/completions")
    monkeypatch.setenv("OPENAI_API_BASE", "https://fallback/v1/chat/completions")
    monkeypatch.setenv("LLM_MODEL", "m")
    c = load_llm_config_from_env()
    assert c.api_base == "https://primary/v1/chat/completions"


def test_load_llm_config_from_env_collects_backup_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_BASE", "https://x/v1/chat/completions")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "primary")
    monkeypatch.setenv("LLM_API_KEY_BACKUP1", "b1")
    monkeypatch.setenv("LLM_API_KEY_BACKUP2", "b2")
    c = load_llm_config_from_env()
    assert c.api_keys == ("primary", "b1", "b2")
    assert c.api_key == "primary"


def test_load_llm_config_from_env_backup_keys_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_BASE", "https://x/v1/chat/completions")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "same")
    monkeypatch.setenv("LLM_API_KEY_BACKUP1", "same")
    monkeypatch.setenv("LLM_API_KEY_BACKUP2", "other")
    c = load_llm_config_from_env()
    assert c.api_keys == ("same", "other")


def test_call_openai_compatible_chat_retries_on_429_with_next_key() -> None:
    """主密钥 429 后自动用备用密钥重试并成功。"""
    from io import BytesIO

    import urllib.error

    ok_body = json.dumps(
        {"choices": [{"message": {"content": "  成功  "}}]},
        ensure_ascii=False,
    ).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        auth = req.headers.get("Authorization", "")
        if "primary" in auth:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many", hdrs={}, fp=BytesIO(b'{"error":"rate"}')
            )
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value.read.return_value = ok_body
        mock_cm.__exit__.return_value = None
        return mock_cm

    cfg = LlmEnvConfig(
        api_base="https://api.example/v1/chat/completions",
        api_keys=("primary", "backup-ok"),
        model="gpt-test",
        timeout_sec=30.0,
        system_prompt="sys",
    )
    with patch("review_standard_llm_fill.urllib.request.urlopen", side_effect=fake_urlopen):
        text = call_openai_compatible_chat(cfg, "user hello")
    assert text == "成功"


def test_iter_roundtrip_评审标准_json_format() -> None:
    """根目录 评审标准.json 可被解析且一级块为对象。"""
    p = Path(__file__).resolve().parent.parent / "评审标准.json"
    if not p.is_file():
        pytest.skip("评审标准.json 不在预期路径")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data.get("立案登记表"), dict)
    assert "字段" in data["立案登记表"]
