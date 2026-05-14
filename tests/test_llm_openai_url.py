# -*- coding: utf-8 -*-
"""file_flow.llm_openai：Chat Completions URL 规范化（避免仅配置 /v1 时 POST 到错误路径导致 404）。"""

from __future__ import annotations

import pytest

from file_flow.llm_openai import normalize_llm_chat_completions_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"),
        ("https://api.openai.com/v1/", "https://api.openai.com/v1/chat/completions"),
        ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/chat/completions"),
        (
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1/chat/completions",
        ),
        (
            "https://api.openai.com/v1/chat/completions/",
            "https://api.openai.com/v1/chat/completions",
        ),
        ("https://api.example.com", "https://api.example.com/v1/chat/completions"),
        ("https://api.example.com/", "https://api.example.com/v1/chat/completions"),
    ],
)
def test_normalize_llm_chat_completions_url(raw: str, expected: str) -> None:
    assert normalize_llm_chat_completions_url(raw) == expected


def test_normalize_preserves_nonstandard_path() -> None:
    """显式完整路径（非 …/v1 结尾）保持原样，避免破坏自建网关。"""
    u = "https://gateway.local/custom/openai/chat/completions"
    assert normalize_llm_chat_completions_url(u) == u
