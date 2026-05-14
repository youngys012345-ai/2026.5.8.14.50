# -*- coding: utf-8 -*-
"""file_flow.llm_openai：限流时多密钥轮换。"""

from __future__ import annotations

import io
import urllib.error

import pytest

from file_flow.llm_openai import (
    LlmEnvConfig,
    _should_rotate_llm_api_key,
    call_openai_compatible_chat,
)


def test_should_rotate_on_429_and_503() -> None:
    assert _should_rotate_llm_api_key(429, "{}") is True
    assert _should_rotate_llm_api_key(503, "") is True
    assert _should_rotate_llm_api_key(502, "bad gateway") is True


def test_should_rotate_403_even_with_empty_body() -> None:
    """QPM 场景下部分网关固定返回 403，响应体可能为空。"""
    assert _should_rotate_llm_api_key(403, "") is True
    assert _should_rotate_llm_api_key(403, "{}") is True


def test_should_rotate_400_when_body_suggests_qpm() -> None:
    assert _should_rotate_llm_api_key(400, '{"message":"QPM limit exceeded"}') is True
    assert _should_rotate_llm_api_key(403, "请求频率过高") is True


def test_should_not_rotate_400_on_unrelated_error() -> None:
    assert _should_rotate_llm_api_key(400, '{"error":"invalid model name"}') is False


def test_call_rotates_from_primary_to_backup_on_403_empty_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("file_flow.llm_openai.time.sleep", lambda _s: None)
    tokens_seen: list[str | None] = []

    def fake_post(
        url: str,
        model: str,
        user_message: str,
        api_key_token: str | None,
        timeout_sec: float,
        **kwargs: object,
    ) -> str:
        tokens_seen.append(api_key_token)
        if api_key_token == "k0":
            raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs={}, fp=io.BytesIO(b""))
        return "ok"

    monkeypatch.setattr("file_flow.llm_openai._post_chat_completion_once", fake_post)
    cfg = LlmEnvConfig(
        api_base="http://127.0.0.1/v1/chat/completions",
        api_keys=("k0", "k1"),
        model="m",
        timeout_sec=1.0,
        system_prompt="",
    )
    assert call_openai_compatible_chat(cfg, "hi") == "ok"
    assert tokens_seen == ["k0", "k1"]


def test_call_rotates_from_primary_to_backup_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("file_flow.llm_openai.time.sleep", lambda _s: None)
    tokens_seen: list[str | None] = []

    def fake_post(
        url: str,
        model: str,
        user_message: str,
        api_key_token: str | None,
        timeout_sec: float,
        **kwargs: object,
    ) -> str:
        tokens_seen.append(api_key_token)
        if api_key_token == "k0":
            raise urllib.error.HTTPError(
                url,
                429,
                "Too Many",
                hdrs={},
                fp=io.BytesIO(b'{"error":"rate_limit"}'),
            )
        return "ok"

    monkeypatch.setattr("file_flow.llm_openai._post_chat_completion_once", fake_post)
    cfg = LlmEnvConfig(
        api_base="http://127.0.0.1/v1/chat/completions",
        api_keys=("k0", "k1"),
        model="m",
        timeout_sec=1.0,
        system_prompt="s",
    )
    assert call_openai_compatible_chat(cfg, "hi") == "ok"
    assert tokens_seen == ["k0", "k1"]


def test_call_rotates_through_three_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("file_flow.llm_openai.time.sleep", lambda _s: None)
    tokens_seen: list[str | None] = []

    def fake_post(
        url: str,
        model: str,
        user_message: str,
        api_key_token: str | None,
        timeout_sec: float,
        **kwargs: object,
    ) -> str:
        tokens_seen.append(api_key_token)
        if api_key_token in ("k0", "k1"):
            raise urllib.error.HTTPError(
                url,
                429,
                "Too Many",
                hdrs={},
                fp=io.BytesIO(b"{}"),
            )
        return "third"

    monkeypatch.setattr("file_flow.llm_openai._post_chat_completion_once", fake_post)
    cfg = LlmEnvConfig(
        api_base="http://127.0.0.1/v1/chat/completions",
        api_keys=("k0", "k1", "k2"),
        model="m",
        timeout_sec=1.0,
        system_prompt="s",
    )
    assert call_openai_compatible_chat(cfg, "x") == "third"
    assert tokens_seen == ["k0", "k1", "k2"]


def test_last_key_429_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("file_flow.llm_openai.time.sleep", lambda _s: None)

    def fake_post(
        url: str,
        model: str,
        user_message: str,
        api_key_token: str | None,
        timeout_sec: float,
        **kwargs: object,
    ) -> str:
        raise urllib.error.HTTPError(
            url,
            429,
            "Too Many",
            hdrs={},
            fp=io.BytesIO(b'{"error":"still rate limited"}'),
        )

    monkeypatch.setattr("file_flow.llm_openai._post_chat_completion_once", fake_post)
    cfg = LlmEnvConfig(
        api_base="http://127.0.0.1/v1/chat/completions",
        api_keys=("only",),
        model="m",
        timeout_sec=1.0,
        system_prompt="s",
    )
    with pytest.raises(RuntimeError, match="429"):
        call_openai_compatible_chat(cfg, "x")
