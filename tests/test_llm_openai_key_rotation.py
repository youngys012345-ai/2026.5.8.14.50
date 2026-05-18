# -*- coding: utf-8 -*-
"""file_flow.llm_openai：限流时多密钥轮换。"""

from __future__ import annotations

import io
import urllib.error

import pytest

from file_flow.llm_openai import (
    LlmEnvConfig,
    _collect_llm_api_key_chain,
    _should_rotate_llm_api_key,
    call_openai_compatible_chat,
)


def test_collect_llm_api_key_chain_reads_primary_and_four_backups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "p")
    for i in range(1, 5):
        monkeypatch.setenv(f"LLM_API_KEY_BACKUP{i}", f"b{i}")
    assert _collect_llm_api_key_chain() == ("p", "b1", "b2", "b3", "b4")


def test_collect_llm_api_key_chain_keeps_duplicate_backup_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "a")
    monkeypatch.setenv("LLM_API_KEY_BACKUP1", "a")
    monkeypatch.setenv("LLM_API_KEY_BACKUP2", "b")
    assert _collect_llm_api_key_chain() == ("a", "a", "b")


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


def test_call_rotates_through_five_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """主密钥 + BACKUP1～4 共 5 槽：前四次 429，第五次成功。"""
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
        if api_key_token in ("k0", "k1", "k2", "k3"):
            raise urllib.error.HTTPError(
                url,
                429,
                "Too Many",
                hdrs={},
                fp=io.BytesIO(b"{}"),
            )
        return "ok5"

    monkeypatch.setattr("file_flow.llm_openai._post_chat_completion_once", fake_post)
    cfg = LlmEnvConfig(
        api_base="http://127.0.0.1/v1/chat/completions",
        api_keys=("k0", "k1", "k2", "k3", "k4"),
        model="m",
        timeout_sec=1.0,
        system_prompt="",
    )
    assert call_openai_compatible_chat(cfg, "x") == "ok5"
    assert tokens_seen == ["k0", "k1", "k2", "k3", "k4"]


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


def test_transport_timeout_rotates_linear_through_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """连接超时：线性逐槽切换，全部超时则报错。"""
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
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr("file_flow.llm_openai._post_chat_completion_once", fake_post)
    cfg = LlmEnvConfig(
        api_base="http://127.0.0.1/v1/chat/completions",
        api_keys=("k0", "k1", "k2"),
        model="m",
        timeout_sec=1.0,
        system_prompt="s",
    )
    with pytest.raises(RuntimeError, match="最大尝试次数"):
        call_openai_compatible_chat(cfg, "x")
    # 每个密钥槽恰好尝试一次，不回头
    assert tokens_seen == ["k0", "k1", "k2"]


def test_http_429_rotates_linear_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """多槽 429：线性逐槽换，最后一个槽成功即返回，不回头。"""
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
        if len(tokens_seen) < 3:
            raise urllib.error.HTTPError(url, 429, "Too Many", hdrs={}, fp=io.BytesIO(b"{}"))
        return "ok"

    monkeypatch.setattr("file_flow.llm_openai._post_chat_completion_once", fake_post)
    cfg = LlmEnvConfig(
        api_base="http://127.0.0.1/v1/chat/completions",
        api_keys=("k0", "k1", "k2"),
        model="m",
        timeout_sec=1.0,
        system_prompt="s",
    )
    assert call_openai_compatible_chat(cfg, "x") == "ok"
    # 3 个槽线性遍历，第 3 个成功，不回头
    assert tokens_seen == ["k0", "k1", "k2"]


def test_stops_after_max_attempts_when_all_keys_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("file_flow.llm_openai.time.sleep", lambda _s: None)
    monkeypatch.setattr("file_flow.llm_openai._llm_key_rotation_max_attempts", lambda _n: 3)

    def fake_post(
        url: str,
        model: str,
        user_message: str,
        api_key_token: str | None,
        timeout_sec: float,
        **kwargs: object,
    ) -> str:
        raise urllib.error.HTTPError(url, 429, "Too Many", hdrs={}, fp=io.BytesIO(b"{}"))

    monkeypatch.setattr("file_flow.llm_openai._post_chat_completion_once", fake_post)
    cfg = LlmEnvConfig(
        api_base="http://127.0.0.1/v1/chat/completions",
        api_keys=("k0", "k1"),
        model="m",
        timeout_sec=1.0,
        system_prompt="s",
    )
    with pytest.raises(RuntimeError, match="429"):
        call_openai_compatible_chat(cfg, "x")
