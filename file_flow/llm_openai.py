#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_flow 专用：OpenAI 兼容 Chat Completions 文本调用、日志、连接参数解析。

各环节 **system** 须分别指定：``build_llm_env_config(merged, system_prompt)`` 由调用方传入本环节的
``system_prompt``（如 ``schema_llm_extract.load_schema_extract_system_prompt``、
``standards_llm_review.load_standards_review_system_prompt``、或 ``llm_fill`` 使用的字段填答默认）。
勿将 ``FILE_FLOW_SYSTEM_PROMPT``（面向 ``llm_fill``）误当作 schema 摘录的 system。

不依赖 ``review_standard_llm_fill`` / ``openai`` 第三方库：使用标准库 ``urllib`` 向 **Chat Completions**
端点 POST JSON（``messages`` / ``model``），与 OpenAI 及多数兼容网关一致。

``LLM_API_BASE`` 可为：

- **完整 POST URL**（须含路径 ``.../chat/completions``），将原样使用；或
- **常见根地址** ``https://主机/.../v1``（可无尾斜杠），将**自动**补全为 ``.../v1/chat/completions``。

URL/模型/超时等从环境变量或 ``pipeline.json``（由 ``load_merged_pipeline_config`` 仅从磁盘加载；不与环境默认字典合并）读取。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

_LOG = logging.getLogger(__name__)

_LLM_KEY_ROTATION_HTTP_CODES = frozenset({429, 503})


def _extract_message_content(resp: dict[str, Any]) -> str:
    """从 Chat Completions JSON 取出助手文本（与 vlm_client 行为一致）。"""
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        return "".join(chunks)
    return ""


def normalize_llm_chat_completions_url(api_base: str) -> str:
    """
    将 ``LLM_API_BASE`` 规范为可直接 POST 的 Chat Completions URL。

    - 若路径中已含 ``chat/completions``（忽略大小写），返回去除末尾 ``/`` 后的 URL。
    - 若以 ``/v1`` 结尾（OpenAI、DeepSeek、Groq 等常见兼容形态），追加 ``/chat/completions``。
    - 若仅有协议与主机（路径为空或 ``/``），追加 ``/v1/chat/completions``。
    - 其余情况原样返回（便于 Azure 部署 URL、自建网关等显式完整路径）。
    """
    u = (api_base or "").strip()
    if not u:
        return u
    u2 = u.rstrip("/")
    low = u2.lower()
    if "chat/completions" in low:
        return u2
    if low.endswith("/v1"):
        return u2 + "/chat/completions"
    parsed = urlparse(u2)
    path = (parsed.path or "").strip("/")
    if path == "":
        return u2 + "/v1/chat/completions"
    return u2


def is_http_endpoint_url(value: str | None) -> bool:
    if value is None:
        return False
    t = str(value).strip().lower()
    return t.startswith("http://") or t.startswith("https://")


def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def _merged_str(merged: dict[str, Any] | None, key: str) -> str | None:
    if not merged:
        return None
    v = merged.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def configure_logging(level: int | str | None = None, log_file: Path | None = None) -> None:
    """初始化日志；级别可由 ``REVIEW_STANDARD_LLM_LOG_LEVEL`` / ``LLM_LOG_LEVEL`` 指定。"""
    if level is None:
        raw = _env_first("REVIEW_STANDARD_LLM_LOG_LEVEL", "LLM_LOG_LEVEL")
        if raw:
            level = raw.upper()
        else:
            level = logging.INFO
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        lf = Path(log_file)
        lf.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(lf, encoding="utf-8"))

    try:
        logging.basicConfig(
            level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True
        )
    except TypeError:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    _LOG.setLevel(level)


@dataclass(frozen=True)
class LlmEnvConfig:
    """大模型连接参数。"""

    api_base: str | None
    api_keys: tuple[str, ...]
    model: str | None
    timeout_sec: float
    system_prompt: str

    @property
    def api_key(self) -> str | None:
        return self.api_keys[0] if self.api_keys else None


def _mask_secret(s: str | None) -> str:
    if not s:
        return "(未设置)"
    if len(s) <= 8:
        return "已设置(已隐藏)"
    return f"已设置(尾四位 …{s[-4:]})"


def _collect_llm_api_key_chain() -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    primary = _env_first("LLM_API_KEY", "OPENAI_API_KEY")
    if primary:
        out.append(primary)
        seen.add(primary)
    for name in ("LLM_API_KEY_BACKUP1", "LLM_API_KEY_BACKUP2"):
        v = _env_first(name)
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return tuple(out)


def _default_field_fill_system_prompt(merged: dict[str, Any] | None) -> str:
    """
    第二步 ``llm_fill`` 专用：对照各字段 content 与 standards 同下标的评审要点作答。
    与第一步 schema 全文摘录、第三步 standards 清单评审的 system 相互独立，勿混用。
    """
    m = merged or {}
    return (
        _env_first(
            "FILE_FLOW_SYSTEM_PROMPT",
            "REVIEW_FIELD_QA_SYSTEM_PROMPT",
            "LLM_SYSTEM_PROMPT",
        )
        or _merged_str(m, "file_flow_llm_system_prompt")
        or _merged_str(m, "vlm_system_prompt")
        or (
            "你是行政执法案卷评审助手。用户会提供某一字段从案卷中抽取的相关文字（content）、以及需要对照的评审要点。"
            "请严格依据「抽取内容」作答：先给出简要结论，再说明依据；不得臆测材料中不存在的内容。"
            "若材料不足以判断，须明确说明「无法判断」并简述原因。使用简体中文。"
        )
    )


def _timeout_from_env_and_merged(merged: dict[str, Any] | None) -> float:
    raw = _env_first("LLM_TIMEOUT_SEC")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    if merged:
        for k in ("file_flow_llm_timeout_sec", "vlm_timeout_sec"):
            v = merged.get(k)
            if v is not None and str(v).strip() != "":
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
    return 120.0


def build_llm_env_config(merged: dict[str, Any] | None, system_prompt: str) -> LlmEnvConfig:
    """
    仅解析连接参数（URL / 模型 / 超时 / 密钥链），``system_prompt`` 由调用方按环节传入，
    避免把「字段填答/评审」的默认 system 误用到 schema 全文摘录等环节。
    """
    m = merged or {}
    api_base = (
        _env_first("LLM_API_BASE", "OPENAI_API_BASE")
        or _merged_str(m, "file_flow_llm_api_base")
        or _merged_str(m, "vlm_api_base")
    )
    model = (
        _env_first("LLM_MODEL", "OPENAI_MODEL")
        or _merged_str(m, "file_flow_llm_model")
        or _merged_str(m, "vlm_model")
    )
    timeout_sec = _timeout_from_env_and_merged(m)
    key_chain = _collect_llm_api_key_chain()
    cfg = LlmEnvConfig(
        api_base=api_base,
        api_keys=key_chain,
        model=model,
        timeout_sec=timeout_sec,
        system_prompt=system_prompt,
    )
    role = "（本环节 system）"
    sp = (system_prompt or "").replace("\n", " ").strip()
    if len(sp) > 80:
        sp = sp[:79] + "…"
    _LOG.info(
        "[环节:配置] file_flow LLM 连接：api_base=%s model=%s timeout=%s api_key槽位=%s（首个=%s） system 前80字=%s %s",
        cfg.api_base or "(未设置)",
        cfg.model or "(未设置)",
        cfg.timeout_sec,
        len(cfg.api_keys),
        _mask_secret(cfg.api_key),
        sp or "(空)",
        role,
    )
    return cfg


def load_llm_config_for_file_flow(merged: dict[str, Any] | None = None) -> LlmEnvConfig:
    """
    解析 LLM 配置，供 **第二步** ``llm_fill``（字段 content + standards 按下标对齐的评审问题 → answer）使用。

    连接项优先级：**环境变量** > ``pipeline.json`` 的 ``file_flow_llm_*`` > ``vlm_*`` 兜底。
    system 优先级：``FILE_FLOW_SYSTEM_PROMPT`` / ``REVIEW_FIELD_QA_SYSTEM_PROMPT`` / ``LLM_SYSTEM_PROMPT`` /
    ``file_flow_llm_system_prompt`` / ``vlm_system_prompt`` / 内置「对照抽取内容作答」默认。

    **注意**：schema 全文摘录请用 ``build_llm_env_config(merged, load_schema_extract_system_prompt(...))``（见 ``schema_llm_extract``）；
    standards 清单评审请用 ``build_llm_env_config(merged, load_standards_review_system_prompt(...))``（见 ``standards_llm_review``）。
    """
    return build_llm_env_config(merged, _default_field_fill_system_prompt(merged))


def iter_top_level_sections(data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for key, val in data.items():
        if isinstance(val, dict):
            yield key, val


def _post_chat_completion_once(
    url: str,
    model: str,
    system_prompt: str,
    user_text: str,
    api_key_token: str | None,
    timeout_sec: float,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key_token:
        headers["Authorization"] = f"Bearer {api_key_token}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return _extract_message_content(parsed).strip()


def call_openai_compatible_chat(cfg: LlmEnvConfig, user_text: str) -> str:
    """遇 429/503 时在同一调用内轮换 ``api_keys`` 重试。"""
    if not cfg.api_base or not str(cfg.api_base).strip():
        raise RuntimeError(
            "未读取到大模型 API 地址：请设置 LLM_API_BASE 或 OPENAI_API_BASE（OpenAI 兼容网关可用 ``https://主机/.../v1``，"
            "程序将自动补全为 ``.../v1/chat/completions``；或直接写完整 POST URL），"
            "或在 file_flow/pipeline.json 中配置 file_flow_llm_api_base / vlm_api_base。"
            "环境变量从操作系统已注入的键读取；若写在 .env 中：运行前须已执行 ``ensure_step_dotenv_loaded``（"
            "``pipeline_merge`` / ``pdf_prepare`` 等入口会在导入时加载 ``file_flow/.env`` 与上一级目录 ``.env``）。"
        )
    if not cfg.model or not str(cfg.model).strip():
        raise RuntimeError(
            "未读取到大模型名称：请设置 LLM_MODEL（或 OPENAI_MODEL、file_flow_llm_model、vlm_model）。"
            "仅配置 LLM_API_BASE 不足以发起调用。"
        )
    raw = cfg.api_base.strip()
    url = normalize_llm_chat_completions_url(raw)
    if url != raw.rstrip("/"):
        _LOG.info(
            "[环节:API] LLM_API_BASE 已规范为 Chat Completions 地址（原为根路径时可自动补全）: %s -> %s",
            raw,
            url,
        )
    if not is_http_endpoint_url(url):
        preview = url if len(url) <= 120 else url[:117] + "..."
        raise RuntimeError(
            "LLM_API_BASE 须为以 http:// 或 https:// 开头的 URL；若为 OpenAI 兼容网关，"
            "可使用 ``https://主机/.../v1`` 或完整 ``.../v1/chat/completions``。"
            f"当前规范化后为: {preview!r}"
        )
    key_slots: list[str | None] = list(cfg.api_keys) if cfg.api_keys else [None]
    if len(key_slots) > 1:
        _LOG.info(
            "[环节:API请求] POST %s model=%s 超时=%ss 用户消息字符数=%s 密钥轮询槽位=%s",
            url,
            cfg.model,
            cfg.timeout_sec,
            len(user_text),
            len(key_slots),
        )
    else:
        _LOG.info(
            "[环节:API请求] POST %s model=%s 超时=%ss 用户消息字符数=%s",
            url,
            cfg.model,
            cfg.timeout_sec,
            len(user_text),
        )
    _LOG.debug(
        "[环节:prompt预览] system 前80字=%s",
        (cfg.system_prompt[:80] + "…") if len(cfg.system_prompt) > 80 else cfg.system_prompt,
    )

    for idx, token in enumerate(key_slots):
        try:
            text = _post_chat_completion_once(
                url,
                cfg.model,
                cfg.system_prompt,
                user_text,
                token,
                cfg.timeout_sec,
            )
            if idx > 0:
                _LOG.info(
                    "[环节:API响应] 使用第 %s/%s 个密钥成功，助手回复字符数=%s",
                    idx + 1,
                    len(key_slots),
                    len(text),
                )
            else:
                _LOG.info("[环节:API响应] 助手回复字符数=%s", len(text))
            return text
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code in _LLM_KEY_ROTATION_HTTP_CODES and idx < len(key_slots) - 1:
                _LOG.warning(
                    "[环节:密钥轮换] HTTP %s，换用下一密钥 %s/%s URL=%s 响应片段=%s",
                    e.code,
                    idx + 2,
                    len(key_slots),
                    url,
                    detail[:300] + ("…" if len(detail) > 300 else ""),
                )
                continue
            _LOG.error(
                "[环节:API错误] HTTP %s URL=%s 响应正文片段=%s",
                e.code,
                url,
                detail[:500] + ("…" if len(detail) > 500 else ""),
            )
            raise RuntimeError(f"大模型 HTTP 错误 {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            _LOG.error("[环节:API错误] 网络/URL 异常 URL=%s err=%s", url, e)
            raise RuntimeError(f"大模型连接失败: {e}") from e
