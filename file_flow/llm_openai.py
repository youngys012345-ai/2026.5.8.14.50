#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_flow 专用：OpenAI 兼容 Chat Completions 文本调用、日志、连接参数解析。

各环节 **system** 须分别指定：``build_llm_env_config(merged, system_prompt)`` 由调用方传入本环节的
``system_prompt``（如 ``schema_llm_extract.load_schema_extract_system_prompt``、
``standards_llm_review.load_standards_review_system_prompt``）。
勿将 ``FILE_FLOW_SYSTEM_PROMPT`` / ``file_flow_llm_system_prompt``（历史字段级填答用）误当作 schema 摘录的 system。

不依赖 ``review_standard_llm_fill`` / ``openai`` 第三方库：使用标准库 ``urllib`` 向 **Chat Completions**
端点 POST JSON。``file_flow`` 内各环节将**全部指令与结构化内容拼成一条 ``user`` 消息**，**不再**单独发送
``role=system``（兼容：若 ``LlmEnvConfig.system_prompt`` 非空，会合并进同一条 ``user``，仍仅一条 user）。

``LLM_API_BASE`` 可为：

- **完整 POST URL**（须含路径 ``.../chat/completions``），将原样使用；或
- **常见根地址** ``https://主机/.../v1``（可无尾斜杠），将**自动**补全为 ``.../v1/chat/completions``。

若配置了 ``LLM_API_KEY`` 与 ``LLM_API_KEY_BACKUP1`` / ``LLM_API_KEY_BACKUP2``，遇 **429 / 403**（部分厂商 QPM 限流）、**503** 等可重试类错误时，
会在**同一请求内**依次换密钥重试；部分厂商对 QPM 返回 **400** 且正文含限流关键词时也会轮换。
可选 ``LLM_KEY_ROTATION_SLEEP_SEC``：换 key 前休眠秒数（默认 ``0``；QPM 场景可设为 ``0.5``～``2``）。
可选 ``LLM_HTTP_WAIT_LOG_INTERVAL_SEC``：单次请求在等待网关响应期间，每隔多少秒打一条 **WARNING** 心跳日志（默认 ``25``，设为 ``0`` 关闭）；便于排查「长时间无输出直至超时」。

URL/模型/超时等从环境变量或 ``pipeline.json``（由 ``load_merged_pipeline_config`` 仅从磁盘加载；不与环境默认字典合并）读取。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

_LOG = logging.getLogger(__name__)

# 遇限流或临时故障时，在同一请求内依次尝试 LLM_API_KEY → LLM_API_KEY_BACKUP1 → BACKUP2。
# 含 403：部分网关/模型在 QPM 超限时返回 403（正文可能为空或不含关键词），与 429 同样轮换备用密钥。
_LLM_KEY_ROTATION_HTTP_CODES = frozenset({403, 408, 429, 502, 503, 504, 529})


def _response_body_suggests_rate_limit(body: str) -> bool:
    """部分网关对 QPM/并发用 400/422 返回，正文中含限流提示时也轮换密钥。"""
    b = (body or "").lower()
    if not b.strip():
        return False
    hints = (
        "rate limit",
        "too many requests",
        "quota",
        "qpm",
        "rpm",
        "tpm",
        "throttl",
        "throttle",
        "exceed",
        "capacity",
        "限流",
        "请求过快",
        "请求过于频繁",
        "并发",
        "频率",
        "配额",
        "调用次数",
        "每分钟",
        "tokens per",
        "请稍后",
        "retry",
        "busy",
    )
    return any(h in b for h in hints)


def _should_rotate_llm_api_key(http_code: int, response_body: str) -> bool:
    if http_code in _LLM_KEY_ROTATION_HTTP_CODES:
        return True
    if http_code in (400, 422) and _response_body_suggests_rate_limit(response_body):
        return True
    return False


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


def _llm_key_rotation_sleep_sec() -> float:
    """换用下一密钥前的休眠秒数；默认 0，避免拖慢单测与低延迟场景。"""
    raw = _env_first("LLM_KEY_ROTATION_SLEEP_SEC", "LLM_KEY_ROTATION_BACKOFF_SEC")
    if raw is None:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _llm_http_wait_log_interval_sec() -> float:
    """
    单次 POST 阻塞等待期间，每隔多少秒打一条 WARNING 心跳（便于发现网关卡住）。
    环境变量 ``LLM_HTTP_WAIT_LOG_INTERVAL_SEC``：默认 25；0 或负数表示关闭。
    """
    raw = _env_first("LLM_HTTP_WAIT_LOG_INTERVAL_SEC")
    if raw is None:
        return 25.0
    try:
        v = float(raw)
        return 0.0 if v <= 0 else v
    except ValueError:
        return 25.0


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
    """大模型连接参数。``system_prompt`` 在 ``file_flow`` 内应留空；环节指令已并入 ``call_openai_compatible_chat`` 的 ``user_text``。"""

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
    解析连接参数（URL / 模型 / 超时 / 密钥链）。

    ``system_prompt`` 仅写入 ``LlmEnvConfig`` 供**兼容**：``call_openai_compatible_chat`` 会将其与 ``user_text``
    合并为**一条** ``user`` 消息（不再单独发 API ``role=system``）。``file_flow`` 内各环节应传 ``\"\"``，
    并在业务模块内自行把环节指令拼进 ``user_text``。
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
    if not (system_prompt or "").strip():
        sp = "(空，API 仅发单条 user；file_flow 内指令已并入 user 正文)"
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


def iter_top_level_sections(data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for key, val in data.items():
        if isinstance(val, dict):
            yield key, val


def _compose_single_user_message(cfg: LlmEnvConfig, user_text: str) -> str:
    """
    合并为一条 user 正文。``file_flow`` 默认 ``cfg.system_prompt`` 为空；若非空（兼容旧代码）则插在首部。
    """
    sp = (cfg.system_prompt or "").strip()
    ut = user_text or ""
    if not sp:
        return ut
    return f"{sp}\n\n---\n\n{ut}"


def _post_chat_completion_once(
    url: str,
    model: str,
    user_message: str,
    api_key_token: str | None,
    timeout_sec: float,
    *,
    key_attempt: int = 1,
    key_total: int = 1,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
    }
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key_token:
        headers["Authorization"] = f"Bearer {api_key_token}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    interval = _llm_http_wait_log_interval_sec()
    stop_evt = threading.Event()
    t0 = time.monotonic()

    def _heartbeat() -> None:
        while not stop_evt.wait(timeout=interval):
            elapsed = time.monotonic() - t0
            _LOG.warning(
                "[环节:HTTP等待] 仍未收到完整响应（已阻塞约 %.0fs，单次读超时上限 %.0fs）。"
                "可能原因：网关排队、模型推理慢、TLS/代理挂起、或上游未及时返回。",
                elapsed,
                timeout_sec,
            )

    hb_thread: threading.Thread | None = None
    if interval > 0:
        hb_thread = threading.Thread(target=_heartbeat, name="llm-http-wait-log", daemon=True)
        hb_thread.start()

    _LOG.info(
        "[环节:HTTP] 即将进入阻塞等待：POST 密钥槽 %s/%s，JSON payload=%s 字节，读超时=%.1fs，URL=%s",
        key_attempt,
        key_total,
        len(data),
        timeout_sec,
        url,
    )

    body = ""
    http_status: int | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            http_status = resp.getcode()
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, OSError) as e:
        elapsed = time.monotonic() - t0
        err_s = str(e).lower()
        reason = getattr(e, "reason", None)
        timed = (
            "timed out" in err_s
            or "timeout" in err_s
            or isinstance(reason, TimeoutError)
            or (isinstance(reason, BaseException) and "timeout" in str(reason).lower())
        )
        if timed:
            _LOG.error(
                "[环节:API超时或挂起] 约 %.2fs 后仍未完成（读超时设定 %.1fs）URL=%s err=%s",
                elapsed,
                timeout_sec,
                url,
                e,
            )
        else:
            _LOG.error(
                "[环节:API错误] 连接/传输失败 耗时=%.2fs URL=%s err=%s",
                elapsed,
                url,
                e,
            )
        raise
    finally:
        stop_evt.set()
        if hb_thread is not None:
            hb_thread.join(timeout=2.0)

    elapsed = time.monotonic() - t0
    _LOG.info(
        "[环节:HTTP] HTTP 状态=%s，已读完全部响应体，阻塞耗时 %.2fs，原始 JSON 文本约 %s 字符",
        http_status if http_status is not None else "(未知)",
        elapsed,
        len(body),
    )

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as je:
        _LOG.error(
            "[环节:API错误] 响应非合法 JSON（HTTP 已返回，耗时 %.2fs）正文前 500 字=%s",
            elapsed,
            (body[:500] + ("…" if len(body) > 500 else "")),
        )
        raise RuntimeError(f"大模型返回非 JSON: {je}") from je

    text = _extract_message_content(parsed).strip()
    _LOG.info("[环节:HTTP] 解析完成，助手文本长度=%s", len(text))
    return text


def call_openai_compatible_chat(cfg: LlmEnvConfig, user_text: str) -> str:
    """
    发起 Chat Completions 请求：``messages`` 仅含一条 ``role=user``（``file_flow`` 将环节指令已拼入 ``user_text``）。

    若 ``cfg.system_prompt`` 非空（兼容旧代码），会经 ``_compose_single_user_message`` 合并进同一条 ``user``，仍不单独发 ``system``。
    遇限流等错误时在同一调用内依次轮换 ``api_keys`` 重试。
    """
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
    if not is_http_endpoint_url(url):
        preview = url if len(url) <= 120 else url[:117] + "..."
        raise RuntimeError(
            "LLM_API_BASE 须为以 http:// 或 https:// 开头的 URL；若为 OpenAI 兼容网关，"
            "可使用 ``https://主机/.../v1`` 或完整 ``.../v1/chat/completions``。"
            f"当前规范化后为: {preview!r}"
        )
    key_slots: list[str | None] = list(cfg.api_keys) if cfg.api_keys else [None]
    combined = _compose_single_user_message(cfg, user_text)
    if len(key_slots) > 1:
        _LOG.info(
            "[环节:API请求] POST %s model=%s 超时=%ss 单条user字符数=%s 密钥轮询槽位=%s",
            url,
            cfg.model,
            cfg.timeout_sec,
            len(combined),
            len(key_slots),
        )
    else:
        _LOG.info(
            "[环节:API请求] POST %s model=%s 超时=%ss 单条user字符数=%s",
            url,
            cfg.model,
            cfg.timeout_sec,
            len(combined),
        )
    _LOG.debug(
        "[环节:prompt预览] user 前120字=%s",
        (combined[:120] + "…") if len(combined) > 120 else combined,
    )

    for idx, token in enumerate(key_slots):
        try:
            text = _post_chat_completion_once(
                url,
                cfg.model,
                combined,
                token,
                cfg.timeout_sec,
                key_attempt=idx + 1,
                key_total=len(key_slots),
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
            backoff = _llm_key_rotation_sleep_sec()
            if _should_rotate_llm_api_key(e.code, detail) and idx < len(key_slots) - 1:
                _LOG.warning(
                    "[环节:密钥轮换] HTTP %s，%s 后换用第 %s/%s 个密钥 URL=%s 响应片段=%s",
                    e.code,
                    f"休眠 {backoff}s" if backoff > 0 else "无休眠",
                    idx + 2,
                    len(key_slots),
                    url,
                    detail[:300] + ("…" if len(detail) > 300 else ""),
                )
                if backoff > 0:
                    time.sleep(backoff)
                continue
            _LOG.error(
                "[环节:API错误] HTTP %s URL=%s 响应正文片段=%s",
                e.code,
                url,
                detail[:500] + ("…" if len(detail) > 500 else ""),
            )
            raise RuntimeError(f"大模型 HTTP 错误 {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"大模型连接失败: {e}") from e
