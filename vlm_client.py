#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 兼容的视觉对话 API 客户端：将本地图片编码为 data URL，请求 chat completions，
解析模型返回的 JSON，产出与 CLIP 管线一致的 (标签, 置信度)。

适用于多数提供 `/v1/chat/completions` 的网关（OpenAI、部分国产聚合接口等）。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from visual_tagging import VISUAL_LABELS

# 英文/简称别名 → 与 visual_tagging 一致的中文标签
_LABEL_ALIASES: dict[str, str] = {
    "手写签名": "手写签名",
    "签名": "手写签名",
    "signature": "手写签名",
    "handwritten": "手写签名",
    "指印": "指印",
    "指纹": "指印",
    "手印": "指印",
    "fingerprint": "指印",
    "印章": "印章",
    "公章": "印章",
    "红章": "印章",
    "seal": "印章",
    "stamp": "印章",
}

_DEFAULT_SYSTEM_PROMPT = (
    "你是文档图像分析助手，只根据图像内容从给定类别中选择一项并给出置信度。"
)

_DEFAULT_USER_PROMPT = (
    "判断图中哪一种视觉元素最符合（只能选一个）：手写签名、指印、印章。\n"
    "请严格只输出一个 JSON 对象，不要 Markdown 代码围栏，不要其它解释文字。"
    '格式示例：{"label":"手写签名","confidence":0.92}\n'
    "label 取值必须是：手写签名、指印、印章 三者之一；confidence 为 0 到 1 之间的小数。"
)


def parse_vlm_classification_text(content: str) -> tuple[str, float] | None:
    """
    从模型回复文本中解析 label 与 confidence。
    支持裸 JSON、```json 代码块、或正文中的首个 {...}。
    """
    text = (content or "").strip()
    if not text:
        return None

    parsed_obj: Any | None = None
    for candidate in _json_candidates(text):
        try:
            parsed_obj = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if parsed_obj is None:
        return None
    if not isinstance(parsed_obj, dict):
        return None

    raw_label = parsed_obj.get("label")
    if raw_label is None:
        raw_label = parsed_obj.get("类别")
    if not isinstance(raw_label, str):
        return None

    conf = parsed_obj.get("confidence")
    if conf is None:
        conf = parsed_obj.get("score")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return None
    score = float(conf)
    score = max(0.0, min(1.0, score))

    canonical = _normalize_label(raw_label)
    if canonical is None:
        return None
    return canonical, score


def _json_candidates(text: str) -> list[str]:
    """生成若干待尝试的 JSON 子串。"""
    out: list[str] = []
    if text.startswith("{") or text.startswith("["):
        out.append(text)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        out.append(fence.group(1).strip())
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        out.append(brace.group(0))
    if text not in out:
        out.insert(0, text)
    return out


def _normalize_label(raw: str) -> str | None:
    key = raw.strip()
    if key in VISUAL_LABELS:
        return key
    lower = key.lower()
    if lower in _LABEL_ALIASES:
        return _LABEL_ALIASES[lower]
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    return None


def _image_to_data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/png"
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_openai_compatible_vlm_detector(
    *,
    api_base: str,
    api_key: str | None,
    model: str,
    timeout_sec: float = 120.0,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    chat_completions_path: str = "/v1/chat/completions",
) -> Callable[[Path], tuple[str, float]]:
    """
    构造与 enrich_document_with_visual_tags 兼容的检测函数。
    api_base 示例：https://api.openai.com 或 https://your-gateway/v1 的根（会自动拼接路径）。
    """
    base = api_base.rstrip("/")
    path = chat_completions_path if chat_completions_path.startswith("/") else f"/{chat_completions_path}"
    url = f"{base}{path}"

    sys_p = system_prompt if isinstance(system_prompt, str) and system_prompt.strip() else _DEFAULT_SYSTEM_PROMPT
    usr_p = user_prompt if isinstance(user_prompt, str) and user_prompt.strip() else _DEFAULT_USER_PROMPT

    def _detect(image_path: Path) -> tuple[str, float]:
        data_url = _image_to_data_url(image_path)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_p},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": usr_p},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 256,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM HTTP {exc.code}: {detail[:500]}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"VLM 请求失败: {exc}") from exc

        try:
            resp_obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"VLM 响应非 JSON: {raw[:300]}") from exc

        content = _extract_message_content(resp_obj)
        parsed = parse_vlm_classification_text(content)
        if parsed is None:
            raise RuntimeError(f"VLM 返回无法解析为标签 JSON，原文: {content[:400]}")
        return parsed

    return _detect


def _extract_message_content(resp_obj: Any) -> str:
    """从 OpenAI 风格响应中取出 assistant 文本。"""
    choices = resp_obj.get("choices") if isinstance(resp_obj, dict) else None
    if not choices or not isinstance(choices, list):
        return ""
    first = choices[0] if choices else None
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""
