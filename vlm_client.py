#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI 兼容视觉 API：图像分类（签名/指印/印章）与整页转写。"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable


def _extract_message_content(resp: dict[str, Any]) -> str:
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


def parse_vlm_classification_text(text: str) -> tuple[str, float]:
    raw = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.I)
    if m:
        raw = m.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(raw[start : end + 1])
        else:
            return ("手写签名", 0.0)
    label = data.get("label") if isinstance(data, dict) else None
    conf_raw = data.get("confidence") if isinstance(data, dict) else None
    if not isinstance(label, str):
        return ("手写签名", 0.0)
    label_strip = label.strip()
    aliases = {
        "signature": "手写签名",
        "fingerprint": "指印",
        "stamp": "印章",
        "seal": "印章",
    }
    if label_strip.lower() in aliases:
        label_strip = aliases[label_strip.lower()]
    if label_strip not in ("手写签名", "指印", "印章"):
        lower = label_strip.lower()
        if "指纹" in label_strip or "fingerprint" in lower:
            label_strip = "指印"
        elif "印" in label_strip or "stamp" in lower or "seal" in lower:
            label_strip = "印章"
        else:
            label_strip = "手写签名"
    try:
        confidence = float(conf_raw) if conf_raw is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return (label_strip, confidence)


DEFAULT_CLASSIFICATION_SYSTEM = (
    '你是文档图像判别助手，只输出一个 JSON：{"label":"手写签名"|"指印"|"印章","confidence":0到1的数字} ，不要其它说明。'
)


def build_openai_compatible_vlm_detector(
    api_base: str,
    model: str,
    api_key: str | None = None,
    timeout_sec: float = 120.0,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    chat_completions_path: str = "/v1/chat/completions",
) -> Callable[[Path], tuple[str, float]]:
    base = api_base.rstrip("/")
    path = chat_completions_path if chat_completions_path.startswith("/") else f"/{chat_completions_path}"
    sys_msg = (
        system_prompt.strip()
        if isinstance(system_prompt, str) and system_prompt.strip()
        else DEFAULT_CLASSIFICATION_SYSTEM
    )
    usr_tmpl = (
        user_prompt.strip()
        if isinstance(user_prompt, str) and user_prompt.strip()
        else "请判断图像属于：手写签名、指印、印章中的哪一类。"
    )

    def _detect(image_path: Path) -> tuple[str, float]:
        mime, _ = mimetypes.guess_type(str(image_path))
        mime = mime or "image/png"
        b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": usr_tmpl},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        text = _extract_message_content(parsed)
        return parse_vlm_classification_text(text)

    return _detect


DEFAULT_PAGE_SYSTEM = (
    "你是 OCR 备选助手。给定一页 PDF 的渲染图，请逐行提取可读文字，保留段落结构。"
    "若完全没有文字，输出单行：(无可用文本)。只输出正文，不要 JSON。"
)


def build_openai_compatible_vlm_page_transcriber(
    api_base: str,
    model: str,
    api_key: str | None = None,
    timeout_sec: float = 180.0,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    chat_completions_path: str = "/v1/chat/completions",
) -> Callable[[Path, int], str]:
    base = api_base.rstrip("/")
    path = chat_completions_path if chat_completions_path.startswith("/") else f"/{chat_completions_path}"
    sys_msg = (
        system_prompt.strip()
        if isinstance(system_prompt, str) and system_prompt.strip()
        else DEFAULT_PAGE_SYSTEM
    )

    def _transcribe(image_path: Path, page_number: int = 1) -> str:
        usr = user_prompt_template or "这是第 {page_number} 页，请提取全部可见文字。"
        if isinstance(usr, str):
            usr = usr.format(page_number=page_number)
        mime, _ = mimetypes.guess_type(str(image_path))
        mime = mime or "image/png"
        b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": usr},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return _extract_message_content(parsed).strip()

    return _transcribe
