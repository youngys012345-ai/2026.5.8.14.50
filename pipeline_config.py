#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽取管线配置：环境变量 + JSON 文件合并，供 extract_pdf 等脚本复用。

合并优先级（后者覆盖前者）：内置默认值 → 环境变量 → JSON 配置 → 命令行参数。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# 与 extract_pdf 中参数对应的可配置键（便于日后扩展，勿随意删改键名）
CONFIG_KEYS = frozenset(
    {
        "hybrid",
        "hybrid_url",
        "hybrid_mode",
        "hybrid_fallback",
        "hybrid_timeout",
        "skip_health_check",
        "table_method",
        "reading_order",
        "quiet",
        "visual_tagging",
        "visual_min_score",
        "vlm_api_base",
        "vlm_api_key",
        "vlm_model",
        "vlm_timeout_sec",
        "vlm_chat_path",
        "vlm_system_prompt",
        "vlm_user_prompt",
        "recursive",
        "output_dir",
        "json_output_dir",
    }
)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str | None) -> str | None:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def defaults_from_environment() -> dict[str, Any]:
    """从环境变量读取管线默认值（预留与脚本、容器编排对接）。"""
    d: dict[str, Any] = {}
    # Hybrid 客户端（Java 调远端 Docling 服务）
    hu = _env_str("OPENDATALOADER_HYBRID_URL", None)
    if hu is not None:
        d["hybrid_url"] = hu
    hm = _env_str("OPENDATALOADER_HYBRID_MODE", None)
    if hm is not None:
        d["hybrid_mode"] = hm
    hb = _env_str("OPENDATALOADER_HYBRID_BACKEND", None)
    if hb is not None:
        d["hybrid"] = hb
    if os.environ.get("OPENDATALOADER_HYBRID_FALLBACK"):
        d["hybrid_fallback"] = _env_bool("OPENDATALOADER_HYBRID_FALLBACK", False)
    ht = _env_str("OPENDATALOADER_HYBRID_TIMEOUT", None)
    if ht is not None:
        d["hybrid_timeout"] = ht
    if os.environ.get("OPENDATALOADER_SKIP_HEALTH_CHECK"):
        d["skip_health_check"] = _env_bool("OPENDATALOADER_SKIP_HEALTH_CHECK", False)
    # 视觉增强（CLIP / VLM）
    vt = _env_str("OPENDATALOADER_VISUAL_TAGGING", None)
    if vt is not None:
        d["visual_tagging"] = vt
    if os.environ.get("OPENDATALOADER_VISUAL_MIN_SCORE"):
        d["visual_min_score"] = _env_float("OPENDATALOADER_VISUAL_MIN_SCORE", 0.5)
    # VLM（OpenAI 兼容视觉接口）
    vb = _env_str("OPENDATALOADER_VLM_API_BASE", None)
    if vb is not None:
        d["vlm_api_base"] = vb
    vk = _env_str("OPENDATALOADER_VLM_API_KEY", None)
    if vk is not None:
        d["vlm_api_key"] = vk
    vm = _env_str("OPENDATALOADER_VLM_MODEL", None)
    if vm is not None:
        d["vlm_model"] = vm
    if os.environ.get("OPENDATALOADER_VLM_TIMEOUT_SEC"):
        d["vlm_timeout_sec"] = _env_float("OPENDATALOADER_VLM_TIMEOUT_SEC", 120.0)
    vcp = _env_str("OPENDATALOADER_VLM_CHAT_PATH", None)
    if vcp is not None:
        d["vlm_chat_path"] = vcp
    vsp = _env_str("OPENDATALOADER_VLM_SYSTEM_PROMPT", None)
    if vsp is not None:
        d["vlm_system_prompt"] = vsp
    vup = _env_str("OPENDATALOADER_VLM_USER_PROMPT", None)
    if vup is not None:
        d["vlm_user_prompt"] = vup
    # 输出与遍历
    od = _env_str("OPENDATALOADER_OUTPUT_DIR", None)
    if od is not None:
        d["output_dir"] = od
    jd = _env_str("OPENDATALOADER_JSON_OUTPUT_DIR", None)
    if jd is not None:
        d["json_output_dir"] = jd
    if os.environ.get("OPENDATALOADER_RECURSIVE"):
        d["recursive"] = _env_bool("OPENDATALOADER_RECURSIVE", False)
    if os.environ.get("OPENDATALOADER_QUIET"):
        d["quiet"] = _env_bool("OPENDATALOADER_QUIET", False)
    return d


def load_config_file(path: Path) -> dict[str, Any]:
    """加载 JSON 配置文件，忽略未知键并做类型约束。"""
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件必须是 JSON 对象: {path}")
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in CONFIG_KEYS:
            continue
        if v is None:
            continue
        out[k] = v
    return out


def merge_defaults(
    base: dict[str, Any],
    *overlays: dict[str, Any],
) -> dict[str, Any]:
    """合并多组默认值，后面的覆盖前面的。"""
    merged = dict(base)
    for layer in overlays:
        for k, v in layer.items():
            merged[k] = v
    return merged


def pop_config_path_from_argv(argv: list[str]) -> tuple[Path | None, list[str]]:
    """
    从命令行参数中取出 --config，返回 (配置文件路径, 其余参数)。
    支持 --config path 与 --config=path，可出现在任意位置。
    """
    out: list[str] = []
    config_path: Path | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = Path(argv[i + 1])
            i += 2
            continue
        if argv[i].startswith("--config="):
            config_path = Path(argv[i].split("=", 1)[1])
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return config_path, out
