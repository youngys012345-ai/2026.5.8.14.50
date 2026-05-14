#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""file_flow 产出 JSON 的文件名后缀：由 ``pipeline.json`` 中 ``file_flow_suffix_*`` 控制。"""

from __future__ import annotations

import re
from typing import Any

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _one_suffix(merged: dict[str, Any], key: str, default: str) -> str:
    v = merged.get(key)
    if not isinstance(v, str):
        return default
    s = v.strip()
    if not s or "/" in s or "\\" in s or "." in s:
        return default
    if not s.startswith("_"):
        s = "_" + s
    if not _SAFE_TOKEN.match(s[1:]):
        return default
    return s


def file_flow_stem_suffixes(merged: dict[str, Any]) -> tuple[str, str, str]:
    """
    返回 (work, answered, review) 三段后缀，形如 ``_work``、``_answered``、``_review``，
    用于 ``{pdf_stem}{suffix}.json``。
    """
    return (
        _one_suffix(merged, "file_flow_suffix_work", "_work"),
        _one_suffix(merged, "file_flow_suffix_answered", "_answered"),
        _one_suffix(merged, "file_flow_suffix_review", "_review"),
    )


def work_json_filename_for_stem(pdf_stem: str, merged: dict[str, Any]) -> str:
    sw, _, _ = file_flow_stem_suffixes(merged)
    return f"{pdf_stem}{sw}.json"


def work_glob_pattern(merged: dict[str, Any]) -> str:
    sw, _, _ = file_flow_stem_suffixes(merged)
    return f"*{sw}.json"


def answered_glob_pattern(merged: dict[str, Any]) -> str:
    _, sa, _ = file_flow_stem_suffixes(merged)
    return f"*{sa}.json"


def review_glob_pattern(merged: dict[str, Any]) -> str:
    _, _, sr = file_flow_stem_suffixes(merged)
    return f"*{sr}.json"


def replace_work_json_name_with_answered(name: str, merged: dict[str, Any]) -> str:
    sw, sa, _ = file_flow_stem_suffixes(merged)
    return name.replace(f"{sw}.json", f"{sa}.json", 1)


def stem_base_from_stage_stem(stem: str, merged: dict[str, Any]) -> str:
    """从 ``某案_work`` / ``某案_answered`` 形式的 stem 得到基础名 ``某案``。"""
    sw, sa, _ = file_flow_stem_suffixes(merged)
    if stem.endswith(sw):
        return stem[: -len(sw)]
    if stem.endswith(sa):
        return stem[: -len(sa)]
    return stem


def review_json_filename_for_base(base: str, merged: dict[str, Any]) -> str:
    _, _, sr = file_flow_stem_suffixes(merged)
    return f"{base}{sr}.json"
