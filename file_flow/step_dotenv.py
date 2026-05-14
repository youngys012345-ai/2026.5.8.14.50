#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅供 ``file_flow`` 使用：加载 **工作区目录**及其**上一级目录**下的 ``.env`` / ``环节变量.env``。

加载顺序（``python-dotenv`` 的 ``override`` 含义见各步注释）：

1. ``{workspace}/.env`` — ``override=False``，不覆盖操作系统已有键；
2. ``{workspace 的父目录}/.env`` — ``override=False``，只填补上一步仍未出现的键（便于与仓库根 ``.env`` 共用 ``LLM_*``）；
3. ``{workspace}/环节变量.env`` — ``override=True``，覆盖前两步写入的同名键。

不读取 ``cwd`` 下的 ``.env``，避免与无关工作目录混淆；请将本流程专用覆盖放在 ``file_flow/环节变量.env``。
"""

from __future__ import annotations

from pathlib import Path

_done = False
_loaded_paths: list[Path] = []
_dotenv_unavailable = False


def ensure_step_dotenv_loaded(workspace: Path | None = None) -> tuple[list[Path], bool]:
    """
    同一进程内仅实际读盘一次。

    1. ``{workspace}/.env`` — 不覆盖操作系统已有键；
    2. ``{workspace 父目录}/.env`` — 不覆盖已有键（常用于从仓库根 .env 补缺 ``LLM_*``）；
    3. ``{workspace}/环节变量.env`` — 覆盖上一步写入的同名键。

    ``workspace`` 为 ``None`` 时默认 **file_flow 包目录**（与 ``pipeline.json`` 同级）。
    返回 ``(已加载文件的绝对路径列表, 是否因未安装 python-dotenv 而跳过)``。
    """
    global _done, _loaded_paths, _dotenv_unavailable
    if _done:
        return list(_loaded_paths), _dotenv_unavailable

    root = workspace if workspace is not None else Path(__file__).resolve().parent

    try:
        from dotenv import load_dotenv
    except ImportError:
        _dotenv_unavailable = True
        _done = True
        return [], True

    loaded: list[Path] = []
    steps: list[tuple[Path, bool]] = [
        (root / ".env", False),
    ]
    if root.parent != root:
        steps.append((root.parent / ".env", False))
    steps.append((root / "环节变量.env", True))
    for path, override in steps:
        if path.is_file():
            load_dotenv(path, override=override)
            loaded.append(path.resolve())

    _loaded_paths = loaded
    _done = True
    return list(_loaded_paths), False
