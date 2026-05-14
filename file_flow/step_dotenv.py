#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅供 ``file_flow`` 使用：从仓库根与当前工作目录加载 ``.env`` / ``环节变量.env``。

与仓库根 ``step_dotenv.py`` 逻辑一致，但独立维护，避免 ``file_flow`` 依赖父目录同名模块。
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
    2. ``{workspace}/环节变量.env`` — 覆盖上一步写入的同名键；
    3. ``{cwd}/.env``、``{cwd}/环节变量.env`` — 后加载者覆盖前者。

    ``workspace`` 应传**仓库根**（含 ``pipeline.json`` 的目录）。
    返回 ``(已加载文件的绝对路径列表, 是否因未安装 python-dotenv 而跳过)``。
    """
    global _done, _loaded_paths, _dotenv_unavailable
    if _done:
        return list(_loaded_paths), _dotenv_unavailable

    root = workspace if workspace is not None else Path(__file__).resolve().parent.parent

    try:
        from dotenv import load_dotenv
    except ImportError:
        _dotenv_unavailable = True
        _done = True
        return [], True

    loaded: list[Path] = []
    steps: list[tuple[Path, bool]] = [
        (root / ".env", False),
        (root / "环节变量.env", True),
        (Path.cwd() / ".env", True),
        (Path.cwd() / "环节变量.env", True),
    ]
    for path, override in steps:
        if path.is_file():
            load_dotenv(path, override=override)
            loaded.append(path.resolve())

    _loaded_paths = loaded
    _done = True
    return list(_loaded_paths), False
