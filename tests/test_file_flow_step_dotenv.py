# -*- coding: utf-8 -*-
"""file_flow.step_dotenv：上一级目录 .env 补缺加载。"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

pytest.importorskip("dotenv")


def test_parent_dotenv_fills_llm_api_base(tmp_path: Path, monkeypatch) -> None:
    """仅仓库根存在 .env 时，以 file_flow 为 workspace 仍应读到 LLM_API_BASE。"""
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    repo = tmp_path
    ff = repo / "file_flow"
    ff.mkdir(parents=True)
    (repo / ".env").write_text(
        "LLM_API_BASE=https://example.com/v1/chat/completions\n",
        encoding="utf-8",
    )
    import file_flow.step_dotenv as sd

    importlib.reload(sd)
    loaded, missing = sd.ensure_step_dotenv_loaded(ff)
    assert not missing
    assert any(p.name == ".env" for p in loaded)
    assert os.environ.get("LLM_API_BASE") == "https://example.com/v1/chat/completions"


def test_file_flow_dotenv_overrides_parent(tmp_path: Path, monkeypatch) -> None:
    """同级 .env 优先于上一级（两文件均 override=False 时先加载的键保留）。"""
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    repo = tmp_path
    ff = repo / "file_flow"
    ff.mkdir(parents=True)
    (repo / ".env").write_text("LLM_API_BASE=https://parent/v1/chat/completions\n", encoding="utf-8")
    (ff / ".env").write_text("LLM_API_BASE=https://child/v1/chat/completions\n", encoding="utf-8")
    import file_flow.step_dotenv as sd

    importlib.reload(sd)
    sd.ensure_step_dotenv_loaded(ff)
    assert os.environ.get("LLM_API_BASE") == "https://child/v1/chat/completions"
