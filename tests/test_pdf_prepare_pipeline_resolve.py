# -*- coding: utf-8 -*-
"""file_flow 默认 pipeline 路径与 PyMuPDF 抽取模式单测。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_resolve_pipeline_disk_path_uses_workspace_pipeline_json(tmp_path: Path) -> None:
    p = tmp_path / "pipeline.json"
    p.write_text(json.dumps({"backend": "opendataloader"}), encoding="utf-8")

    from file_flow.pipeline_merge import resolve_pipeline_disk_path

    got = resolve_pipeline_disk_path(tmp_path, None)
    assert got is not None
    assert got.resolve() == p.resolve()


def test_resolve_pipeline_disk_path_none_when_missing(tmp_path: Path) -> None:
    from file_flow.pipeline_merge import resolve_pipeline_disk_path

    got = resolve_pipeline_disk_path(tmp_path, None)
    assert got is None


def test_pymupdf_mode_extracts_text(tmp_path: Path) -> None:
    try:
        import fitz
    except ImportError:
        pytest.skip("无 PyMuPDF")

    pdf = tmp_path / "t.pdf"
    doc = fitz.open()
    doc.new_page()
    doc[0].insert_text((72, 72), "hello_file_flow")
    doc.save(str(pdf))
    doc.close()

    outd = tmp_path / "out"
    outd.mkdir(parents=True)

    from file_flow.pdf_text_extract import extract_pdf_full_text_unified

    text, meta = extract_pdf_full_text_unified(
        pdf,
        {"file_flow_pdf_text_backend": "pymupdf"},
        workspace=tmp_path,
        cwd=tmp_path,
        out_dir=outd,
    )
    assert "hello_file_flow" in text
    assert meta.get("pdf_text_backend") == "pymupdf"


def test_mineru_backend_uses_pymupdf_with_meta(tmp_path: Path) -> None:
    try:
        import fitz
    except ImportError:
        pytest.skip("无 PyMuPDF")

    pdf = tmp_path / "m.pdf"
    doc = fitz.open()
    doc.new_page()
    doc[0].insert_text((72, 72), "mineru_skipped")
    doc.save(str(pdf))
    doc.close()
    outd = tmp_path / "out2"
    outd.mkdir(parents=True)

    from file_flow.pdf_text_extract import extract_pdf_full_text_unified

    text, meta = extract_pdf_full_text_unified(
        pdf,
        {"backend": "mineru"},
        workspace=tmp_path,
        cwd=tmp_path,
        out_dir=outd,
    )
    assert "mineru_skipped" in text
    assert meta.get("pdf_text_mode") == "mineru_disabled_use_pymupdf"
    assert meta.get("pdf_text_backend") == "pymupdf"
