# -*- coding: utf-8 -*-
"""file_flow 默认 pipeline 路径与 PyMuPDF 抽取模式单测。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_resolve_pipeline_disk_path_prefers_file_flow(tmp_path: Path) -> None:
    (tmp_path / "file_flow").mkdir(parents=True)
    ff = tmp_path / "file_flow" / "pipeline.json"
    ff.write_text(json.dumps({"backend": "opendataloader"}), encoding="utf-8")
    root_p = tmp_path / "pipeline.json"
    root_p.write_text(json.dumps({"backend": "mineru"}), encoding="utf-8")

    from file_flow.pipeline_merge import resolve_pipeline_disk_path

    got = resolve_pipeline_disk_path(tmp_path, None)
    assert got is not None
    assert got.resolve() == ff.resolve()


def test_resolve_pipeline_disk_path_fallback_root(tmp_path: Path) -> None:
    root_p = tmp_path / "pipeline.json"
    root_p.write_text(json.dumps({"backend": "mineru"}), encoding="utf-8")

    from file_flow.pipeline_merge import resolve_pipeline_disk_path

    got = resolve_pipeline_disk_path(tmp_path, None)
    assert got is not None
    assert got.resolve() == root_p.resolve()


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
