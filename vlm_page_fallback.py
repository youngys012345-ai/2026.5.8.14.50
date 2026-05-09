#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抽取文本过短时可选：PyMuPDF 渲染页面 + VLM 转写并合并到 document kids。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def estimate_extracted_text_length(document: dict[str, Any]) -> int:
    """粗略统计 kids 树中文本长度。"""

    def walk_kids(kids: list[Any]) -> int:
        total = 0
        for item in kids:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in ("paragraph", "heading", "caption"):
                c = item.get("content")
                if isinstance(c, str):
                    total += len(c.strip())
            elif t == "table":
                for row in item.get("rows", []) or []:
                    if isinstance(row, dict):
                        for cell in row.get("cells", []) or []:
                            if isinstance(cell, dict):
                                total += walk_kids(cell.get("kids", []) or [])
            elif t == "list":
                for li in item.get("list items", []) or []:
                    if isinstance(li, dict):
                        total += walk_kids(li.get("kids", []) or [])
            elif t in ("header", "footer", "text block"):
                total += walk_kids(item.get("kids", []) or [])
        return total

    kids = document.get("kids", []) or []
    return walk_kids(kids) if isinstance(kids, list) else 0


def document_needs_vlm_fallback(document: dict[str, Any], threshold: int) -> bool:
    try:
        th = max(0, int(threshold))
    except (TypeError, ValueError):
        th = 80
    return estimate_extracted_text_length(document) < th


def render_pdf_pages_to_png(pdf_file: Path, out_dir: Path, dpi: int = 120) -> list[Path]:
    """将 PDF 每页渲染为 PNG 到 out_dir。"""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("VLM 回退需要 PyMuPDF：pip install pymupdf") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_file)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    paths: list[Path] = []
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out = out_dir / f"page_{i + 1:04d}.png"
            pix.save(str(out))
            paths.append(out)
    finally:
        doc.close()
    return paths


def merge_vlm_page_transcripts(
    document: dict[str, Any],
    page_image_paths: list[Path],
    transcribe_fn: Callable[[Path, int], str],
) -> dict[str, Any]:
    """在 document kids 末尾追加 VLM 段落。"""
    kids = document.get("kids")
    if not isinstance(kids, list):
        kids = []
        document["kids"] = kids

    vlm_chunks: list[dict[str, Any]] = []
    max_id = 0

    def scan_ids(items: list[Any]) -> None:
        nonlocal max_id
        for it in items:
            if isinstance(it, dict):
                nid = it.get("id")
                if isinstance(nid, int):
                    max_id = max(max_id, nid)
                scan_ids(it.get("kids", []) or [])

    scan_ids(kids)

    for idx, img_path in enumerate(page_image_paths, start=1):
        text = transcribe_fn(img_path, idx).strip()
        if not text or text.startswith("(无可用文本)"):
            continue
        max_id += 1
        node = {
            "type": "paragraph",
            "id": max_id,
            "page number": idx,
            "content": text,
            "vlm_fallback": True,
            "vlm_role": "page_transcript",
        }
        kids.append(node)
        vlm_chunks.append({"page": idx, "preview": text[:200]})

    meta = document.get("extraction_meta")
    if not isinstance(meta, dict):
        meta = {}
        document["extraction_meta"] = meta
    meta["vlm_fallback_merge"] = {
        "pages_rendered": len(page_image_paths),
        "segments_appended": len(vlm_chunks),
        "chunks": vlm_chunks,
    }
    return document
