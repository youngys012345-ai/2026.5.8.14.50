#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenDataLoader PDF（opendataloader_pdf）适配：调用 Java 管线，产出 JSON / Markdown 路径。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class OpenDataLoaderExtractionError(RuntimeError):
    """OpenDataLoader 提取异常。"""


def locate_opendataloader_json(work_dir: Path, pdf_file: Path) -> Path:
    """OpenDataLoader 默认在输出目录根下生成 ``<stem>.json``。"""
    stem = pdf_file.stem
    direct = work_dir / f"{stem}.json"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(work_dir.glob(f"**/{stem}.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise OpenDataLoaderExtractionError(f"未在 {work_dir} 找到 OpenDataLoader JSON: {stem}.json")
    return candidates[0].resolve()


def locate_opendataloader_markdown(work_dir: Path, pdf_file: Path) -> Path | None:
    stem = pdf_file.stem
    direct = work_dir / f"{stem}.md"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(work_dir.glob(f"**/{stem}.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].resolve() if candidates else None


def run_opendataloader_for_pdf(
    pdf_file: Path,
    work_dir: Path,
    *,
    table_method: str = "cluster",
    reading_order: str = "xycut",
    hybrid: str | None = None,
    hybrid_url: str | None = None,
    hybrid_mode: str = "auto",
    hybrid_timeout: str = "0",
    hybrid_fallback: bool = False,
    quiet: bool = False,
) -> tuple[Path, Path | None]:
    """调用 ``opendataloader_pdf.convert``，返回 (json_path, markdown_path_or_none)。"""
    try:
        import opendataloader_pdf
    except ImportError as exc:
        raise OpenDataLoaderExtractionError(
            "未安装 opendataloader-pdf。请执行: pip install -U opendataloader-pdf，并安装 Java 11+。"
        ) from exc

    work_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "input_path": str(pdf_file.resolve()),
        "output_dir": str(work_dir.resolve()),
        "format": ["json", "markdown"],
        "quiet": quiet,
        "table_method": table_method,
        "reading_order": reading_order,
        "hybrid_fallback": hybrid_fallback,
    }
    if hybrid and hybrid != "off":
        kwargs["hybrid"] = hybrid
        if hybrid_url:
            kwargs["hybrid_url"] = hybrid_url
        kwargs["hybrid_mode"] = hybrid_mode
        kwargs["hybrid_timeout"] = hybrid_timeout

    try:
        opendataloader_pdf.convert(**kwargs)
    except Exception as exc:
        raise OpenDataLoaderExtractionError(f"OpenDataLoader 解析失败: {exc}") from exc

    json_path = locate_opendataloader_json(work_dir, pdf_file)
    md_path = locate_opendataloader_markdown(work_dir, pdf_file)
    return json_path, md_path


def load_opendataloader_document(json_path: Path, source_pdf: Path) -> dict[str, Any]:
    """读取 OpenDataLoader 导出的 JSON（已是 kids 结构），补充 extraction_meta。"""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenDataLoaderExtractionError(f"读取 JSON 失败: {json_path}，原因: {exc}") from exc

    if not isinstance(data, dict):
        raise OpenDataLoaderExtractionError("OpenDataLoader JSON 根节点必须是对象")

    meta = data.get("extraction_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["backend"] = "opendataloader"
    meta["opendataloader_json"] = str(json_path.resolve())
    data["extraction_meta"] = meta
    if "file name" not in data:
        data["file name"] = source_pdf.name
    return data
