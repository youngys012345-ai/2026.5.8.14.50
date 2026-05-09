#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将统一的文档 JSON（kids 树：段落/标题/表格等）导出为 Markdown。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _escape_md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _paragraph_content(node: dict[str, Any]) -> str:
    c = node.get("content")
    return c.strip() if isinstance(c, str) else ""


def _table_to_markdown(table: dict[str, Any]) -> str:
    rows_out: list[list[str]] = []
    for row in table.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        cells: list[str] = []
        for cell in row.get("cells", []) or []:
            if not isinstance(cell, dict):
                cells.append("")
                continue
            parts: list[str] = []
            for kid in cell.get("kids", []) or []:
                if isinstance(kid, dict) and kid.get("type") == "paragraph":
                    t = _paragraph_content(kid)
                    if t:
                        parts.append(t)
            cells.append(" ".join(parts).strip())
        if cells:
            rows_out.append(cells)
    if not rows_out:
        return ""
    ncol = max(len(r) for r in rows_out)
    norm = [r + [""] * (ncol - len(r)) for r in rows_out]
    header = norm[0]
    sep = ["---"] * ncol
    lines = [
        "| " + " | ".join(_escape_md_cell(c) for c in header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in norm[1:]:
        lines.append("| " + " | ".join(_escape_md_cell(c) for c in r) + " |")
    return "\n".join(lines)


def _walk_kids(lines: list[str], kids: list[Any], heading_stack: list[int] | None = None) -> None:
    if heading_stack is None:
        heading_stack = []
    for item in kids:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "heading":
            level = item.get("heading level")
            if not isinstance(level, int) or level < 1:
                level = 1
            level = min(level, 6)
            text = _paragraph_content(item) or (
                item.get("content") if isinstance(item.get("content"), str) else ""
            )
            prefix = "#" * level
            lines.append(f"{prefix} {text.strip()}")
            lines.append("")
        elif t == "paragraph":
            text = _paragraph_content(item)
            if text:
                lines.append(text)
                lines.append("")
        elif t == "table":
            md = _table_to_markdown(item)
            if md:
                lines.append(md)
                lines.append("")
        elif t == "list":
            for li in item.get("list items", []) or []:
                if not isinstance(li, dict):
                    continue
                for kid in li.get("kids", []) or []:
                    if isinstance(kid, dict) and kid.get("type") == "paragraph":
                        txt = _paragraph_content(kid)
                        if txt:
                            lines.append(f"- {txt}")
            lines.append("")
        elif t in ("header", "footer", "text block"):
            _walk_kids(lines, item.get("kids", []) or [], heading_stack)
        elif t == "caption":
            c = _paragraph_content(item)
            if c:
                lines.append(f"*{c}*")
                lines.append("")


def _append_visual_tag_markdown(lines: list[str], document: dict[str, Any]) -> None:
    """
    在正文之后追加视觉标签摘要块；不修改段落/表格中的 OCR 文本。
    细则标签仍仅在 JSON 的 visual_tags / visual_tag_details 中保留。
    """
    stats = document.get("visual_tag_stats")
    if not isinstance(stats, dict):
        return
    summary = stats.get("summary_sentence")
    if not isinstance(summary, str) or not summary.strip():
        return
    total = stats.get("total")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 视觉标签摘要")
    lines.append("")
    if isinstance(total, int):
        lines.append(f"<!-- visual_tag_total: {total} -->")
    lines.append(summary.strip())


def document_to_markdown(document: dict[str, Any]) -> str:
    """从 document 字典生成 Markdown 字符串。"""
    lines: list[str] = []
    title = document.get("title")
    fname = document.get("file name")
    if isinstance(title, str) and title.strip():
        lines.append(f"# {title.strip()}")
        lines.append("")
    elif isinstance(fname, str) and fname.strip():
        lines.append(f"# {Path(fname).stem}")
        lines.append("")

    pages = document.get("number of pages")
    if isinstance(pages, int):
        lines.append(f"<!-- pages: {pages} -->")
        lines.append("")

    _walk_kids(lines, document.get("kids", []) or [])

    _append_visual_tag_markdown(lines, document)

    meta = document.get("extraction_meta")
    if isinstance(meta, dict) and meta.get("backend"):
        lines.append("")
        lines.append("---")
        lines.append(f"<!-- extraction: {meta.get('backend')} -->")

    text = "\n".join(lines).strip()
    return text + "\n" if text else ""
