#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将统一的文档 JSON（kids 树：段落/标题/表格等）导出为 Markdown。

说明（与后端关系）：
- **OpenDataLoader**：Java 侧可同时产出原生 `.md`（流水线里复制为 ``*_opendataloader_native.md``），
  本模块仍基于统一 JSON 生成 ``*.md`` / ``*_by_page.md``，结构与两种后端一致。
- **MinerU**：适配层只消费 ``content_list`` JSON，再转为统一结构；Markdown 均由本模块由 JSON 导出，
  而非 MinerU CLI 直接给出的成品 MD。
"""

from __future__ import annotations

from collections import defaultdict
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
        elif t == "image":
            src = item.get("source")
            if isinstance(src, str) and src.strip():
                lines.append(f"![]({src.strip()})")
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


def _page_number_for_node(node: dict[str, Any], *, fallback: int = 1) -> int:
    pn = node.get("page number")
    if isinstance(pn, int) and pn >= 1:
        return pn
    return fallback


def _first_heading_text_in_subtree(node: dict[str, Any]) -> str | None:
    """深度优先，返回子树中首个标题纯文本。"""
    t = node.get("type")
    if t == "heading":
        text = _paragraph_content(node)
        if not text and isinstance(node.get("content"), str):
            text = str(node["content"]).strip()
        if text:
            return text.strip()
    for child in node.get("kids", []) or []:
        if isinstance(child, dict):
            got = _first_heading_text_in_subtree(child)
            if got:
                return got
    return None


def _infer_page_heading_label(page_nodes: list[dict[str, Any]]) -> str:
    """从该页顶层节点中推断「页面标题」（首个 heading）。"""
    for n in page_nodes:
        if not isinstance(n, dict):
            continue
        h = _first_heading_text_in_subtree(n)
        if h:
            return h
    return "—"


def _resolve_total_pages(document: dict[str, Any], page_to_nodes: dict[int, list[dict[str, Any]]]) -> int:
    declared = document.get("number of pages")
    from_keys = max(page_to_nodes.keys()) if page_to_nodes else 0
    if isinstance(declared, int) and declared >= 1:
        return max(declared, from_keys)
    return max(from_keys, 1)


def document_to_markdown_by_page(document: dict[str, Any]) -> str:
    """
    按 PDF 页码聚合 Markdown：每页一块，含页码、推断的页面标题与正文，
    便于下游大模型按页引用（RAG 切块、页级问答）。

    分组依据为顶层 ``kids`` 的 ``page number``；缺省页码视为第 1 页。
    """
    lines: list[str] = []
    fname = document.get("file name")
    stem = Path(str(fname)).stem if isinstance(fname, str) and fname.strip() else "document"

    lines.append("<!-- llm_layout: by-page -->")
    lines.append(f"<!-- source_pdf: {stem} -->")
    lines.append("")

    title = document.get("title")
    if isinstance(title, str) and title.strip():
        lines.append(f"# {title.strip()}")
    elif isinstance(fname, str) and fname.strip():
        lines.append(f"# {Path(fname).stem}")
    else:
        lines.append("# （未命名文档）")
    lines.append("")

    pages_decl = document.get("number of pages")
    if isinstance(pages_decl, int):
        lines.append(f"<!-- total_pages_declared: {pages_decl} -->")
        lines.append("")

    page_to_nodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for kid in document.get("kids", []) or []:
        if not isinstance(kid, dict):
            continue
        pn = _page_number_for_node(kid, fallback=1)
        page_to_nodes[pn].append(kid)

    total = _resolve_total_pages(document, page_to_nodes)

    for page_idx in range(1, total + 1):
        nodes = page_to_nodes.get(page_idx, [])
        heading_label = _infer_page_heading_label(nodes)

        lines.append("---")
        lines.append("")
        lines.append(f"## 第 {page_idx} 页 · Page {page_idx}")
        lines.append("")
        lines.append(f"**本页标题:** {heading_label}")
        lines.append("")
        lines.append(f"<!-- page_index: {page_idx} -->")
        lines.append("")

        if not nodes:
            lines.append("*（本页无结构化文本块；可能仅为空白页、未解析图形或信息在其他页。）*")
            lines.append("")
            continue

        _walk_kids(lines, nodes)

    _append_visual_tag_markdown(lines, document)

    meta = document.get("extraction_meta")
    if isinstance(meta, dict) and meta.get("backend"):
        lines.append("")
        lines.append("---")
        lines.append(f"<!-- extraction: {meta.get('backend')} -->")

    text = "\n".join(lines).strip()
    return text + "\n" if text else ""


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
