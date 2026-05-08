#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按模板（example.json）从 OpenDataLoader 导出的 JSON 中提取信息并回填。

所有输入 PDF 对应一份提取结果，统一写入一个输出文件：结构为
``meta`` + ``documents`` 列表，每条含 ``json_file``、``source_pdf``、``result``，
便于同批多份材料、多 title 在单文件内对照，无法仅靠文件名区分时仍可追溯。

规则：
1. 先匹配一级标题（如“立案登记表”），匹配来源为文档 title 与 heading。
2. 一级标题键中若含 “/”，表示并列标题（如 “登记表A/登记表B”），任意一侧在文档中匹配成功即可填写下方字段；检索字段时在两侧 heading 命中的页面范围内合并检索。
3. 若一级标题（含并列）均未匹配，则该标题下所有字段“内容”统一填“内容缺失”。
4. 若一级标题匹配成功，再逐个匹配字段名（如“案件来源”）并填充“内容”。
5. 字段匹配优先表格行，其次段落。
6. 若字段“是否需要识别手写体”为“是”，则附加 visual_tag_stats.summary_sentence。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


def _extract_content_from_node(node: dict[str, Any]) -> str:
    """递归提取节点文本内容。"""
    texts: list[str] = []
    content = node.get("content")
    if isinstance(content, str) and content.strip():
        texts.append(content.strip())

    for child in node.get("kids", []) or []:
        if isinstance(child, dict):
            child_text = _extract_content_from_node(child)
            if child_text:
                texts.append(child_text)
    return " ".join(texts).strip()


def _collect_heading_pages(document: dict[str, Any], title_name: str) -> dict[int, list[str]]:
    """按一级标题名称匹配 heading，并返回命中页面。"""
    pattern = re.compile(re.escape(title_name))
    page_titles: dict[int, list[str]] = {}
    for item in document.get("kids", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "heading":
            continue
        page_number = item.get("page number")
        content = item.get("content", "")
        if not isinstance(page_number, int) or not isinstance(content, str):
            continue
        if pattern.search(content):
            page_titles.setdefault(page_number, []).append(content)
    return page_titles


def _extract_row_cells(row: dict[str, Any]) -> list[str]:
    """提取表格行中每个单元格的文本。"""
    row_cells: list[str] = []
    for cell in row.get("cells", []) or []:
        if not isinstance(cell, dict):
            continue
        cell_text = _extract_content_from_node(cell)
        if cell_text:
            row_cells.append(cell_text)
    return row_cells


def _document_has_title(document: dict[str, Any], title_name: str) -> bool:
    """判断一级标题是否存在于文档 title 或 heading。"""
    pattern = re.compile(re.escape(title_name))
    doc_title = document.get("title", "")
    if isinstance(doc_title, str) and pattern.search(doc_title):
        return True
    return bool(_collect_heading_pages(document, title_name))


def _split_level1_key(level1_key: str) -> list[str]:
    """将模板中的一级标题键拆分为并列标题（以 “/” 分隔，两端空白去除）。"""
    parts = [p.strip() for p in level1_key.split("/")]
    return [p for p in parts if p]


def _level1_title_matches_document(document: dict[str, Any], level1_key: str) -> bool:
    """并列标题：任一分支在 title 或 heading 中出现即视为一级标题命中。"""
    alternatives = _split_level1_key(level1_key)
    if not alternatives:
        return _document_has_title(document, level1_key.strip())
    return any(_document_has_title(document, alt) for alt in alternatives)


def _merged_heading_pages(document: dict[str, Any], level1_key: str) -> dict[int, list[str]]:
    """并列标题下，合并各分支在 heading 上命中的页面（用于限定字段检索范围）。"""
    merged: dict[int, list[str]] = {}
    for alt in _split_level1_key(level1_key):
        for page_num, titles in _collect_heading_pages(document, alt).items():
            merged.setdefault(page_num, []).extend(titles)
    return merged


def _search_field_content(
    document: dict[str, Any],
    field_name: str,
    allowed_pages: set[int] | None,
) -> str | None:
    """搜索字段内容，优先表格行，其次段落。"""
    keyword_pattern = re.compile(re.escape(field_name))
    for item in document.get("kids", []) or []:
        if not isinstance(item, dict):
            continue
        page_number = item.get("page number")
        if not isinstance(page_number, int):
            continue
        if allowed_pages is not None and page_number not in allowed_pages:
            continue

        if item.get("type") == "table":
            for row in item.get("rows", []) or []:
                if not isinstance(row, dict):
                    continue
                row_cells = _extract_row_cells(row)
                row_text = " ".join(row_cells)
                if row_text and keyword_pattern.search(row_text):
                    return row_text

        if item.get("type") == "paragraph":
            paragraph = item.get("content", "")
            if isinstance(paragraph, str) and paragraph and keyword_pattern.search(paragraph):
                return paragraph

    return None


def _get_visual_summary(document: dict[str, Any]) -> str | None:
    stats = document.get("visual_tag_stats")
    if not isinstance(stats, dict):
        return None
    summary = stats.get("summary_sentence")
    return summary if isinstance(summary, str) and summary else None


def _fill_template_for_document(
    document: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any]:
    """按模板回填单个文档的提取结果。"""
    result = copy.deepcopy(template)
    visual_summary = _get_visual_summary(document)

    for level1_title, level1_body in result.items():
        if not isinstance(level1_body, dict):
            continue
        field_map = level1_body.get("字段")
        if not isinstance(field_map, dict):
            continue

        title_found = _level1_title_matches_document(document, level1_title)
        if not title_found:
            for field_obj in field_map.values():
                if isinstance(field_obj, dict):
                    field_obj["内容"] = "内容缺失"
            continue

        title_pages = _merged_heading_pages(document, level1_title)
        allowed_pages: set[int] | None = set(title_pages.keys()) if title_pages else None

        for field_name, field_obj in field_map.items():
            if not isinstance(field_obj, dict):
                continue
            matched_content = _search_field_content(document, field_name, allowed_pages)
            if not matched_content:
                field_obj["内容"] = "内容缺失"
                continue

            need_visual = str(field_obj.get("是否需要识别手写体", "")).strip() == "是"
            if need_visual and visual_summary:
                field_obj["内容"] = f"{matched_content}；视觉识别摘要：{visual_summary}"
            else:
                field_obj["内容"] = matched_content

    return result


def _load_template_json(path: Path) -> dict[str, Any]:
    """加载模板文件，须为符合标准的 JSON 对象。"""
    text = path.read_text(encoding="utf-8")
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"模板文件必须是 JSON 对象: {path}")
    return loaded


def _collect_json_files(json_dir: Path, recursive: bool = False) -> list[Path]:
    """收集待查询的 JSON 文件。"""
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(p.resolve() for p in json_dir.glob(pattern) if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="按模板批量提取 JSON 并输出结构化结果")
    parser.add_argument(
        "--json-dir",
        required=True,
        help="JSON 文件目录（由 extract_pdf.py 生成）",
    )
    parser.add_argument(
        "--template",
        required=True,
        help="模板文件路径（如 example.json）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="structured_extract_result.json",
        help="整合输出文件路径（单文件包含全部 PDF 的 documents 列表）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归检索 JSON 子目录",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir).resolve()
    if not json_dir.is_dir():
        print(f"JSON 目录不存在: {json_dir}")
        return 1

    json_files = _collect_json_files(json_dir, recursive=args.recursive)
    if not json_files:
        print(f"未在目录中找到 JSON 文件: {json_dir}")
        return 1

    template_path = Path(args.template).resolve()
    if not template_path.is_file():
        print(f"模板文件不存在: {template_path}")
        return 1
    try:
        template = _load_template_json(template_path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"模板文件读取失败: {template_path}，原因: {exc}")
        return 1

    extracted_docs: list[dict[str, Any]] = []
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                document = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"跳过文件（读取失败）: {json_path}，原因: {exc}")
            continue
        extracted_docs.append(
            {
                "json_file": str(json_path),
                "source_pdf": str(document.get("file name", "")),
                "result": _fill_template_for_document(document=document, template=template),
            }
        )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "meta": {
            "json_dir": str(json_dir),
            "template": str(template_path),
            "file_count": len(json_files),
            "processed_count": len(extracted_docs),
        },
        "documents": extracted_docs,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"提取完成。扫描 JSON: {len(json_files)}，成功处理: {len(extracted_docs)}")
    print(f"输出文件: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
