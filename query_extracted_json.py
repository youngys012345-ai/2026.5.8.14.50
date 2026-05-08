#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按模板（example.json）从 OpenDataLoader 导出的 JSON 中提取信息并回填。

所有输入 PDF 对应一份提取结果，统一写入一个输出文件：结构为
``meta`` + ``documents`` 列表，每条含 ``json_file``、``source_pdf``、``result``，
便于同批多份材料、多 title 在单文件内对照，无法仅靠文件名区分时仍可追溯。

规则：
1. 先匹配一级标题（如“立案登记表”）：在文档 title 与任意含页码节点中提取的文本上匹配，不依赖节点的 ``type``（识别结果可能将标题标为 heading、paragraph、表格单元格等）。
2. 一级标题键中若含 “/”，表示并列标题；任一分支在文档中出现即视为命中；字段检索限在该分支命中的页面范围内（多页合并为一段连续文本后再抽取）。
3. 若一级标题（含并列）均未匹配，则该标题下所有字段“内容”统一填“内容缺失”。
4. 一级标题命中后，优先基于字段小标题的 bounding box，在该位置右侧或下方的近邻文本块中取值（不向左、不向上取值）。
5. 若右侧/下方近邻未命中，再回退到“小标题后区块正则截取 + 关键词检索”兼容逻辑。
6. 若字段“是否需要识别手写体”为“是”，优先读取该字段目标位置周围最近的 visual_tags（可多个）；若未找到则回退 visual_tag_stats.summary_sentence。
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


def _bbox_center_xy(node: dict[str, Any]) -> tuple[float, float] | None:
    """返回节点 bounding box 的中心坐标；无坐标时返回 None。"""
    bb = node.get("bounding box")
    if isinstance(bb, list) and len(bb) >= 4:
        x0, y0, x1, y1 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return None


def _is_valid_bbox(bbox: Any) -> bool:
    """判断 bbox 是否为 4 个数值。"""
    return isinstance(bbox, list) and len(bbox) == 4 and all(
        isinstance(v, (int, float)) for v in bbox
    )


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _reading_sort_key(
    node: dict[str, Any],
    kid_index: int,
    sub_index: int,
) -> tuple[float, float, int, int]:
    """阅读顺序：先上下、再左右；无 bbox 时用 kids 顺序兜底（避免 object id 打乱段落顺序）。"""
    xy = _bbox_center_xy(node)
    if xy is not None:
        cx, cy = xy
        return (-cy, cx, kid_index, sub_index)
    return (0.0, 0.0, kid_index, sub_index)


def _row_sort_proxy(row: dict[str, Any]) -> dict[str, Any]:
    """表格行排序时用首个带 bounding box 的单元格代理该行位置。"""
    for cell in row.get("cells", []) or []:
        if isinstance(cell, dict) and isinstance(cell.get("bounding box"), list):
            return cell
    return row


def _collect_title_hit_pages(document: dict[str, Any], title_name: str) -> dict[int, list[str]]:
    """按一级标题文本在任意节点上匹配（不限 type），返回命中页面。"""
    pattern = re.compile(re.escape(title_name))
    page_hits: dict[int, list[str]] = {}
    for item in document.get("kids", []) or []:
        if not isinstance(item, dict):
            continue
        page_number = item.get("page number")
        if not isinstance(page_number, int):
            continue
        block_text = _extract_content_from_node(item)
        if not isinstance(block_text, str) or not block_text.strip():
            continue
        if pattern.search(block_text):
            snippet = block_text.strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            page_hits.setdefault(page_number, []).append(snippet)
    return page_hits


def _collect_segments_for_page(document: dict[str, Any], page_num: int) -> list[tuple[tuple[float, float, int, int], str]]:
    """单页内按阅读顺序收集文本片段（表格拆成按行）。"""
    segments: list[tuple[tuple[float, float, int, int], str]] = []
    for kid_index, item in enumerate(document.get("kids", []) or []):
        if not isinstance(item, dict):
            continue
        if item.get("page number") != page_num:
            continue
        if item.get("type") == "table":
            for row_index, row in enumerate(item.get("rows", []) or []):
                if not isinstance(row, dict):
                    continue
                row_cells = _extract_row_cells(row)
                row_text = " ".join(row_cells).strip()
                if not row_text:
                    continue
                proxy = _row_sort_proxy(row)
                segments.append((_reading_sort_key(proxy, kid_index, row_index), row_text))
        else:
            text = _extract_content_from_node(item).strip()
            if text:
                segments.append((_reading_sort_key(item, kid_index, 0), text))
    segments.sort(key=lambda t: t[0])
    return segments


def _build_page_plain_text(document: dict[str, Any], page_num: int) -> str:
    """拼接单页阅读顺序纯文本。"""
    return "\n".join(seg[1] for seg in _collect_segments_for_page(document, page_num))


def _all_pages_in_document(document: dict[str, Any]) -> set[int]:
    """推断文档页码集合：优先 number of pages，否则从 kids 收集。"""
    np = document.get("number of pages")
    if isinstance(np, int) and np >= 1:
        return set(range(1, np + 1))
    pages: set[int] = set()
    for item in document.get("kids", []) or []:
        if isinstance(item, dict):
            pn = item.get("page number")
            if isinstance(pn, int):
                pages.add(pn)
    return pages if pages else {1}


def _merge_pages_plain_text(document: dict[str, Any], page_nums: set[int]) -> str:
    """多页文本按页码顺序合并。"""
    parts: list[str] = []
    for p in sorted(page_nums):
        parts.append(_build_page_plain_text(document, p))
    return "\n\n".join(parts)


def _iter_text_blocks(
    document: dict[str, Any], allowed_pages: set[int] | None
) -> list[dict[str, Any]]:
    """收集可用于字段匹配的文本块（段落与表格单元格，均需有效 bbox）。"""
    blocks: list[dict[str, Any]] = []
    kids = document.get("kids", []) or []
    for kid_index, item in enumerate(kids):
        if not isinstance(item, dict):
            continue
        page_number = item.get("page number")
        if not isinstance(page_number, int):
            continue
        if allowed_pages is not None and page_number not in allowed_pages:
            continue

        if item.get("type") == "table":
            for row_index, row in enumerate(item.get("rows", []) or []):
                if not isinstance(row, dict):
                    continue
                for col_index, cell in enumerate(row.get("cells", []) or []):
                    if not isinstance(cell, dict):
                        continue
                    cell_page = cell.get("page number")
                    if not isinstance(cell_page, int):
                        cell_page = page_number
                    if allowed_pages is not None and cell_page not in allowed_pages:
                        continue
                    bbox = cell.get("bounding box")
                    text = _extract_content_from_node(cell).strip()
                    if not _is_valid_bbox(bbox) or not text:
                        continue
                    blocks.append(
                        {
                            "text": text,
                            "bbox": [float(v) for v in bbox],
                            "page": cell_page,
                            "source_node": cell,
                            "node_tags": [str(t) for t in cell.get("visual_tags", []) if isinstance(t, str)],
                            "sort_key": _reading_sort_key(
                                cell, kid_index, row_index * 1000 + col_index
                            ),
                        }
                    )
            continue

        if item.get("type") == "image":
            continue

        bbox = item.get("bounding box")
        text = _extract_content_from_node(item).strip()
        if _is_valid_bbox(bbox) and text:
            blocks.append(
                {
                    "text": text,
                    "bbox": [float(v) for v in bbox],
                    "page": page_number,
                    "source_node": item,
                    "node_tags": [str(t) for t in item.get("visual_tags", []) if isinstance(t, str)],
                    "sort_key": _reading_sort_key(item, kid_index, 0),
                }
            )

    blocks.sort(key=lambda b: b["sort_key"])
    return blocks


def _score_right_or_below_candidate(target_bbox: list[float], cand_bbox: list[float]) -> float | None:
    """只接受目标右方或下方的候选，返回越小越近的评分。"""
    tx0, ty0, tx1, ty1 = target_bbox
    cx0, cy0, cx1, cy1 = cand_bbox
    t_cx, t_cy = _bbox_center(target_bbox)
    c_cx, c_cy = _bbox_center(cand_bbox)
    t_w = max(1.0, tx1 - tx0)
    t_h = max(1.0, ty1 - ty0)

    right_dx = cx0 - tx1
    right_dy = abs(c_cy - t_cy)
    if right_dx >= -2.0 and right_dx <= max(220.0, t_w * 4.0) and right_dy <= max(30.0, t_h * 1.5):
        return max(0.0, right_dx) + 0.35 * right_dy

    below_dy = ty0 - cy1
    below_dx = abs(c_cx - t_cx)
    if below_dy >= -2.0 and below_dy <= max(260.0, t_h * 6.0) and below_dx <= max(45.0, t_w * 2.0):
        return max(0.0, below_dy) + 0.35 * below_dx + 2.0

    return None


def _extract_field_content_by_bbox(
    document: dict[str, Any],
    field_name: str,
    allowed_pages: set[int] | None,
    peer_field_names: list[str],
) -> tuple[str | None, dict[str, Any] | None]:
    """先定位字段名文本块，再只取其右侧或下侧近邻文本块内容。"""
    blocks = _iter_text_blocks(document, allowed_pages)
    if not blocks:
        return None, None
    key_re = re.compile(re.escape(field_name))
    target_idx = next((idx for idx, b in enumerate(blocks) if key_re.search(b["text"])), None)
    if target_idx is None:
        return None, None

    target = blocks[target_idx]
    inline = _extract_field_block_after_subheading(target["text"], field_name, peer_field_names)
    if inline:
        return inline, target

    best_text: str | None = None
    best_score: float | None = None
    for idx, block in enumerate(blocks):
        if idx == target_idx:
            continue
        if block["page"] != target["page"]:
            continue
        score = _score_right_or_below_candidate(target["bbox"], block["bbox"])
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_text = block["text"].strip()

    return best_text, target


def _slice_from_level1_title(text: str, level1_alternatives: list[str]) -> str:
    """从首个一级标题出现位置起截取，减少页眉等与表单无关的干扰。"""
    best: int | None = None
    for name in level1_alternatives:
        n = name.strip()
        if not n:
            continue
        m = re.search(re.escape(n), text)
        if m:
            pos = m.start()
            best = pos if best is None else min(best, pos)
    if best is None:
        return text
    return text[best:]


def _extract_field_block_after_subheading(
    region_text: str,
    field_name: str,
    peer_field_names: list[str],
) -> str | None:
    """匹配小标题（字段名），截取其后至下一模板字段名之前的区块。"""
    peers = [p for p in peer_field_names if p and p != field_name]
    peers_long_first = sorted(peers, key=len, reverse=True)

    start_re = re.compile(re.escape(field_name) + r"(?:\s*[:：])?")
    m = start_re.search(region_text)
    if not m:
        return None
    rest = region_text[m.end() :]

    if not peers_long_first:
        block = rest.strip()
        return block if block else None

    earliest = len(rest)
    for other in peers_long_first:
        om = re.search(re.escape(other), rest)
        if om:
            earliest = min(earliest, om.start())

    block = rest[:earliest].strip()
    return block if block else None


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
    """判断一级标题是否存在于文档 title，或任意节点的正文文本中。"""
    pattern = re.compile(re.escape(title_name))
    doc_title = document.get("title", "")
    if isinstance(doc_title, str) and pattern.search(doc_title):
        return True
    return bool(_collect_title_hit_pages(document, title_name))


def _split_level1_key(level1_key: str) -> list[str]:
    """将模板中的一级标题键拆分为并列标题（以 “/” 分隔，两端空白去除）。"""
    parts = [p.strip() for p in level1_key.split("/")]
    return [p for p in parts if p]


def _level1_title_matches_document(document: dict[str, Any], level1_key: str) -> bool:
    """并列标题：任一分支在 title 或正文节点中出现即视为一级标题命中。"""
    alternatives = _split_level1_key(level1_key)
    if not alternatives:
        return _document_has_title(document, level1_key.strip())
    return any(_document_has_title(document, alt) for alt in alternatives)


def _merged_heading_pages(document: dict[str, Any], level1_key: str) -> dict[int, list[str]]:
    """并列标题下，合并各分支在正文中命中的页面（用于限定字段检索范围）。"""
    merged: dict[int, list[str]] = {}
    for alt in _split_level1_key(level1_key):
        for page_num, titles in _collect_title_hit_pages(document, alt).items():
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


def _collect_nearest_visual_tags(
    document: dict[str, Any],
    target_block: dict[str, Any] | None,
    allowed_pages: set[int] | None,
) -> list[str]:
    """读取目标附近最近的 visual_tags，可能返回多个标签。"""
    if not target_block:
        return []
    target_bbox = target_block.get("bbox")
    target_page = target_block.get("page")
    if not _is_valid_bbox(target_bbox) or not isinstance(target_page, int):
        return []

    candidates: list[tuple[float, list[str]]] = []
    for block in _iter_text_blocks(document, allowed_pages):
        if block["page"] != target_page:
            continue
        tags = block.get("node_tags")
        if tags is None:
            tags = []
        if not tags and isinstance(block.get("source_node"), dict):
            node_tags = block["source_node"].get("visual_tags", [])
            if isinstance(node_tags, list):
                tags = [str(t) for t in node_tags if isinstance(t, str)]
        if not tags:
            continue

        bx = block.get("bbox")
        if not _is_valid_bbox(bx):
            continue
        t_cx, t_cy = _bbox_center(target_bbox)
        b_cx, b_cy = _bbox_center(bx)
        dist = ((t_cx - b_cx) ** 2 + (t_cy - b_cy) ** 2) ** 0.5
        candidates.append((dist, tags))

    if not candidates:
        return []

    min_dist = min(d for d, _ in candidates)
    tolerance = min_dist + 12.0
    merged: list[str] = []
    for dist, tags in sorted(candidates, key=lambda x: x[0]):
        if dist > tolerance:
            break
        for tag in tags:
            if tag not in merged:
                merged.append(tag)
    return merged


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

        alternatives = _split_level1_key(level1_title) or [level1_title.strip()]
        page_scope = allowed_pages if allowed_pages is not None else _all_pages_in_document(document)
        combined_text = _merge_pages_plain_text(document, page_scope)
        region_text = _slice_from_level1_title(combined_text, alternatives)
        field_keys = [k for k in field_map if isinstance(k, str)]

        for field_name, field_obj in field_map.items():
            if not isinstance(field_obj, dict):
                continue
            matched_content, field_target_block = _extract_field_content_by_bbox(
                document=document,
                field_name=str(field_name),
                allowed_pages=allowed_pages,
                peer_field_names=field_keys,
            )
            if not matched_content:
                matched_content = _extract_field_block_after_subheading(
                    region_text, str(field_name), field_keys
                )
            if not matched_content:
                matched_content = _search_field_content(document, field_name, allowed_pages)
            if not matched_content:
                field_obj["内容"] = "内容缺失"
                continue

            need_visual = str(field_obj.get("是否需要识别手写体", "")).strip() == "是"
            if need_visual:
                nearest_tags = _collect_nearest_visual_tags(
                    document=document,
                    target_block=field_target_block,
                    allowed_pages=allowed_pages,
                )
                if nearest_tags:
                    field_obj["内容"] = f"{matched_content}；视觉标签：{'、'.join(nearest_tags)}"
                elif visual_summary:
                    field_obj["内容"] = f"{matched_content}；视觉识别摘要：{visual_summary}"
                else:
                    field_obj["内容"] = matched_content
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
