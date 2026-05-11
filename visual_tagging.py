#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 PDF 抽取结果中的图片进行视觉标签识别，并回填到可被字段检索覆盖的节点。

优先顺序：
1. 与图片包围盒相交的表格行：向该行每一个单元格写入标签（整行与关键词行匹配时均可汇总到 visual_tags）；
2. 与图片包围盒相交的段落；
3. 同页上包围盒中心距离最近的段落或单元格（回退）。
同时在 image 节点上写入 visual_classification，便于与其它导出字段对齐追溯。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


VISUAL_LABELS = ("手写签名", "指印", "印章")


@dataclass
class AnchorNode:
    """候选锚点：可承载视觉标签的段落或表格单元格。"""

    node: dict[str, Any]
    page_number: int
    bbox: list[float]


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance(a: list[float], b: list[float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def _bbox_intersects(a: list[float], b: list[float]) -> bool:
    """判断两个轴对齐矩形是否相交（PDF 页坐标系，左下或左上依渲染约定）。"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _point_in_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
    x, y = point
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


def _is_valid_bbox(bbox: Any) -> bool:
    return isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)


def collect_anchor_nodes(document: dict[str, Any]) -> list[AnchorNode]:
    """收集可被标注的段落和表格单元格。"""
    anchors: list[AnchorNode] = []

    for item in document.get("kids", []) or []:
        if not isinstance(item, dict):
            continue

        if item.get("type") == "paragraph":
            page_number = item.get("page number")
            bbox = item.get("bounding box")
            if isinstance(page_number, int) and _is_valid_bbox(bbox):
                anchors.append(AnchorNode(node=item, page_number=page_number, bbox=bbox))
            continue

        if item.get("type") == "table":
            for row in item.get("rows", []) or []:
                if not isinstance(row, dict):
                    continue
                for cell in row.get("cells", []) or []:
                    if not isinstance(cell, dict):
                        continue
                    page_number = cell.get("page number") or item.get("page number")
                    bbox = cell.get("bounding box")
                    if isinstance(page_number, int) and _is_valid_bbox(bbox):
                        anchors.append(AnchorNode(node=cell, page_number=page_number, bbox=bbox))

    return anchors


def iter_image_nodes(document: dict[str, Any]) -> list[dict[str, Any]]:
    """收集图片节点。"""
    image_nodes: list[dict[str, Any]] = []
    for item in document.get("kids", []) or []:
        if isinstance(item, dict) and item.get("type") == "image":
            image_nodes.append(item)
    return image_nodes


def _append_visual_tag(target_node: dict[str, Any], label: str, source: str, score: float) -> None:
    """把视觉标签写入目标节点。"""
    tags = target_node.setdefault("visual_tags", [])
    if label not in tags:
        tags.append(label)

    details = target_node.setdefault("visual_tag_details", [])
    details.append(
        {
            "label": label,
            "source": source,
            "score": round(score, 4),
        }
    )


def _label_count_phrase(label: str, count: int) -> str:
    """将标签计数转换为中文短语。"""
    if label == "手写签名":
        return f"{count}个人的手写签名"
    return f"{count}个{label}"


def build_visual_summary_sentence(label_counts: dict[str, int]) -> str:
    """把视觉标签计数汇总成一句中文描述。"""
    ordered_phrases: list[str] = []
    for label in VISUAL_LABELS:
        count = label_counts.get(label, 0)
        if count > 0:
            ordered_phrases.append(_label_count_phrase(label, count))
    if not ordered_phrases:
        return "未识别到手写签名、指印或印章"
    return f"存在{'和'.join(ordered_phrases)}。"


def _find_nearest_anchor(image_node: dict[str, Any], anchors: list[AnchorNode]) -> AnchorNode | None:
    page_number = image_node.get("page number")
    image_bbox = image_node.get("bounding box")
    if not isinstance(page_number, int) or not _is_valid_bbox(image_bbox):
        return None

    same_page_anchors = [anchor for anchor in anchors if anchor.page_number == page_number]
    if not same_page_anchors:
        return None

    return min(same_page_anchors, key=lambda a: _distance(image_bbox, a.bbox))


def _find_overlapping_table_row(
    document: dict[str, Any],
    image_bbox: list[float],
    page_number: int,
) -> dict[str, Any] | None:
    """查找与图片相交的表格行（任一角落在图片内或与图片相交即视为命中该行）。"""
    for item in document.get("kids", []) or []:
        if not isinstance(item, dict) or item.get("type") != "table":
            continue
        for row in item.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            for cell in row.get("cells", []) or []:
                if not isinstance(cell, dict):
                    continue
                cell_page = cell.get("page number") or item.get("page number")
                cell_bbox = cell.get("bounding box")
                if cell_page != page_number or not _is_valid_bbox(cell_bbox):
                    continue
                if _bbox_intersects(image_bbox, cell_bbox):
                    return row
                icx, icy = _bbox_center(image_bbox)
                if _point_in_bbox((icx, icy), cell_bbox):
                    return row
    return None


def _find_overlapping_paragraph(
    document: dict[str, Any],
    image_bbox: list[float],
    page_number: int,
) -> dict[str, Any] | None:
    """查找与图片相交或包含图片中心的段落。"""
    center = _bbox_center(image_bbox)
    for item in document.get("kids", []) or []:
        if not isinstance(item, dict) or item.get("type") != "paragraph":
            continue
        if item.get("page number") != page_number:
            continue
        pb = item.get("bounding box")
        if not _is_valid_bbox(pb):
            continue
        if _bbox_intersects(image_bbox, pb) or _point_in_bbox(center, pb):
            return item
    return None


def _apply_tag_to_table_row(row: dict[str, Any], label: str, source: str, score: float) -> None:
    """将标签写入该行每一个单元格，保证按行抽取时能汇总到 visual_tags。"""
    for cell in row.get("cells", []) or []:
        if isinstance(cell, dict):
            _append_visual_tag(cell, label, source, score)


def build_clip_detector() -> Callable[[Path], tuple[str, float]]:
    """构建基于 CLIP 的三分类检测器（签名/指印/印章）。"""
    try:
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "缺少视觉模型依赖，请安装: pip install pillow transformers torch"
        ) from exc

    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    prompts = [
        "a handwritten signature on document",
        "a fingerprint mark on paper",
        "an official red stamp or seal on document",
    ]

    def _detect(image_path: Path) -> tuple[str, float]:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_idx = int(probs.argmax().item())
        return VISUAL_LABELS[best_idx], float(probs[best_idx].item())

    return _detect


def enrich_document_with_visual_tags(
    document: dict[str, Any],
    json_file_path: Path,
    detect_fn: Callable[[Path], tuple[str, float]] | None = None,
    min_score: float = 0.5,
) -> int:
    """为文档中的图片识别视觉类别，并优先回填到与图片几何相交的表格行或段落。"""
    json_file_path = Path(json_file_path)
    detector = detect_fn or build_clip_detector()
    anchors = collect_anchor_nodes(document)
    if not anchors:
        document["visual_tag_stats"] = {
            "total": 0,
            "counts": {label: 0 for label in VISUAL_LABELS},
            "summary_sentence": build_visual_summary_sentence({}),
        }
        return 0

    hit_count = 0
    label_counts = {label: 0 for label in VISUAL_LABELS}
    for image_node in iter_image_nodes(document):
        source = image_node.get("source")
        if not isinstance(source, str) or not source:
            continue

        image_path = (json_file_path.parent / source).resolve()
        if not image_path.is_file():
            continue

        image_bbox = image_node.get("bounding box")
        page_number = image_node.get("page number")
        if not _is_valid_bbox(image_bbox) or not isinstance(page_number, int):
            continue

        label, score = detector(image_path)
        if score < min_score:
            continue

        anchor_mode = "none"
        overlapping_row = _find_overlapping_table_row(document, image_bbox, page_number)
        if overlapping_row is not None:
            _apply_tag_to_table_row(overlapping_row, label, source, score)
            anchor_mode = "table_row_overlap"
        else:
            overlapping_para = _find_overlapping_paragraph(document, image_bbox, page_number)
            if overlapping_para is not None:
                _append_visual_tag(overlapping_para, label, source, score)
                anchor_mode = "paragraph_overlap"
            else:
                nearest_anchor = _find_nearest_anchor(image_node, anchors)
                if nearest_anchor is None:
                    continue
                _append_visual_tag(nearest_anchor.node, label, source, score)
                anchor_mode = "nearest"

        image_node["visual_classification"] = {
            "label": label,
            "score": round(score, 4),
        }
        image_node["visual_anchor_mode"] = anchor_mode

        hit_count += 1

        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    document["visual_tag_stats"] = {
        "total": hit_count,
        "counts": label_counts,
        "summary_sentence": build_visual_summary_sentence(label_counts),
    }
    return hit_count
