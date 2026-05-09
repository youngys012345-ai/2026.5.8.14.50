#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinerU 提取适配层（仅支持本地 MinerU 源码仓库）。

职责：
1. 在指定 MinerU 项目根目录下执行 ``python -m mineru.cli.client``；
2. 定位 MinerU 生成的 content_list / content_list_v2；
3. 转换为当前项目既有的文档结构（含 kids、table rows/cells、image 节点等）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

MINERU_CLI_SHIM = "-m"
MINERU_CLI_MODULE = "mineru.cli.client"


class MinerUExtractionError(RuntimeError):
    """MinerU 提取异常。"""


class _TableHTMLParser(HTMLParser):
    """最小化 HTML 表格解析器，仅提取行列文本。"""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._in_cell = False
        self._current_row: list[str] = []
        self._current_cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() in ("td", "th"):
            self._in_cell = True
            self._current_cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in ("td", "th"):
            text = "".join(self._current_cell_parts).strip()
            self._current_row.append(re.sub(r"\s+", " ", text))
            self._in_cell = False
            self._current_cell_parts = []
        elif lowered == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell_parts.append(data)


def _normalize_bbox(raw_bbox: Any) -> list[float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) < 4:
        return None
    out: list[float] = []
    for value in raw_bbox[:4]:
        if not isinstance(value, (int, float)):
            return None
        out.append(float(value))
    return out


def _flatten_v2_content(value: Any) -> str:
    """把 content_list_v2 的嵌套 content 递归拍平为文本。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_flatten_v2_content(v) for v in value]
        return " ".join(p for p in parts if p).strip()
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return str(value["content"]).strip()
        parts = [_flatten_v2_content(v) for v in value.values()]
        return " ".join(p for p in parts if p).strip()
    return ""


def _parse_html_table_rows(table_html: str) -> list[list[str]]:
    parser = _TableHTMLParser()
    try:
        parser.feed(table_html)
        parser.close()
    except Exception:
        return []
    return [row for row in parser.rows if any(cell for cell in row)]


def _build_table_node(
    page_number: int,
    bbox: list[float] | None,
    rows: list[list[str]],
    node_id: int,
) -> dict[str, Any]:
    table_node: dict[str, Any] = {
        "type": "table",
        "id": node_id,
        "page number": page_number,
        "rows": [],
    }
    if bbox is not None:
        table_node["bounding box"] = bbox
    for row_idx, row_cells in enumerate(rows, start=1):
        row_node: dict[str, Any] = {
            "type": "table row",
            "row number": row_idx,
            "cells": [],
        }
        for col_idx, cell_text in enumerate(row_cells, start=1):
            cell_node: dict[str, Any] = {
                "type": "table cell",
                "page number": page_number,
                "row number": row_idx,
                "column number": col_idx,
                "row span": 1,
                "column span": 1,
                "kids": [
                    {
                        "type": "paragraph",
                        "page number": page_number,
                        "content": cell_text.strip(),
                    }
                ],
            }
            row_node["cells"].append(cell_node)
        table_node["rows"].append(row_node)
    return table_node


def _resolve_image_source(content_list_path: Path, image_path: str) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str((content_list_path.parent / p).resolve())


def _convert_content_list_v1(payload: list[dict[str, Any]], content_list_path: Path) -> list[dict[str, Any]]:
    kids: list[dict[str, Any]] = []
    node_id = 1
    for item in payload:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip().lower()
        page_number = int(item.get("page_idx", 0)) + 1
        bbox = _normalize_bbox(item.get("bbox"))

        if item_type == "table":
            table_html = item.get("table_body")
            rows = _parse_html_table_rows(table_html) if isinstance(table_html, str) else []
            if not rows:
                fallback_text = _flatten_v2_content(item.get("table_caption")) or _flatten_v2_content(
                    item.get("table_footnote")
                )
                if fallback_text:
                    rows = [[fallback_text]]
            if rows:
                kids.append(_build_table_node(page_number=page_number, bbox=bbox, rows=rows, node_id=node_id))
                node_id += 1
            continue

        if item_type in ("image", "chart", "seal"):
            img_path = item.get("img_path")
            if isinstance(img_path, str) and img_path:
                image_node: dict[str, Any] = {
                    "type": "image",
                    "id": node_id,
                    "page number": page_number,
                    "source": _resolve_image_source(content_list_path, img_path),
                }
                if bbox is not None:
                    image_node["bounding box"] = bbox
                kids.append(image_node)
                node_id += 1
            continue

        text = _flatten_v2_content(item.get("text"))
        if not text:
            continue
        text_level = item.get("text_level", 0)
        is_heading = isinstance(text_level, int) and text_level > 0
        node: dict[str, Any] = {
            "type": "heading" if is_heading else "paragraph",
            "id": node_id,
            "page number": page_number,
            "content": text,
        }
        if is_heading:
            node["heading level"] = int(text_level)
        if bbox is not None:
            node["bounding box"] = bbox
        kids.append(node)
        node_id += 1
    return kids


def _convert_content_list_v2(payload: list[Any], content_list_path: Path) -> list[dict[str, Any]]:
    kids: list[dict[str, Any]] = []
    node_id = 1
    for page_idx, blocks in enumerate(payload):
        if not isinstance(blocks, list):
            continue
        page_number = page_idx + 1
        for item in blocks:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            bbox = _normalize_bbox(item.get("bbox"))
            content = item.get("content", {})

            if item_type == "title":
                text = _flatten_v2_content(content.get("title_content"))
                if not text:
                    continue
                level = content.get("level", 1) if isinstance(content, dict) else 1
                node: dict[str, Any] = {
                    "type": "heading",
                    "id": node_id,
                    "page number": page_number,
                    "content": text,
                    "heading level": int(level) if isinstance(level, int) and level > 0 else 1,
                }
                if bbox is not None:
                    node["bounding box"] = bbox
                kids.append(node)
                node_id += 1
                continue

            if item_type == "table":
                table_body = content.get("table_body") if isinstance(content, dict) else None
                rows = _parse_html_table_rows(table_body) if isinstance(table_body, str) else []
                if not rows:
                    fallback_text = _flatten_v2_content(content)
                    if fallback_text:
                        rows = [[fallback_text]]
                if rows:
                    kids.append(_build_table_node(page_number=page_number, bbox=bbox, rows=rows, node_id=node_id))
                    node_id += 1
                continue

            if item_type in ("image", "chart", "seal"):
                img_path = content.get("img_path") if isinstance(content, dict) else None
                if not isinstance(img_path, str) or not img_path:
                    img_path = item.get("img_path")
                if isinstance(img_path, str) and img_path:
                    image_node: dict[str, Any] = {
                        "type": "image",
                        "id": node_id,
                        "page number": page_number,
                        "source": _resolve_image_source(content_list_path, img_path),
                    }
                    if bbox is not None:
                        image_node["bounding box"] = bbox
                    kids.append(image_node)
                    node_id += 1
                continue

            text = _flatten_v2_content(content)
            if not text:
                continue
            node = {
                "type": "paragraph",
                "id": node_id,
                "page number": page_number,
                "content": text,
            }
            if bbox is not None:
                node["bounding box"] = bbox
            kids.append(node)
            node_id += 1
    return kids


def convert_mineru_content_list_to_document(content_list_path: Path, source_pdf: Path) -> dict[str, Any]:
    """将 MinerU 的 content_list 输出转为当前项目使用的文档 JSON 结构。"""
    try:
        payload = json.loads(content_list_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MinerUExtractionError(f"读取 MinerU 输出失败: {content_list_path}，原因: {exc}") from exc

    kids: list[dict[str, Any]]
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        kids = _convert_content_list_v2(payload=payload, content_list_path=content_list_path)
    elif isinstance(payload, list):
        kids = _convert_content_list_v1(payload=payload, content_list_path=content_list_path)
    else:
        raise MinerUExtractionError(
            f"MinerU 输出结构不符合预期（仅支持 content_list/content_list_v2 顶层列表）: {content_list_path}"
        )

    pages = {int(item.get("page number")) for item in kids if isinstance(item.get("page number"), int)}
    num_pages = max(pages) if pages else 1
    title = ""
    for item in kids:
        if item.get("type") == "heading":
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                title = content.strip()
                break

    return {
        "file name": source_pdf.name,
        "number of pages": num_pages,
        "title": title or source_pdf.stem,
        "kids": kids,
    }


def locate_mineru_content_list(output_root: Path, pdf_file: Path) -> Path:
    """在 MinerU 输出目录中定位当前 PDF 对应的 content_list 文件。"""
    stem = pdf_file.stem
    candidates = sorted(
        list(output_root.glob(f"**/{stem}_content_list_v2.json"))
        + list(output_root.glob(f"**/{stem}_content_list.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise MinerUExtractionError(f"未找到 MinerU 输出文件: {stem}_content_list(_v2).json")
    return candidates[0]


def _validate_mineru_source_root(root: Path) -> None:
    """确认目录为 MinerU 克隆仓库根（含 mineru 包目录）。"""
    pkg = root / "mineru"
    if not pkg.is_dir():
        raise MinerUExtractionError(
            f"MinerU 源码路径无效（缺少子目录 mineru/）: {root}\n"
            "请配置 mineru_project_root 为本地 MinerU 仓库根路径。"
        )


def run_mineru_cli_for_pdf(
    pdf_file: Path,
    output_root: Path,
    mineru_project_root: str | Path,
    backend: str | None = None,
    api_url: str | None = None,
    model_source: str | None = None,
    mineru_tools_config_json: str | None = None,
    cli_timeout_sec: float | None = None,
) -> Path:
    """在本地 MinerU 源码树中执行 CLI，并返回 content_list 文件路径。"""
    root = Path(mineru_project_root).resolve()
    _validate_mineru_source_root(root)

    cmd: list[str] = [
        sys.executable,
        MINERU_CLI_SHIM,
        MINERU_CLI_MODULE,
        "-p",
        str(pdf_file),
        "-o",
        str(output_root),
    ]
    if backend:
        cmd.extend(["-b", backend])
    if api_url:
        if backend in ("vlm-http-client", "hybrid-http-client"):
            cmd.extend(["-u", api_url])
        else:
            cmd.extend(["--api-url", api_url])

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if model_source:
        env["MINERU_MODEL_SOURCE"] = model_source
    if mineru_tools_config_json:
        env["MINERU_TOOLS_CONFIG_JSON"] = mineru_tools_config_json
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{root}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = str(root)

    try:
        subprocess.run(
            cmd,
            check=True,
            env=env,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            timeout=cli_timeout_sec,
        )
    except OSError as exc:
        raise MinerUExtractionError(
            f"无法启动 MinerU CLI（请在该环境中已安装 MinerU 依赖）: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MinerUExtractionError(
            f"MinerU 子进程超时（{cli_timeout_sec} 秒）。若为首次运行可能在下载模型；"
            "请配置本地模型（mineru_tools_config_json / local_models）或指定常驻 --mineru-api-url。"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise MinerUExtractionError(f"MinerU 解析失败，退出码: {exc.returncode}") from exc

    return locate_mineru_content_list(output_root=output_root, pdf_file=pdf_file)
