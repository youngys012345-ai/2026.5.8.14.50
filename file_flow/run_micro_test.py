#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Micro test：用 schema 前 10% document_types + standards 前几条，跑完整管线。

用法（在仓库根执行）::

    python -m file_flow.run_micro_test
    python -m file_flow.run_micro_test --schema-pct 20 --standards-n 5
    python -m file_flow.run_micro_test --config my_pipeline.json --dry-run

产出文件（out/ 目录下）::

    schema_micro.json          — 裁剪后的 schema
    standards_micro.json       — 裁剪后的 standards
    field_review_items_micro.json — 基于裁剪 schema 生成的字段关联
    {pdf_stem}_micro_work.json         — 工作 JSON
    {pdf_stem}_micro_review.json       — 评审结果
    {pdf_stem}_micro_review.html       — HTML 报告
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_FILE_FLOW_DIR = Path(__file__).resolve().parent


def _resolve_schema_path(merged: dict, workspace: Path) -> Path:
    raw = merged.get("file_flow_schema_json")
    if isinstance(raw, str) and raw.strip():
        p = Path(raw.strip())
        if p.is_absolute():
            return p.resolve()
        hit = (Path.cwd() / p).resolve()
        if hit.is_file():
            return hit
        return (workspace / p).resolve()
    return workspace / "out" / "schema_example.json"


def _resolve_standards_path(merged: dict, workspace: Path) -> Path:
    raw = merged.get("file_flow_standards_json")
    if isinstance(raw, str) and raw.strip():
        p = Path(raw.strip())
        if p.is_absolute():
            return p.resolve()
        hit = (Path.cwd() / p).resolve()
        if hit.is_file():
            return hit
        return (workspace / p).resolve()
    return workspace / "out" / "standards_example.json"


def trim_schema(schema: dict, keep_pct: float) -> dict:
    """保留前 keep_pct%（至少 1 条）的 document_types。"""
    docs = schema.get("document_types", [])
    if not isinstance(docs, list) or not docs:
        return schema
    n = max(1, math.ceil(len(docs) * keep_pct / 100))
    out = dict(schema)
    out["document_types"] = docs[:n]
    return out


def trim_standards(standards: dict, keep_n: int) -> dict:
    """保留前 keep_n 条 items。"""
    if isinstance(standards, dict):
        items = standards.get("items", [])
        if isinstance(items, list) and items:
            out = dict(standards)
            out["items"] = items[:keep_n]
            return out
    return standards


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Micro test：裁剪 schema / standards，跑完整 file_flow 管线"
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="管线 JSON；默认 file_flow/pipeline.json",
    )
    ap.add_argument(
        "--schema-pct",
        type=float,
        default=10.0,
        help="schema document_types 保留百分比（默认 10.0）",
    )
    ap.add_argument(
        "--standards-n",
        type=int,
        default=3,
        help="standards items 保留前 N 条（默认 3）",
    )
    ap.add_argument("--dry-run", action="store_true", help="各 LLM 步骤 dry-run")
    ns = ap.parse_args(argv)

    from .pipeline_merge import (
        file_flow_root,
        load_merged_pipeline_config,
        resolve_pipeline_disk_path,
        run_file_flow,
    )
    from .aggregate_review_items import aggregate_field_items
    from .step_dotenv import ensure_step_dotenv_loaded

    ensure_step_dotenv_loaded(_FILE_FLOW_DIR)

    ws = _FILE_FLOW_DIR
    disk = resolve_pipeline_disk_path(ws, ns.config)
    merged = load_merged_pipeline_config(disk if disk is not None and disk.is_file() else None)

    # --- 1. 裁剪 schema ---
    schema_path = _resolve_schema_path(merged, ws)
    if not schema_path.is_file():
        print(f"错误: 找不到 schema: {schema_path}", file=sys.stderr)
        return 1
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    trimmed_schema = trim_schema(schema, ns.schema_pct)
    micro_schema_path = ws / "out" / "schema_micro.json"
    micro_schema_path.parent.mkdir(parents=True, exist_ok=True)
    micro_schema_path.write_text(
        json.dumps(trimmed_schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    kept_docs = len(trimmed_schema.get("document_types", []))
    print(f"[micro] schema: {schema_path.name} → {micro_schema_path.name} "
          f"（{kept_docs} 个 document_types，原 {len(schema.get('document_types', []))} 个）")

    # --- 2. 裁剪 standards ---
    standards_path = _resolve_standards_path(merged, ws)
    if not standards_path.is_file():
        print(f"错误: 找不到 standards: {standards_path}", file=sys.stderr)
        return 1
    standards = json.loads(standards_path.read_text(encoding="utf-8"))
    if not isinstance(standards, dict):
        print("错误: standards JSON 顶层须为对象", file=sys.stderr)
        return 1
    trimmed_standards = trim_standards(standards, ns.standards_n)
    micro_standards_path = ws / "out" / "standards_micro.json"
    micro_standards_path.write_text(
        json.dumps(trimmed_standards, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    kept_items = len(trimmed_standards.get("items", []))
    print(f"[micro] standards: {standards_path.name} → {micro_standards_path.name} "
          f"（{kept_items} 条 items，原 {len(standards.get('items', []))} 条）")

    # --- 3. 基于裁剪 schema 生成 field_review_items ---
    field_items = aggregate_field_items(trimmed_schema)
    micro_field_items_path = ws / "out" / "field_review_items_micro.json"
    micro_field_items_path.write_text(
        json.dumps(field_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[micro] field_review_items: {micro_field_items_path.name} "
          f"（{len(field_items)} 个编号）")

    # --- 4. 覆写 merged 配置，指向微缩文件 ---
    merged["file_flow_schema_json"] = str(micro_schema_path.resolve())
    merged["file_flow_standards_json"] = str(micro_standards_path.resolve())
    merged["file_flow_field_review_items_json"] = str(micro_field_items_path.resolve())
    # 输出文件加 _micro 后缀，避免与正式运行混淆
    sw = merged.get("file_flow_suffix_work", "_work")
    if not sw.endswith("_micro"):
        merged["file_flow_suffix_work"] = sw + "_micro"
    sr = merged.get("file_flow_suffix_review", "_review")
    if not sr.endswith("_micro"):
        merged["file_flow_suffix_review"] = sr + "_micro"

    print(f"[micro] 启动管线: file_flow_schema={merged['file_flow_schema_json']} "
          f"file_flow_standards={merged['file_flow_standards_json']} "
          f"suffix_work={merged['file_flow_suffix_work']} "
          f"suffix_review={merged['file_flow_suffix_review']}")
    print()

    # --- 5. 跑全流程 ---
    return run_file_flow(
        workspace=ws,
        config_path=None,  # merged 已注入，不再重读磁盘
        merged=merged,
        dry_run=ns.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
