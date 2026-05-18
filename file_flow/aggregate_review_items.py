#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 schema JSON（与 ``out/schema_example.json`` 一致）中提取所有 ``related_review_items``，
按编号（如 "1.4"、"4.2.1"）聚合，生成两个 JSON 文件：

1. **document_review_items.json** — document 级别：每个编号下有哪些 ``document_name``。
2. **field_review_items.json** — field 级别：每个编号下以 document_name 为键，列出该文书中关联的 field_name 列表。

用法::

    python -m file_flow.aggregate_review_items -s out/schema_example.json -d out/document_review_items.json -f out/field_review_items.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_schema(path: str | Path) -> dict[str, Any]:
    """读取 schema JSON。"""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def aggregate_document_items(
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    遍历 schema 中的 ``document_types``，提取每条 document 的 ``related_review_items``，
    按编号聚合，输出 ``[{number, documents: [document_name, ...]}, ...]``（数组，按 number 排序）。
    """
    temp: dict[str, list[str]] = {}
    docs = schema.get("document_types", [])
    if not isinstance(docs, list):
        return []

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_name = doc.get("document_name", "")
        items = doc.get("related_review_items")
        if not isinstance(items, list):
            continue
        for num in items:
            if not isinstance(num, str) or not num.strip():
                continue
            key = num.strip()
            if key not in temp:
                temp[key] = []
            if doc_name not in temp[key]:
                temp[key].append(doc_name)

    result: list[dict[str, Any]] = []
    for num in sorted(temp.keys(), key=_sort_review_number):
        result.append({"number": num, "documents": temp[num]})
    return result


def aggregate_field_items(
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    遍历 schema 中的 ``document_types``，提取每条 document 下每个 field 的 ``related_review_items``，
    按编号聚合，输出 ``[{number, documents: [{document_name, field_names}, ...]}, ...]``（数组，按 number 排序）。

    同一编号下按 document_name 分组，field_names 为该文书下关联的 field 名称列表。
    """
    from collections import OrderedDict

    temp: dict[str, dict[str, list[str]]] = {}
    docs = schema.get("document_types", [])
    if not isinstance(docs, list):
        return []

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_name = doc.get("document_name", "")
        if not doc_name:
            continue
        fields_list = doc.get("fields")
        if not isinstance(fields_list, list):
            continue
        for field in fields_list:
            if not isinstance(field, dict):
                continue
            field_name = field.get("field_name", "")
            if not field_name:
                continue
            items = field.get("related_review_items")
            if not isinstance(items, list):
                continue
            for num in items:
                if not isinstance(num, str) or not num.strip():
                    continue
                key = num.strip()
                if key not in temp:
                    temp[key] = {}
                if doc_name not in temp[key]:
                    temp[key][doc_name] = []
                if field_name not in temp[key][doc_name]:
                    temp[key][doc_name].append(field_name)

    result: list[dict[str, Any]] = []
    for num in sorted(temp.keys(), key=_sort_review_number):
        docs_list: list[dict[str, Any]] = []
        for d_name in sorted(temp[num].keys()):
            docs_list.append({
                "document_name": d_name,
                "field_names": temp[num][d_name],
            })
        result.append({"number": num, "documents": docs_list})

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="从 schema JSON 中按编号聚合 related_review_items，输出 document 级和 field 级两个 JSON 文件"
    )
    ap.add_argument(
        "-s", "--schema", required=True,
        help="schema JSON 路径（与 out/schema_example.json 一致）",
    )
    ap.add_argument(
        "-d", "--document-output", default="out/document_review_items.json",
        help="document 级聚合结果输出路径（默认 out/document_review_items.json）",
    )
    ap.add_argument(
        "-f", "--field-output", default="out/field_review_items.json",
        help="field 级聚合结果输出路径（默认 out/field_review_items.json）",
    )
    args = ap.parse_args()

    schema = load_schema(args.schema)

    doc_result = aggregate_document_items(schema)
    field_result = aggregate_field_items(schema)

    doc_path = Path(args.document_output)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        json.dumps(doc_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"document 级聚合已写出: {doc_path.resolve()}")

    field_path = Path(args.field_output)
    field_path.parent.mkdir(parents=True, exist_ok=True)
    field_path.write_text(
        json.dumps(field_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"field 级聚合已写出: {field_path.resolve()}")

    # 汇总统计
    doc_nums = [e["number"] for e in doc_result]
    field_nums = [e["number"] for e in field_result]
    print(f"\ndocument 级: {len(doc_nums)} 个编号")
    print(f"field 级: {len(field_nums)} 个编号")
    print(f"field 级特有编号: {sorted(set(field_nums) - set(doc_nums), key=_sort_review_number)}")
    print(f"document 级特有编号: {sorted(set(doc_nums) - set(field_nums), key=_sort_review_number)}")


def _sort_review_number(n: str) -> tuple[int, ...]:
    """将 '1.4' / '4.2.1' 等编号转为可排序的整数元组。"""
    parts = n.split(".")
    result: list[int] = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            result.append(0)
    return tuple(result)


if __name__ == "__main__":
    main()
