#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键串联 PDF 抽取与评审信息提取流程。

流程：
1. 读取 PDF 目录，调用 extract_pdf.py 批量抽取为 JSON。
2. 将所有抽取 JSON 合并为一个单独文件（便于集中归档/排查）。
3. 调用 query_extracted_json.py，按评审模板回填“内容”字段，
   输出到 output 目录下模板同名副本（如 output/评审标准.json）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _collect_json_files(json_dir: Path, recursive: bool = False) -> list[Path]:
    """收集目录中的 JSON 文件。"""
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(p.resolve() for p in json_dir.glob(pattern) if p.is_file())


def merge_extracted_jsons(json_dir: Path, merged_output: Path, recursive: bool = False) -> int:
    """合并抽取结果 JSON 到单文件。返回成功合并的文件数。"""
    json_files = _collect_json_files(json_dir, recursive=recursive)
    merged_docs: list[dict[str, Any]] = []

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        merged_docs.append(
            {
                "json_file": str(json_file),
                "source_pdf": str(doc.get("file name", "")),
                "document": doc,
            }
        )

    merged_output.parent.mkdir(parents=True, exist_ok=True)
    merged_payload = {
        "meta": {
            "json_dir": str(json_dir),
            "file_count": len(json_files),
            "merged_count": len(merged_docs),
        },
        "documents": merged_docs,
    }
    with merged_output.open("w", encoding="utf-8") as f:
        json.dump(merged_payload, f, ensure_ascii=False, indent=2)

    return len(merged_docs)


def _run_command(command: list[str]) -> None:
    """执行子命令并在失败时抛出异常。"""
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="串联 PDF 抽取与评审模板提取")
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="待处理 PDF 目录",
    )
    parser.add_argument(
        "--template",
        required=True,
        help="评审标准模板路径（标准 JSON，如 评审标准.json）",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="输出目录（默认 output）",
    )
    parser.add_argument(
        "--extract-config",
        default=None,
        help="extract_pdf.py 的配置文件路径（可选）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理 PDF 子目录（并递归读取抽取 JSON 子目录）",
    )
    parser.add_argument(
        "--merged-json-name",
        default="merged_extracted.json",
        help="合并后的原始抽取文件名（默认 merged_extracted.json）",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    pdf_dir = Path(args.pdf_dir).resolve()
    template_path = Path(args.template).resolve()
    output_dir = Path(args.output_dir).resolve()
    extracted_json_dir = output_dir / "extracted_json"
    merged_json_path = output_dir / args.merged_json_name
    review_output_path = output_dir / template_path.name

    if not pdf_dir.is_dir():
        print(f"PDF 目录不存在: {pdf_dir}", file=sys.stderr)
        return 1
    if not template_path.is_file():
        print(f"模板文件不存在: {template_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_json_dir.mkdir(parents=True, exist_ok=True)

    extract_cmd = [
        sys.executable,
        str(workspace / "extract_pdf.py"),
        str(pdf_dir),
        "--json-output-dir",
        str(extracted_json_dir),
    ]
    if args.recursive:
        extract_cmd.append("--recursive")
    if args.extract_config:
        extract_cmd.extend(["--config", str(Path(args.extract_config).resolve())])

    print("步骤1/3：批量抽取 PDF ...")
    try:
        _run_command(extract_cmd)
    except subprocess.CalledProcessError as exc:
        print(f"抽取失败，退出码: {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print("步骤2/3：合并抽取 JSON ...")
    merged_count = merge_extracted_jsons(
        json_dir=extracted_json_dir,
        merged_output=merged_json_path,
        recursive=args.recursive,
    )
    print(f"已合并 {merged_count} 份抽取结果 -> {merged_json_path}")

    query_cmd = [
        sys.executable,
        str(workspace / "query_extracted_json.py"),
        "--json-dir",
        str(extracted_json_dir),
        "--template",
        str(template_path),
        "-o",
        str(review_output_path),
    ]
    if args.recursive:
        query_cmd.append("--recursive")

    print("步骤3/3：按评审模板提取并回填内容字段 ...")
    try:
        _run_command(query_cmd)
    except subprocess.CalledProcessError as exc:
        print(f"评审提取失败，退出码: {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print("流程完成。")
    print(f"- 原始抽取合并文件: {merged_json_path}")
    print(f"- 评审结果文件: {review_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
