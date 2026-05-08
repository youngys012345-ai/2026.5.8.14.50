#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地一键流水线（路径全部写在下方常量里）。

用法：直接修改「=== 可修改配置 ===」区域的常量，然后运行：
  python run_pipeline_hardcoded.py

流程与 run_pipeline.py 相同：抽取 PDF → 合并原始 JSON → 按模板回填「内容」。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from run_pipeline import merge_extracted_jsons


# ---------------------------------------------------------------------------
# === 可修改配置（切换数据源与输出位置只需改这里）===
# ---------------------------------------------------------------------------

# 待抽取的 PDF 所在目录（目录内多个 pdf 会一并处理）
PDF_INPUT_DIR = Path(r"D:\agent实践\pdf信息抽取实践\opendataloader_quickstart\pdfs")

# 评审模板 JSON（标准格式，含一级标题与「字段」「内容」等）
TEMPLATE_JSON = Path(r"D:\agent实践\pdf信息抽取实践\opendataloader_quickstart\example.json")

# 本轮任务的输出根目录（将创建 extracted_json、合并文件与评审结果）
OUTPUT_ROOT = Path(r"D:\agent实践\pdf信息抽取实践\opendataloader_quickstart\output_run")

# 合并后的原始抽取结果文件名（保存在 OUTPUT_ROOT 下）
MERGED_RAW_NAME = "merged_extracted.json"

# 抽取阶段传给 extract_pdf.py 的配置文件；不需要则设为 None
EXTRACT_PIPELINE_CONFIG: Path | None = None
# 示例：Path(r"D:\agent实践\pdf信息抽取实践\opendataloader_quickstart\config\pipeline.example.json")

# 是否递归扫描子目录中的 PDF / JSON
RECURSIVE_PDF = False

# ---------------------------------------------------------------------------
# 以下为脚本内部路径（一般不用改）
# ---------------------------------------------------------------------------

_WORKSPACE = Path(__file__).resolve().parent
_EXTRACT_SCRIPT = _WORKSPACE / "extract_pdf.py"
_QUERY_SCRIPT = _WORKSPACE / "query_extracted_json.py"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    pdf_dir = PDF_INPUT_DIR.resolve()
    template_path = TEMPLATE_JSON.resolve()
    output_dir = OUTPUT_ROOT.resolve()
    extracted_json_dir = output_dir / "extracted_json"
    merged_path = output_dir / MERGED_RAW_NAME
    review_out = output_dir / template_path.name

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
        str(_EXTRACT_SCRIPT),
        str(pdf_dir),
        "--json-output-dir",
        str(extracted_json_dir),
    ]
    if RECURSIVE_PDF:
        extract_cmd.append("--recursive")
    if EXTRACT_PIPELINE_CONFIG is not None:
        cfg = EXTRACT_PIPELINE_CONFIG.resolve()
        if not cfg.is_file():
            print(f"抽取配置文件不存在: {cfg}", file=sys.stderr)
            return 1
        extract_cmd.extend(["--config", str(cfg)])

    print("步骤1/3：批量抽取 PDF ...")
    try:
        _run(extract_cmd)
    except subprocess.CalledProcessError as exc:
        print(f"抽取失败，退出码: {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print("步骤2/3：合并抽取 JSON ...")
    n = merge_extracted_jsons(
        json_dir=extracted_json_dir,
        merged_output=merged_path,
        recursive=RECURSIVE_PDF,
    )
    print(f"已合并 {n} 份 -> {merged_path}")

    query_cmd = [
        sys.executable,
        str(_QUERY_SCRIPT),
        "--json-dir",
        str(extracted_json_dir),
        "--template",
        str(template_path),
        "-o",
        str(review_out),
    ]
    if RECURSIVE_PDF:
        query_cmd.append("--recursive")

    print("步骤3/3：按模板提取并回填「内容」...")
    try:
        _run(query_cmd)
    except subprocess.CalledProcessError as exc:
        print(f"提取失败，退出码: {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print("完成。")
    print(f"  合并原始抽取: {merged_path}")
    print(f"  评审回填结果: {review_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
