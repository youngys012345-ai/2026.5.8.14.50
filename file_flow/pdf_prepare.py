#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件模式第一步：从 PDF 目录读取 PDF，用 PyMuPDF 抽取全文，按 schema（如 result_json/评审标准.json）
深拷贝生成「工作用 JSON」：为每个栏目写入「问题」（由「要求」拼接）、「内容」（当前版本为整篇全文占位，便于后续你改为精准匹配）、「回答」置空。

脚本在 ``file_flow/`` 下，导入时会执行 ``ensure_step_dotenv_loaded(项目根)``，因此会加载**仓库根**下的
``.env`` / ``环节变量.env``（与 ``step_dotenv`` 约定一致），便于后续步骤与管线共用 ``LLM_*`` 等变量。

用法（在项目根执行）::

    python file_flow/pdf_prepare.py --pdf-dir ./你的pdf文件夹 --schema result_json/评审标准.json --out file_flow/out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from step_dotenv import ensure_step_dotenv_loaded  # noqa: E402

ensure_step_dotenv_loaded(_ROOT)


def _requirements_to_question(reqs: Any) -> str:
    """将「要求」列表或单值转为一条「问题」正文。"""
    if isinstance(reqs, list):
        lines = [str(x).strip() for x in reqs if str(x).strip()]
        return "\n".join(lines)
    if reqs is None:
        return ""
    return str(reqs).strip()


def _deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))


def build_work_json_from_schema_and_text(schema_root: dict[str, Any], full_text: str) -> dict[str, Any]:
    """
    在 schema 副本上为每个栏目补充：问题、回答；并将「内容」设为当前抽取全文（占位，后续可改为按文书切片等）。
    """
    out: dict[str, Any] = _deep_copy(schema_root)
    for _section_name, block in out.items():
        if not isinstance(block, dict):
            continue
        fields_obj = block.get("字段")
        if not isinstance(fields_obj, dict):
            continue
        for _field_name, field_obj in fields_obj.items():
            if not isinstance(field_obj, dict):
                continue
            reqs = field_obj.get("要求")
            field_obj["问题"] = _requirements_to_question(reqs)
            field_obj["内容"] = full_text
            field_obj["回答"] = ""
    return out


def _extract_pdf_full_text(pdf_path: Path) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    try:
        parts: list[str] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            parts.append(page.get_text("text") or "")
        return "\n".join(parts).strip()
    finally:
        doc.close()


def _resolve(p: Path, cwd: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    hit = (cwd / p).resolve()
    if hit.exists():
        return hit
    return (_ROOT / p).resolve()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PDF 全文抽取 + 按 schema 装配工作 JSON（文件模式）")
    ap.add_argument("--pdf-dir", type=Path, required=True, help="存放 PDF 的目录")
    ap.add_argument(
        "--schema",
        type=Path,
        default=_ROOT / "result_json" / "评审标准.json",
        help="schema JSON 路径，默认 result_json/评审标准.json",
    )
    ap.add_argument("--out", type=Path, default=_ROOT / "file_flow" / "out", help="输出目录")
    ns = ap.parse_args(argv)

    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(_ROOT)
    if dotenv_missing:
        print(
            "警告: 未安装 python-dotenv，已跳过 .env；请 pip install python-dotenv",
            file=sys.stderr,
        )
    elif env_loaded:
        print(f"已加载环境文件: {len(env_loaded)} 个（仓库根 .env / 环节变量.env 等）")

    pdf_dir = _resolve(ns.pdf_dir, Path.cwd())
    schema_path = _resolve(ns.schema, Path.cwd())
    out_dir = _resolve(ns.out, Path.cwd())
    if not pdf_dir.is_dir():
        print(f"错误: 不是目录: {pdf_dir}", file=sys.stderr)
        return 1
    if not schema_path.is_file():
        print(f"错误: 找不到 schema: {schema_path}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw = schema_path.read_text(encoding="utf-8")
        schema_data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as e:
        print(f"错误: 无法解析 schema JSON: {e}", file=sys.stderr)
        return 1
    if not isinstance(schema_data, dict):
        print("错误: schema 根节点须为 JSON 对象", file=sys.stderr)
        return 1

    pdfs = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
    if not pdfs:
        print(f"提示: 目录内无 PDF 文件: {pdf_dir}", file=sys.stderr)
        return 0

    for pdf in pdfs:
        text = _extract_pdf_full_text(pdf)
        work = build_work_json_from_schema_and_text(schema_data, text)
        # 元信息仅作追溯，不参与栏目遍历
        work["_file_flow_meta"] = {
            "pdf_path": str(pdf.resolve()),
            "全文字符数": len(text),
            "schema_path": str(schema_path.resolve()),
        }
        dest = out_dir / f"{pdf.stem}_work.json"
        dest.write_text(json.dumps(work, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出: {dest} （全文 {len(text)} 字符）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
