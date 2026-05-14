#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件模式第一步：从 PDF 目录读取 PDF，抽取全文（默认走 **OpenDataLoader**（Java + 可选 Hybrid、
``hybrid_fallback`` 兜底）；``backend=mineru`` 在 file_flow 中暂不支持，会自动改用 PyMuPDF）。
可用 ``file_flow_pdf_text_backend=pymupdf`` 强制仅用本地 PyMuPDF。
再按 **document_types** schema（与 ``out/schema_example.json`` 一致）深拷贝生成工作 JSON；
全文写入与 ``*_work.json`` 同目录的 ``{pdf_stem}_fulltext.txt``，**不**再把整篇 PDF 文本复制进各字段 ``content``（非 LLM 模式下 ``content`` 保持空串，供后续环节或人工填写）。
若管线步骤中含 ``schema_llm_extract``，大模型摘录在该步执行；否则可由 ``file_flow_llm_extract`` 在本模块内联完成。

schema **必须**包含非空 ``document_types`` 数组；默认 schema 路径为 ``out/schema_example.json``
（可被 ``pipeline.json`` 的 ``file_flow_schema_json`` 或 ``--schema`` 覆盖）。

用法（在包含 ``file_flow`` 包的上级目录执行）::

    python -m file_flow.pdf_prepare --pdf-dir ./pdfs --schema out/schema_example.json --out ./out
    python -m file_flow.pdf_prepare --config pipeline.json
    python -m file_flow.pdf_prepare ... --llm-extract
    python -m file_flow.pdf_prepare ... --llm-extract --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_FILE_FLOW_DIR = Path(__file__).resolve().parent

from .pipeline_merge import (  # noqa: E402
    load_merged_pipeline_config,
    resolve_pipeline_disk_path,
)
from .step_dotenv import ensure_step_dotenv_loaded  # noqa: E402
from .llm_openai import (  # noqa: E402
    build_llm_env_config,
    configure_logging,
    is_http_endpoint_url,
)
from .pdf_text_extract import extract_pdf_full_text_unified  # noqa: E402
from .naming import work_json_filename_for_stem  # noqa: E402
from .schema_llm_extract import (  # noqa: E402
    enrich_work_json_with_llm_schema_extract,
)

ensure_step_dotenv_loaded(_FILE_FLOW_DIR)


def _deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))


def is_document_types_schema(schema_root: dict[str, Any]) -> bool:
    """是否为带 ``document_types`` 的公文 schema（与 schema_example.json 一致）。"""
    dt = schema_root.get("document_types")
    return isinstance(dt, list) and len(dt) > 0


def _annotate_work_fields(out: dict[str, Any]) -> None:
    """为 workflow 在字段对象上补充 ``content`` / ``answer`` 初值（不改变 schema 原有键）。"""
    docs = out.get("document_types")
    if not isinstance(docs, list):
        return
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        fields = doc.get("fields")
        if not isinstance(fields, list):
            continue
        for field_obj in fields:
            if not isinstance(field_obj, dict):
                continue
            field_obj.setdefault("content", "")
            field_obj.setdefault("answer", "")


def build_work_json_skeleton(schema_root: dict[str, Any]) -> dict[str, Any]:
    """深拷贝 schema，并为各 field 补充 ``content``、``answer`` 空串。"""
    out: dict[str, Any] = _deep_copy(schema_root)
    _annotate_work_fields(out)
    return out


def apply_full_text_to_all_contents(work: dict[str, Any], full_text: str) -> None:
    """占位模式：将同一全文写入每个 field 的 ``content``（就地修改）。"""
    docs = work.get("document_types")
    if not isinstance(docs, list):
        return
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        fields = doc.get("fields")
        if not isinstance(fields, list):
            continue
        for field_obj in fields:
            if isinstance(field_obj, dict):
                field_obj["content"] = full_text


def build_work_json_from_schema_and_text(schema_root: dict[str, Any], full_text: str) -> dict[str, Any]:
    """在 schema 副本上为各 field 写入全文占位 ``content``。"""
    w = build_work_json_skeleton(schema_root)
    apply_full_text_to_all_contents(w, full_text)
    return w


def _resolve(p: Path, cwd: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    hit = (cwd / p).resolve()
    if hit.exists():
        return hit
    return (_FILE_FLOW_DIR / p).resolve()


def resolve_llm_extract_enabled(merged: dict[str, Any], llm_extract: bool | None) -> bool:
    """
    是否启用 schema 大模型摘录。

    - 命令行显式传入 ``True``/``False`` 时以命令行为准；
    - 否则读 ``merged["file_flow_llm_extract"]``；**键未出现**时默认为 ``True``（与编排默认一致）；
    - 显式 ``false``/``0``/``off`` 等则关闭。
    """
    if llm_extract is not None:
        return llm_extract
    raw = merged.get("file_flow_llm_extract")
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("0", "false", "no", "off"):
        return False
    return s in ("1", "true", "yes", "on")


def run_pdf_prepare(
    merged: dict[str, Any],
    *,
    workspace: Path,
    cwd: Path,
    pdf_dir: Path | None = None,
    schema: Path | None = None,
    out_dir: Path | None = None,
    llm_extract: bool | None = None,
    dry_run: bool = False,
    log_level: str | None = None,
    log_file: Path | None = None,
) -> int:
    """
    可编程入口：按 ``merged`` 与路径参数生成各 PDF 的 ``*_work.json``。
    ``llm_extract`` 为 ``None`` 时由 ``resolve_llm_extract_enabled`` 解析（未配置 pipeline 键时默认开启摘录）。
    """
    llm_extract = resolve_llm_extract_enabled(merged, llm_extract)

    if llm_extract:
        configure_logging(level=log_level, log_file=log_file)

    pdf_dir_raw = pdf_dir
    if pdf_dir_raw is None:
        mdir = merged.get("file_flow_pdf_dir")
        if not isinstance(mdir, str) or not mdir.strip():
            print("错误: 未指定 pdf_dir，或在 pipeline.json 中设置 file_flow_pdf_dir", file=sys.stderr)
            return 1
        pdf_dir_raw = Path(mdir.strip())
    schema_raw = schema
    if schema_raw is None:
        ms = merged.get("file_flow_schema_json")
        if isinstance(ms, str) and ms.strip():
            schema_raw = Path(ms.strip())
        else:
            schema_raw = workspace / "out" / "schema_example.json"
    out_raw = out_dir
    if out_raw is None:
        mo = merged.get("file_flow_out_dir")
        if isinstance(mo, str) and mo.strip():
            out_raw = Path(mo.strip())
        else:
            out_raw = workspace / "out"

    pdf_dir_p = _resolve(Path(pdf_dir_raw), cwd)
    schema_path = _resolve(Path(schema_raw), cwd)
    out_dir_p = _resolve(Path(out_raw), cwd)
    if not pdf_dir_p.is_dir():
        print(f"错误: 不是目录: {pdf_dir_p}", file=sys.stderr)
        return 1
    if not schema_path.is_file():
        print(f"错误: 找不到 schema: {schema_path}", file=sys.stderr)
        return 1
    out_dir_p.mkdir(parents=True, exist_ok=True)

    try:
        raw = schema_path.read_text(encoding="utf-8")
        schema_data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as e:
        print(f"错误: 无法解析 schema JSON: {e}", file=sys.stderr)
        return 1
    if not isinstance(schema_data, dict):
        print("错误: schema 根节点须为 JSON 对象", file=sys.stderr)
        return 1
    if not is_document_types_schema(schema_data):
        print(
            "错误: schema 须包含非空的 document_types 数组（结构见 out/schema_example.json）",
            file=sys.stderr,
        )
        return 1

    pdfs = sorted(pdf_dir_p.glob("*.pdf")) + sorted(pdf_dir_p.glob("*.PDF"))
    if not pdfs:
        print(f"提示: 目录内无 PDF 文件: {pdf_dir_p}", file=sys.stderr)
        return 0

    for pdf in pdfs:
        try:
            text, text_meta = extract_pdf_full_text_unified(
                pdf,
                merged,
                workspace=workspace,
                cwd=cwd,
                out_dir=out_dir_p,
            )
        except (RuntimeError, OSError) as e:
            print(f"错误: 全文抽取失败 {pdf.name}: {e}", file=sys.stderr)
            return 1
        # 全文单独落盘，工作 JSON 仅保留字段骨架；大模型抽取时再读入全文拼 prompt 写回各 content
        fulltext_basename = f"{pdf.stem}_fulltext.txt"
        fulltext_path = out_dir_p / fulltext_basename
        fulltext_path.write_text(text, encoding="utf-8")

        work = build_work_json_skeleton(schema_data)
        mode_note = "fulltext_file_only"
        if llm_extract:
            # 连接参数与「全文摘录」专用 system 一并构造，勿用 FILE_FLOW_SYSTEM_PROMPT 等评审向默认文案
            base_cfg = build_llm_env_config(merged, "")
            dry = dry_run
            if not dry and (
                not base_cfg.api_base
                or not base_cfg.model
                or not is_http_endpoint_url((base_cfg.api_base or "").strip())
            ):
                print(
                    "警告: 大模型抽取未配置有效 URL/模型，改为 dry-run 占位。",
                    file=sys.stderr,
                )
                dry = True
            try:
                work = enrich_work_json_with_llm_schema_extract(work, text, base_cfg, merged, dry_run=dry)
            except RuntimeError as e:
                print(f"错误: 大模型抽取失败: {e}", file=sys.stderr)
                return 1
            mode_note = "llm_schema_extract_dry_run" if dry else "llm_schema_extract"

        meta_out: dict[str, Any] = {
            "pdf_path": str(pdf.resolve()),
            "全文字符数": len(text),
            "schema_path": str(schema_path.resolve()),
            "内容填充模式": mode_note,
            "file_flow_pdf_fulltext_file": fulltext_basename,
        }
        if isinstance(text_meta, dict) and text_meta:
            meta_out.update(text_meta)
        work["_file_flow_meta"] = meta_out
        dest = out_dir_p / work_json_filename_for_stem(pdf.stem, merged)
        dest.write_text(json.dumps(work, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"已写出: {dest} （全文 {len(text)} 字符，全文文件={fulltext_path.name}，模式={mode_note}）",
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PDF 全文抽取 + 按 document_types schema 装配工作 JSON")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="管线 JSON；默认使用 file_flow 目录下的 pipeline.json，仅读取其中已声明的 file_flow_* 及 backend/hybrid 等键（不与仓库根配置、不与环境默认字典合并）",
    )
    ap.add_argument("--pdf-dir", type=Path, default=None, help="存放 PDF 的目录（可与 pipeline 中 file_flow_pdf_dir 二选一）")
    ap.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="schema JSON；未传时用 pipeline 的 file_flow_schema_json，否则 out/schema_example.json（相对 file_flow 目录）",
    )
    ap.add_argument("--out", type=Path, default=None, help="输出目录；未传时用 pipeline 的 file_flow_out_dir 或 out/")
    ap.add_argument(
        "--llm-extract",
        action="store_true",
        help="启用大模型按字段写入 content（否则各 field 的 content 为空；全文见同目录 *_fulltext.txt）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="与 --llm-extract 联用时不请求 API，content 写占位句",
    )
    ap.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="启用 --llm-extract 时的日志级别",
    )
    ap.add_argument("--log-file", type=Path, default=None, help="启用 --llm-extract 时追加日志文件（UTF-8）")
    ns = ap.parse_args(argv)

    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(_FILE_FLOW_DIR)
    if dotenv_missing:
        print(
            "警告: 未安装 python-dotenv，已跳过 .env；请 pip install python-dotenv",
            file=sys.stderr,
        )
    elif env_loaded:
        print(f"已加载环境文件: {len(env_loaded)} 个（file_flow 目录 .env / 环节变量.env 等）")

    if ns.llm_extract:
        configure_logging(level=ns.log_level, log_file=ns.log_file)

    disk = resolve_pipeline_disk_path(_FILE_FLOW_DIR, ns.config)
    merged = load_merged_pipeline_config(disk if disk is not None and disk.is_file() else None)

    return run_pdf_prepare(
        merged,
        workspace=_FILE_FLOW_DIR,
        cwd=Path.cwd(),
        pdf_dir=ns.pdf_dir,
        schema=ns.schema,
        out_dir=ns.out,
        llm_extract=(ns.llm_extract or None),
        dry_run=ns.dry_run,
        log_level=ns.log_level,
        log_file=ns.log_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
