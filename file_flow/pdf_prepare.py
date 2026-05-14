#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件模式第一步：从 PDF 目录读取 PDF，抽取全文（默认走 **OpenDataLoader**（Java + 可选 Hybrid、
``hybrid_fallback`` 兜底）；``backend=mineru`` 在 file_flow 中暂不支持，会自动改用 PyMuPDF）。
可用 ``file_flow_pdf_text_backend=pymupdf`` 强制仅用本地 PyMuPDF。
再按 **document_types** schema（与 ``file_flow/out/schema_example.json`` 一致）深拷贝生成工作 JSON。

schema **必须**包含非空 ``document_types`` 数组；默认 schema 路径为 ``file_flow/out/schema_example.json``
（可被 ``pipeline.json`` 的 ``file_flow_schema_json`` 或 ``--schema`` 覆盖）。

用法::

    python file_flow/pdf_prepare.py --pdf-dir ./pdfs --schema file_flow/out/schema_example.json --out file_flow/out
    python file_flow/pdf_prepare.py --config file_flow/pipeline.json
    python file_flow/pdf_prepare.py ... --llm-extract
    python file_flow/pdf_prepare.py ... --llm-extract --dry-run
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

from file_flow.pipeline_merge import (  # noqa: E402
    load_merged_pipeline_config,
    resolve_pipeline_disk_path,
)
from file_flow.step_dotenv import ensure_step_dotenv_loaded  # noqa: E402
from file_flow.llm_openai import (  # noqa: E402
    configure_logging,
    is_http_endpoint_url,
    load_llm_config_for_file_flow,
)
from file_flow.pdf_text_extract import extract_pdf_full_text_unified  # noqa: E402
from file_flow.schema_llm_extract import enrich_work_json_with_llm_schema_extract  # noqa: E402

ensure_step_dotenv_loaded(_ROOT)


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
    return (_ROOT / p).resolve()


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
    ``llm_extract`` 为 ``None`` 时读取 ``merged["file_flow_llm_extract"]``（布尔）。
    """
    if llm_extract is None:
        raw = merged.get("file_flow_llm_extract")
        llm_extract = bool(raw) if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes")

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
            schema_raw = workspace / "file_flow" / "out" / "schema_example.json"
    out_raw = out_dir
    if out_raw is None:
        mo = merged.get("file_flow_out_dir")
        if isinstance(mo, str) and mo.strip():
            out_raw = Path(mo.strip())
        else:
            out_raw = workspace / "file_flow" / "out"

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
            "错误: schema 须包含非空的 document_types 数组（结构见 file_flow/out/schema_example.json）",
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
        work = build_work_json_skeleton(schema_data)
        mode_note = "fulltext_placeholder"
        if llm_extract:
            base_cfg = load_llm_config_for_file_flow(merged)
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
        else:
            apply_full_text_to_all_contents(work, text)

        meta_out: dict[str, Any] = {
            "pdf_path": str(pdf.resolve()),
            "全文字符数": len(text),
            "schema_path": str(schema_path.resolve()),
            "内容填充模式": mode_note,
        }
        if isinstance(text_meta, dict) and text_meta:
            meta_out.update(text_meta)
        work["_file_flow_meta"] = meta_out
        dest = out_dir_p / f"{pdf.stem}_work.json"
        dest.write_text(json.dumps(work, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出: {dest} （全文 {len(text)} 字符，模式={mode_note}）")

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PDF 全文抽取 + 按 document_types schema 装配工作 JSON")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="管线 JSON；默认优先 file_flow/pipeline.json，与 pipeline_config 合并后读取 file_flow_* 及 backend/hybrid 等键",
    )
    ap.add_argument("--pdf-dir", type=Path, default=None, help="存放 PDF 的目录（可与 pipeline 中 file_flow_pdf_dir 二选一）")
    ap.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="schema JSON；未传时用 pipeline 的 file_flow_schema_json，否则 file_flow/out/schema_example.json",
    )
    ap.add_argument("--out", type=Path, default=None, help="输出目录；未传时用 pipeline 的 file_flow_out_dir 或 file_flow/out")
    ap.add_argument(
        "--llm-extract",
        action="store_true",
        help="启用大模型按字段写入 content（否则将整篇全文写入各 field 的 content 作占位）",
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

    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(_ROOT)
    if dotenv_missing:
        print(
            "警告: 未安装 python-dotenv，已跳过 .env；请 pip install python-dotenv",
            file=sys.stderr,
        )
    elif env_loaded:
        print(f"已加载环境文件: {len(env_loaded)} 个（仓库根 .env / 环节变量.env 等）")

    if ns.llm_extract:
        configure_logging(level=ns.log_level, log_file=ns.log_file)

    disk = resolve_pipeline_disk_path(_ROOT, ns.config)
    merged = load_merged_pipeline_config(disk if disk is not None and disk.is_file() else None)

    return run_pdf_prepare(
        merged,
        workspace=_ROOT,
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
