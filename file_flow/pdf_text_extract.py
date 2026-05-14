#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_flow 全文抽取：与仓库根 ``extract_pdf.py`` 使用同一套 ``pipeline.json`` 键，

- ``backend`` = ``mineru`` | ``opendataloader``：分别走 MinerU CLI 或 ``opendataloader_pdf.convert``（Java）；
- ``opendataloader`` 时 ``hybrid`` / ``hybrid_url`` / ``hybrid_mode`` / ``hybrid_timeout`` /
  ``hybrid_fallback`` / ``skip_health_check`` 等与主流程一致（云端 Hybrid 配 ``hybrid_url``，
  ``hybrid_fallback=true`` 时 Hybrid 失败由 **Java 管线兜底**，由 ``opendataloader_pdf`` 内部处理）。

手动开关（合并后的 ``merged``）：

- ``file_flow_pdf_text_backend``：``pipeline``（默认，跟随 ``backend``）| ``pymupdf``（强制仅用本地 PyMuPDF）；
- ``file_flow_pdf_fallback_pymupdf``：``true``（默认）时，管线抽取失败再尝试 PyMuPDF。
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from document_export import document_to_markdown  # noqa: E402
from mineru_adapter import (  # noqa: E402
    MinerUExtractionError,
    convert_mineru_content_list_to_document,
    run_mineru_cli_for_pdf,
)
from opendataloader_adapter import (  # noqa: E402
    OpenDataLoaderExtractionError,
    load_opendataloader_document,
    run_opendataloader_for_pdf,
)

_LOG = logging.getLogger(__name__)


def _truthy(merged: dict[str, Any], key: str, default: bool = True) -> bool:
    if key not in merged:
        return default
    v = merged[key]
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _resolve_path_str(raw: str | Path | None, workspace: Path, cwd: Path) -> Path | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    hit = (cwd / p).resolve()
    if hit.exists():
        return hit
    return (workspace / p).resolve()


def _extract_pymupdf(pdf_path: Path) -> str:
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


def _backend_from_merged(merged: dict[str, Any]) -> str:
    b = merged.get("backend")
    s = str(b).strip().lower() if b is not None else ""
    return s if s in ("mineru", "opendataloader") else "opendataloader"


def _extract_opendataloader(
    pdf: Path,
    merged: dict[str, Any],
    out_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """OpenDataLoader + 可选 Hybrid；Java 兜底由 ``hybrid_fallback`` 控制。"""
    table_method = str(merged.get("table_method") or "cluster").strip()
    reading_order = str(merged.get("reading_order") or "xycut").strip()
    hybrid_raw = merged.get("hybrid")
    hybrid_s = str(hybrid_raw).strip() if hybrid_raw is not None else "off"
    hybrid = None if hybrid_s == "off" else hybrid_s
    hybrid_url = merged.get("hybrid_url")
    url_s = str(hybrid_url).strip() if hybrid_url is not None else ""
    hybrid_mode = str(merged.get("hybrid_mode") or "auto").strip()
    hybrid_timeout = merged.get("hybrid_timeout")
    if hybrid_timeout is None:
        hybrid_timeout = "0"
    hybrid_fallback = _truthy(merged, "hybrid_fallback", True)
    quiet = _truthy(merged, "quiet", False)
    try:
        health_t = float(merged.get("hybrid_health_timeout_sec") or 15.0)
    except (TypeError, ValueError):
        health_t = 15.0
    skip_health = _truthy(merged, "skip_health_check", False)

    _odl_key = hashlib.sha256(str(pdf.resolve()).encode("utf-8")).hexdigest()[:24]
    per_dir = (out_dir / "_opendataloader_work" / f"odl_{_odl_key}").resolve()
    if per_dir.exists():
        shutil.rmtree(per_dir, ignore_errors=True)
    per_dir.mkdir(parents=True, exist_ok=False)
    try:
        json_path_odl, _md_native = run_opendataloader_for_pdf(
            pdf,
            per_dir,
            table_method=table_method,
            reading_order=reading_order,
            hybrid=hybrid,
            hybrid_url=url_s if hybrid else None,
            hybrid_mode=hybrid_mode,
            hybrid_timeout=hybrid_timeout,
            hybrid_fallback=hybrid_fallback,
            quiet=quiet,
            hybrid_health_timeout_sec=health_t,
            skip_hybrid_health_check=skip_health,
        )
        document = load_opendataloader_document(json_path_odl, pdf)
        text = document_to_markdown(document).strip()
        meta: dict[str, Any] = {
            "pdf_text_backend": "opendataloader",
            "pdf_text_hybrid": hybrid_s,
            "pdf_text_hybrid_fallback": hybrid_fallback,
        }
        return text, meta
    finally:
        shutil.rmtree(per_dir, ignore_errors=True)


def _extract_mineru(
    pdf: Path,
    merged: dict[str, Any],
    workspace: Path,
    cwd: Path,
    out_dir: Path,
) -> tuple[str, dict[str, Any]]:
    mpr = _resolve_path_str(merged.get("mineru_project_root"), workspace, cwd)
    if mpr is None or not mpr.is_dir() or not (mpr / "mineru").is_dir():
        raise MinerUExtractionError(f"mineru_project_root 无效或缺少 mineru/ 子目录: {mpr}")
    raw_sub = (out_dir / "_mineru_raw" / pdf.stem).resolve()
    if raw_sub.exists():
        shutil.rmtree(raw_sub, ignore_errors=True)
    raw_sub.mkdir(parents=True, exist_ok=True)
    content_list_path = run_mineru_cli_for_pdf(
        pdf_file=pdf,
        output_root=raw_sub,
        mineru_project_root=mpr,
        backend=str(merged.get("mineru_backend") or "pipeline").strip() or None,
        api_url=str(merged.get("mineru_api_url") or "").strip() or None,
        model_source=str(merged.get("mineru_model_source") or "").strip() or None,
        mineru_tools_config_json=str(merged.get("mineru_tools_config_json") or "").strip() or None,
        cli_timeout_sec=merged.get("mineru_cli_timeout_sec"),
    )
    document = convert_mineru_content_list_to_document(
        content_list_path=content_list_path,
        source_pdf=pdf,
    )
    text = document_to_markdown(document).strip()
    meta = {
        "pdf_text_backend": "mineru",
        "pdf_text_mineru_root": str(mpr),
    }
    return text, meta


def extract_pdf_full_text_unified(
    pdf: Path,
    merged: dict[str, Any],
    *,
    workspace: Path,
    cwd: Path,
    out_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """
    按 ``merged`` 与 ``file_flow_pdf_text_backend`` 抽取全文，返回 ``(text, meta_dict)``。

    ``meta_dict`` 会并入 ``_file_flow_meta``（如 ``pdf_text_backend``、回退原因等）。
    """
    mode = str(merged.get("file_flow_pdf_text_backend") or "pipeline").strip().lower()
    fallback = _truthy(merged, "file_flow_pdf_fallback_pymupdf", True)

    if mode == "pymupdf":
        t = _extract_pymupdf(pdf)
        return t, {"pdf_text_backend": "pymupdf", "pdf_text_mode": "forced_local"}

    if mode not in ("pipeline", "auto", ""):
        _LOG.warning("[全文抽取] 未知 file_flow_pdf_text_backend=%r，按 pipeline 处理", mode)

    backend = _backend_from_merged(merged)
    try:
        if backend == "mineru":
            text, meta = _extract_mineru(pdf, merged, workspace, cwd, out_dir)
            meta["pdf_text_mode"] = "pipeline"
            return text, meta
        text, meta = _extract_opendataloader(pdf, merged, out_dir)
        meta["pdf_text_mode"] = "pipeline"
        return text, meta
    except (MinerUExtractionError, OpenDataLoaderExtractionError, OSError, RuntimeError) as exc:
        if not fallback:
            raise
        _LOG.warning("[全文抽取] 管线后端失败，回退 PyMuPDF: %s", exc)
        t = _extract_pymupdf(pdf)
        return t, {
            "pdf_text_backend": "pymupdf",
            "pdf_text_mode": "fallback_after_pipeline_error",
            "pdf_text_pipeline_backend": backend,
            "pdf_text_fallback_reason": str(exc)[:800],
        }
