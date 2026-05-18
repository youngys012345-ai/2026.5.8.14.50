#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_flow 全文抽取：与主流程 ``extract_pdf`` 对齐 **OpenDataLoader** 相关键（``backend`` 为
``opendataloader`` 或未识别时走 Java + 可选 Hybrid）。

**说明**：file_flow 当前**暂不支持** ``backend=mineru``；若配置仍为 mineru，将记录警告并
**改用 PyMuPDF** 本地抽字（与 ``file_flow_pdf_text_backend=pymupdf`` 效果类似，元数据会标明
``mineru_disabled_use_pymupdf``）。

- ``hybrid`` / ``hybrid_url`` / ``hybrid_mode`` / ``hybrid_timeout`` / ``hybrid_fallback`` 等与主流程一致；
  云端 Hybrid 须正确配置 ``hybrid_url``（未传入时底层可能访问默认端口如 8000）。
- ``skip_health_check``：file_flow 内**默认 true**（不请求 ``{hybrid_url}/health``）；若需启动前探测请设为 ``false``。
  ``hybrid_fallback=true`` 时 Hybrid 失败由 **Java 管线兜底**（``opendataloader_pdf`` 内部）。

手动开关（``pipeline.json`` 读入的 ``merged``）：

- ``file_flow_pdf_text_backend``：``pipeline``（默认，OpenDataLoader 管线）| ``pymupdf``（强制仅用本地 PyMuPDF）；
- ``file_flow_pdf_fallback_pymupdf``：``true``（默认）时，OpenDataLoader 失败再尝试 PyMuPDF。
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from .document_export import document_to_markdown
from .opendataloader_adapter import client_base_url_for_local_service
from .opendataloader_adapter import (
    OpenDataLoaderExtractionError,
    load_opendataloader_document,
    run_opendataloader_for_pdf,
)

_LOG = logging.getLogger(__name__)


def _hybrid_url_effective(merged: dict[str, Any]) -> str:
    """pipeline.json 的 hybrid_url；缺省时读环境（不再与 defaults_from_environment 合并时仍可用 .env）。"""
    v = merged.get("hybrid_url")
    if isinstance(v, str) and v.strip():
        return v.strip()
    for key in ("OPENDATALOADER_HYBRID_URL", "HYBRID_URL"):
        e = os.environ.get(key)
        if isinstance(e, str) and e.strip():
            return e.strip()
    return ""


def _truthy(merged: dict[str, Any], key: str, default: bool = True) -> bool:
    if key not in merged:
        return default
    v = merged[key]
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


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
    raw_hybrid = _hybrid_url_effective(merged)
    url_s = client_base_url_for_local_service(raw_hybrid)
    if url_s != raw_hybrid:
        _LOG.warning("hybrid_url 已从 %r 规范为 %r（0.0.0.0 不可作为 HTTP 客户端目标）", raw_hybrid, url_s)
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
    # file_flow 默认跳过 /health，避免无谓探测；若需探测请在 pipeline.json 设 skip_health_check: false
    skip_health = _truthy(merged, "skip_health_check", True)

    if hybrid and hybrid != "off" and not url_s:
        raise OpenDataLoaderExtractionError(
            "已启用 hybrid 但未解析到 hybrid_url（请在 file_flow/pipeline.json 填写 hybrid_url，"
            "或在 file_flow/.env 设置 OPENDATALOADER_HYBRID_URL）。"
            "未传 URL 时底层库可能访问默认端口（常见 8000），与本地 Hybrid 端口不一致。"
        )

    _LOG.info(
        "[全文抽取] hybrid=%s hybrid_url=%s skip_health_check=%s",
        hybrid_s,
        url_s or "(空)",
        skip_health,
    )

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
            "pdf_text_hybrid_url": url_s,
        }
        return text, meta
    finally:
        shutil.rmtree(per_dir, ignore_errors=True)


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

    ``workspace`` / ``cwd`` 保留签名供日后扩展；当前 OpenDataLoader 分支未使用。
    ``meta_dict`` 会并入 ``_file_flow_meta``。
    """
    _ = workspace, cwd  # 预留与路径解析扩展一致

    mode = str(merged.get("file_flow_pdf_text_backend") or "pipeline").strip().lower()
    fallback = _truthy(merged, "file_flow_pdf_fallback_pymupdf", True)

    if mode == "pymupdf":
        t = _extract_pymupdf(pdf)
        return t, {"pdf_text_backend": "pymupdf", "pdf_text_mode": "forced_local"}

    if mode not in ("pipeline", "auto", ""):
        _LOG.warning("[全文抽取] 未知 file_flow_pdf_text_backend=%r，按 pipeline 处理", mode)

    backend = _backend_from_merged(merged)
    if backend == "mineru":
        _LOG.warning(
            "[全文抽取] file_flow 暂不支持 MinerU（backend=mineru），已改用 PyMuPDF 抽取全文。"
        )
        t = _extract_pymupdf(pdf)
        return t, {
            "pdf_text_backend": "pymupdf",
            "pdf_text_mode": "mineru_disabled_use_pymupdf",
        }

    try:
        text, meta = _extract_opendataloader(pdf, merged, out_dir)
        meta["pdf_text_mode"] = "pipeline"
        return text, meta
    except (OpenDataLoaderExtractionError, OSError, RuntimeError) as exc:
        if not fallback:
            raise
        _LOG.warning("[全文抽取] OpenDataLoader 失败，回退 PyMuPDF: %s", exc)
        t = _extract_pymupdf(pdf)
        return t, {
            "pdf_text_backend": "pymupdf",
            "pdf_text_mode": "fallback_after_pipeline_error",
            "pdf_text_pipeline_backend": "opendataloader",
            "pdf_text_fallback_reason": str(exc)[:800],
        }
