#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenDataLoader PDF（opendataloader_pdf）适配：调用 Java 管线，产出 JSON / Markdown 路径。"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .urlutil import client_base_url_for_local_service


class OpenDataLoaderExtractionError(RuntimeError):
    """OpenDataLoader 提取异常。"""


def locate_opendataloader_json(work_dir: Path, pdf_file: Path) -> Path:
    """OpenDataLoader 默认在输出目录根下生成 ``<stem>.json``。"""
    stem = pdf_file.stem
    direct = work_dir / f"{stem}.json"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(work_dir.glob(f"**/{stem}.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise OpenDataLoaderExtractionError(f"未在 {work_dir} 找到 OpenDataLoader JSON: {stem}.json")
    return candidates[0].resolve()


def _assert_hybrid_service_ready(hybrid_url: str, *, timeout_sec: float) -> None:
    """
    启用 hybrid 时，先探测 Docling Fast 服务是否可达（避免 Java CLI 打出长栈后才失败）。
    hybrid_url 示例：http://127.0.0.1:5002
    """
    base = hybrid_url.rstrip("/")
    health = f"{base}/health"
    try:
        with urlopen(health, timeout=timeout_sec) as resp:
            if getattr(resp, "status", 200) >= 400:
                raise OpenDataLoaderExtractionError(
                    f"Hybrid 服务响应异常: GET {health} -> HTTP {getattr(resp, 'status', '?')}"
                )
    except URLError as exc:
        raise OpenDataLoaderExtractionError(
            "未检测到 Docling Hybrid 服务（full/auto 模式依赖该服务）。\n"
            "请先在本机启动：pip install \"opendataloader-pdf[hybrid]\" 后执行\n"
            "  opendataloader-pdf-hybrid --port 5002\n"
            "若使用其他端口，请在配置中同步修改 hybrid_url。\n"
            f"当前探测: GET {health} 失败 — {exc}"
        ) from exc


def locate_opendataloader_markdown(work_dir: Path, pdf_file: Path) -> Path | None:
    stem = pdf_file.stem
    direct = work_dir / f"{stem}.md"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(work_dir.glob(f"**/{stem}.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].resolve() if candidates else None


def run_opendataloader_for_pdf(
    pdf_file: Path,
    work_dir: Path,
    *,
    table_method: str,
    reading_order: str,
    hybrid: str | None,
    hybrid_url: str | None,
    hybrid_mode: str,
    hybrid_timeout: str | int | float,
    hybrid_fallback: bool,
    quiet: bool,
    hybrid_health_timeout_sec: float,
    skip_hybrid_health_check: bool,
) -> tuple[Path, Path | None]:
    """
    调用 ``opendataloader_pdf.convert``，返回 (json_path, markdown_path_or_none)。

    Windows 上 Java CLI 对「命令行 / 工作目录」中含中文或非 ANSI 路径支持不稳，会导致解析失败。
    因此在传入 JVM 前将 PDF 复制到系统临时目录下的纯 ASCII 路径 ``…/input.pdf``，
    输出写入同一临时目录，成功后再整体镜像到 ``work_dir``（须由调用方保证为 ASCII 安全路径，见 extract_pdf）。
    """
    try:
        import opendataloader_pdf
    except ImportError as exc:
        raise OpenDataLoaderExtractionError(
            "未安装 opendataloader-pdf。请执行: pip install -U opendataloader-pdf，并安装 Java 11+。"
        ) from exc

    temp_root = Path(tempfile.mkdtemp(prefix="opdl_", suffix="_ascii"))
    safe_pdf = temp_root / "input.pdf"
    try:
        shutil.copy2(pdf_file, safe_pdf)
        kwargs: dict[str, Any] = {
            "input_path": str(safe_pdf.resolve()),
            "output_dir": str(temp_root.resolve()),
            "format": ["json", "markdown"],
            "quiet": quiet,
            "table_method": table_method,
            "reading_order": reading_order,
            "hybrid_fallback": hybrid_fallback,
        }
        if hybrid and hybrid != "off":
            kwargs["hybrid"] = hybrid
            eff_url = client_base_url_for_local_service(str(hybrid_url or "").strip())
            if eff_url:
                kwargs["hybrid_url"] = eff_url
                if not skip_hybrid_health_check:
                    _assert_hybrid_service_ready(eff_url, timeout_sec=hybrid_health_timeout_sec)
            else:
                raise OpenDataLoaderExtractionError(
                    "hybrid 已启用但未提供 hybrid_url，底层 opendataloader_pdf 可能使用内置默认端口（常见 8000）。"
                    "请在配置中传入 hybrid_url（例如 http://127.0.0.1:5002）。"
                )
            kwargs["hybrid_mode"] = hybrid_mode
            kwargs["hybrid_timeout"] = str(hybrid_timeout)

        try:
            opendataloader_pdf.convert(**kwargs)
        except Exception as exc:
            raise OpenDataLoaderExtractionError(f"OpenDataLoader 解析失败: {exc}") from exc

        json_src = locate_opendataloader_json(temp_root, safe_pdf)
        md_src = locate_opendataloader_markdown(temp_root, safe_pdf)

        work_dir.parent.mkdir(parents=True, exist_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        shutil.copytree(temp_root, work_dir)

        json_out = work_dir / json_src.name
        md_out = work_dir / md_src.name if md_src is not None else None
        return json_out, md_out
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def load_opendataloader_document(json_path: Path, source_pdf: Path) -> dict[str, Any]:
    """读取 OpenDataLoader 导出的 JSON（已是 kids 结构），补充 extraction_meta。"""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenDataLoaderExtractionError(f"读取 JSON 失败: {json_path}，原因: {exc}") from exc

    if not isinstance(data, dict):
        raise OpenDataLoaderExtractionError("OpenDataLoader JSON 根节点必须是对象")

    meta = data.get("extraction_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["backend"] = "opendataloader"
    meta["opendataloader_json"] = str(json_path.resolve())
    data["extraction_meta"] = meta
    if "file name" not in data:
        data["file name"] = source_pdf.name
    return data
