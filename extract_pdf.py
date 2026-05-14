#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 批量解析：

- **mineru**：在项目配置的本地 MinerU 源码目录下执行 ``python -m mineru.cli.client``，
  将 content_list 转为统一 kids 结构。
- **opendataloader**：调用 ``opendataloader_pdf.convert``（需 Java 11+）。

输出目录、后端参数等默认在项目根 ``pipeline.json``（或 ``--config``）；同名键可被环境变量（如 ``OPENDATALOADER_HYBRID_URL``）覆盖，再被命令行覆盖。

模块导入时会通过 ``step_dotenv`` 加载项目根与当前目录下的 ``.env`` / ``环节变量.env``。
若未设置 ``OPENDATALOADER_VLM_*``，VLM 可选用与评审环节相同的 ``LLM_API_BASE``（完整 POST URL）、
``LLM_MODEL``、``LLM_API_KEY``；若 ``pipeline.json`` 中 ``vlm_api_base`` 为服务根且非 ``http(s)`` 全链，
可另配 ``vlm_chat_path``（见 ``vlm_client.join_openai_compatible_endpoint_url``）。

统一 JSON 会导出 ``output/markdown/<stem>.md``；``markdown_by_page`` 为 true 时另存 ``<stem>_by_page.md``（按页分块、标页码与推断标题，便于大模型按页处理）。

可选：
- ``--visual-tagging clip|vlm``：对裁剪图做签名/指印/印章识别并回填。
- ``--vlm-fallback auto|force``：当页面文本极少或强制开启时，用 VLM 按页转写并合并（需 PyMuPDF + VLM 配置）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 保证与同目录模块（mineru_adapter 等）的导入在任何启动方式下都可用，
# 例如 python -P、部分 IDE、「以模块方式调试」等非传统 sys.path。
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from step_dotenv import ensure_step_dotenv_loaded  # noqa: E402

ensure_step_dotenv_loaded(_repo_root)

import argparse
import hashlib
import json
import os
import shutil
import tempfile

from document_export import document_to_markdown, document_to_markdown_by_page
from mineru_adapter import (
    MinerUExtractionError,
    convert_mineru_content_list_to_document,
    run_mineru_cli_for_pdf,
)
from opendataloader_adapter import (
    OpenDataLoaderExtractionError,
    load_opendataloader_document,
    run_opendataloader_for_pdf,
)
from pipeline_config import (
    defaults_from_environment,
    load_config_file,
    merge_defaults,
    pop_config_path_from_argv,
    resolve_pipeline_config_path,
)
from visual_tagging import enrich_document_with_visual_tags
from vlm_client import (
    build_openai_compatible_vlm_detector,
    build_openai_compatible_vlm_page_transcriber,
    is_http_endpoint_url,
)
from vlm_page_fallback import (
    document_needs_vlm_fallback,
    merge_vlm_page_transcripts,
    render_pdf_pages_to_png,
)


def _mineru_pipeline_models_parent(mineru_repo: Path) -> Path | None:
    """
    解析 pipeline 模型根目录（其下须有 ``models/``，与 PDF-Extract-Kit / mineru-models-download 布局一致）。
    依次尝试：仓库根 ``MinerU``、``MinerU/pipeline``。
    """
    root = mineru_repo.resolve()
    for parent in (root, root / "pipeline"):
        if parent.is_dir() and (parent / "models").is_dir():
            return parent
    return None


def _resolve_mineru_repo_for_models(workspace: Path, args: argparse.Namespace) -> Path | None:
    """根据 mineru_project_root 或常见目录定位 MinerU 源码根（须含 mineru/）。"""
    mpr = getattr(args, "mineru_project_root", None)
    if isinstance(mpr, str) and mpr.strip():
        p = Path(mpr.strip()).resolve()
        if (p / "mineru").is_dir():
            return p
    for cand in (workspace / "MinerU", workspace.parent / "MinerU"):
        if (cand / "mineru").is_dir():
            return cand.resolve()
    return None


def _auto_fill_local_mineru_settings(args: argparse.Namespace, workspace: Path) -> None:
    """
    仅在 backend=mineru 时生效。
    优先顺序：
    1. ``config/mineru.local.json``（若存在）；
    2. MinerU 源码目录内自带的 pipeline 模型（``MinerU/models/...`` 或 ``MinerU/pipeline/models/...``）；
    3. 兼容旧路径：项目根 ``local_models/mineru/pipeline``。
    命中本地目录时生成临时 tools JSON，并设置 ``mineru_model_source=local``。
    """
    if getattr(args, "backend", None) != "mineru":
        return

    local_cfg = workspace / "config" / "mineru.local.json"
    legacy_pipeline_root = (workspace / "local_models" / "mineru" / "pipeline").resolve()

    if args.mineru_tools_config_json is None:
        if local_cfg.is_file():
            args.mineru_tools_config_json = str(local_cfg.resolve())
        else:
            pipeline_root: Path | None = None
            mineru_repo = _resolve_mineru_repo_for_models(workspace, args)
            if mineru_repo is not None:
                pipeline_root = _mineru_pipeline_models_parent(mineru_repo)
            if pipeline_root is None and legacy_pipeline_root.is_dir() and (legacy_pipeline_root / "models").is_dir():
                pipeline_root = legacy_pipeline_root

            if pipeline_root is not None and args.mineru_model_source in (None, "local"):
                payload = {"models-dir": {"pipeline": str(pipeline_root)}}
                fd, tmp_path = tempfile.mkstemp(prefix="mineru_tools_", suffix=".json")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh, ensure_ascii=False, indent=2)
                except OSError:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    raise
                args.mineru_tools_config_json = tmp_path
                setattr(args, "_mineru_autogen_tools_json", tmp_path)

    if args.mineru_model_source is None and args.mineru_tools_config_json:
        args.mineru_model_source = "local"


def _resolve_path_for_input(raw: str, workspace: Path) -> Path:
    """
    将配置或命令行中的 PDF/目录路径解析为绝对路径。
    若为相对路径：优先当前工作目录，其次项目根（extract_pdf 所在目录）。
    """
    p = Path(raw.strip())
    if p.is_absolute():
        return p.resolve()
    cwd_hit = (Path.cwd() / p).resolve()
    ws_hit = (workspace / p).resolve()
    if cwd_hit.is_file() or cwd_hit.is_dir():
        return cwd_hit
    if ws_hit.is_file() or ws_hit.is_dir():
        return ws_hit
    return cwd_hit


def _resolve_path_for_workspace_relative(raw: str | None, workspace: Path) -> str | None:
    """
    输出类相对路径锚定到项目根，避免因在其他目录启动脚本而写到别处。
    绝对路径保持不变。
    """
    if raw is None or not str(raw).strip():
        return None
    p = Path(str(raw).strip())
    if p.is_absolute():
        return str(p.resolve())
    return str((workspace / p).resolve())


def _normalize_path_args_after_parse(args: argparse.Namespace, workspace: Path) -> None:
    """根据项目根规整路径类参数。"""
    if getattr(args, "input", None):
        resolved = _resolve_path_for_input(args.input, workspace)
        args.input = str(resolved)
    mtcj = getattr(args, "mineru_tools_config_json", None)
    if isinstance(mtcj, str) and mtcj.strip():
        p = Path(mtcj.strip())
        if not p.is_absolute():
            cwd_p = (Path.cwd() / p).resolve()
            ws_p = (workspace / p).resolve()
            args.mineru_tools_config_json = str(cwd_p if cwd_p.is_file() else ws_p if ws_p.is_file() else cwd_p)
        else:
            args.mineru_tools_config_json = str(p.resolve())

    od = getattr(args, "output_dir", None)
    args.output_dir = _resolve_path_for_workspace_relative(od, workspace) if od else None

    jd = getattr(args, "json_output_dir", None)
    args.json_output_dir = _resolve_path_for_workspace_relative(jd, workspace) if jd else None

    mdod = getattr(args, "markdown_output_dir", None)
    args.markdown_output_dir = _resolve_path_for_workspace_relative(mdod, workspace) if mdod else None

    mpr = getattr(args, "mineru_project_root", None)
    if isinstance(mpr, str) and mpr.strip():
        p = Path(mpr.strip())
        if not p.is_absolute():
            cwd_p = (Path.cwd() / p).resolve()
            ws_p = (workspace / p).resolve()
            args.mineru_project_root = str(
                cwd_p if (cwd_p / "mineru").is_dir() else ws_p if (ws_p / "mineru").is_dir() else cwd_p
            )
        else:
            args.mineru_project_root = str(p.resolve())


def _collect_pdf_files(input_path: Path, recursive: bool = False) -> list[Path]:
    """收集待处理 PDF（单文件或目录）。"""
    if input_path.is_file():
        return [input_path.resolve()]
    if input_path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        return sorted(p.resolve() for p in input_path.glob(pattern) if p.is_file())
    return []


def _resolve_output_layout(
    workspace: Path,
    input_path: Path,
    output_dir: str | None,
    json_output_dir: str | None,
    markdown_output_dir: str | None,
) -> tuple[Path, Path, Path]:
    """
    解析输出目录布局。
    ``output_dir`` 须在 pipeline.json 中配置（或由命令行覆盖）；若为 None 则抛出 ValueError。
    返回 (json_dir, markdown_dir, output_root)。
    """
    if not output_dir or not str(output_dir).strip():
        raise ValueError("output_dir 未配置：请在 pipeline.json 中设置 output_dir")
    output_root = Path(output_dir).resolve()
    json_dir = Path(json_output_dir).resolve() if json_output_dir else (output_root / "json").resolve()
    md_dir = (
        Path(markdown_output_dir).resolve()
        if markdown_output_dir
        else (output_root / "markdown").resolve()
    )
    return json_dir, md_dir, output_root


def _validate_pipeline_args(args: argparse.Namespace) -> list[str]:
    """校验 pipeline.json（及环境变量、命令行）是否提供完整配置；返回人类可读错误列表。"""
    errs: list[str] = []

    def _need(name: str) -> bool:
        v = getattr(args, name, None)
        return v is None or (isinstance(v, str) and not v.strip())

    if _need("output_dir"):
        errs.append("缺少 output_dir：请在 pipeline.json 中配置结果输出根目录（可用环境变量 OPENDATALOADER_OUTPUT_DIR 或命令行 -o 覆盖）。")
    if _need("backend"):
        errs.append("缺少 backend：须为 mineru 或 opendataloader。")
    else:
        b = getattr(args, "backend", None)
        if b not in ("mineru", "opendataloader"):
            errs.append(f"backend 无效: {b!r}，须为 mineru 或 opendataloader。")

    if getattr(args, "recursive", None) is None:
        errs.append("缺少 recursive：请在 pipeline.json 中配置布尔值（是否递归扫描子目录中的 PDF）。")
    if getattr(args, "quiet", None) is None:
        errs.append("缺少 quiet：请在 pipeline.json 中配置布尔值（静默模式）。")
    if getattr(args, "markdown_by_page", None) is None:
        errs.append(
            "缺少 markdown_by_page：请在 pipeline.json 中配置布尔值；"
            "为 true 时额外写出按页聚合的 *_by_page.md（便于大模型按页处理）。"
        )

    if _need("input"):
        errs.append("缺少 input：请在 pipeline.json 中配置 PDF 文件或目录路径，或通过命令行参数传入。")

    if getattr(args, "visual_tagging", None) is None:
        errs.append("缺少 visual_tagging：off | clip | vlm。")
    if getattr(args, "visual_min_score", None) is None:
        errs.append("缺少 visual_min_score：视觉标签置信度下限（0~1）。")

    if getattr(args, "vlm_fallback", None) is None:
        errs.append("缺少 vlm_fallback：off | auto | force。")
    if getattr(args, "vlm_fallback_threshold", None) is None:
        errs.append("缺少 vlm_fallback_threshold：auto 模式下触发页级 VLM 的文本长度阈值。")
    if getattr(args, "vlm_fallback_dpi", None) is None:
        errs.append("缺少 vlm_fallback_dpi：页级 VLM 渲染 PDF 的 DPI。")

    vt = getattr(args, "visual_tagging", None)
    vf = getattr(args, "vlm_fallback", None)
    vlm_needed = vt == "vlm" or vf != "off"
    if vlm_needed:
        if _need("vlm_api_base") or _need("vlm_model"):
            errs.append("启用 visual_tagging=vlm 或 vlm_fallback 时，须在 pipeline.json 中配置 vlm_api_base 与 vlm_model。")
        if getattr(args, "vlm_timeout_sec", None) is None:
            errs.append("启用 VLM 时须配置 vlm_timeout_sec（秒）。")
        vlm_base = getattr(args, "vlm_api_base", None)
        vlm_path = getattr(args, "vlm_chat_path", None)
        base_s = str(vlm_base).strip() if isinstance(vlm_base, str) and vlm_base.strip() else ""
        path_s = str(vlm_path).strip() if isinstance(vlm_path, str) and vlm_path.strip() else ""
        if not base_s or (not is_http_endpoint_url(base_s) and not path_s):
            errs.append(
                "启用 VLM 时：请将 vlm_api_base 设为完整 https:// 的 Chat Completions POST URL；"
                "或同时配置 vlm_api_base（服务根）与 vlm_chat_path（如 /v1/chat/completions）。"
            )
        if vf != "off" and getattr(args, "vlm_page_transcribe_min_timeout_sec", None) is None:
            errs.append("启用 vlm_fallback 时须配置 vlm_page_transcribe_min_timeout_sec（页级转写请求下限超时，秒）。")

    b = getattr(args, "backend", None)
    if b == "mineru":
        if _need("mineru_project_root"):
            errs.append("backend=mineru 时须在 pipeline.json 中配置 mineru_project_root（MinerU 仓库根目录，须含 mineru/）。")
        if _need("mineru_backend"):
            errs.append("backend=mineru 时须配置 mineru_backend（传给 MinerU CLI 的 -b，如 pipeline）。")
    elif b == "opendataloader":
        if _need("table_method"):
            errs.append("backend=opendataloader 时须配置 table_method（OpenDataLoader：default | cluster）。")
        if _need("reading_order"):
            errs.append("backend=opendataloader 时须配置 reading_order（OpenDataLoader：xycut | off）。")
        if getattr(args, "hybrid", None) is None:
            errs.append("backend=opendataloader 时须配置 hybrid：off | docling-fast | hancom-ai。")
        else:
            h = getattr(args, "hybrid", "")
            if h not in ("off", "docling-fast", "hancom-ai"):
                errs.append(f"hybrid 无效: {h!r}。")
            elif h != "off":
                if _need("hybrid_url"):
                    errs.append("启用 hybrid 时须配置 hybrid_url（Docling hybrid 服务根 URL）。")
        if getattr(args, "hybrid_mode", None) is None:
            errs.append("backend=opendataloader 时须配置 hybrid_mode（hybrid 关闭时仍会传给底层占位，建议填 auto）。")
        if getattr(args, "hybrid_timeout", None) is None:
            errs.append("backend=opendataloader 时须配置 hybrid_timeout（毫秒字符串；hybrid 关闭时可填 \"0\"）。")
        if getattr(args, "hybrid_health_timeout_sec", None) is None:
            errs.append("backend=opendataloader 时须配置 hybrid_health_timeout_sec（探测 hybrid /health 的超时秒数）。")
        if getattr(args, "hybrid_fallback", None) is None:
            errs.append("backend=opendataloader 时须配置 hybrid_fallback（hybrid 失败时是否回落 Java 管线）。")
        if getattr(args, "skip_health_check", None) is None:
            errs.append(
                "backend=opendataloader 时须配置 skip_health_check（是否跳过 hybrid 的 /health 探测；"
                "仅用 Java 管线时可填 true）。"
            )
        if getattr(args, "hybrid_force_ocr", None) is None:
            errs.append(
                "backend=opendataloader 时须配置 hybrid_force_ocr（布尔值）。"
                "图片/扫描类 PDF 在 hybrid_mode=full 时建议 true，并须用 scripts/start_docling_hybrid.ps1 "
                "启动 Docling Fast（会加 --force-ocr）；该标志由服务进程读取，opendataloader_pdf.convert 无法在线修改。"
            )

    return errs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDF 批量抽取（MinerU 源码 / OpenDataLoader），输出 JSON 与 Markdown",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON 配置文件（与环境变量、CLI 合并）；默认加载项目根 pipeline.json；示例见 config/pipeline.example.json",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="输入 PDF 或目录（通常写在 pipeline.json；命令行传入时优先生效）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="输出根目录（覆盖 pipeline.json 中的 output_dir）",
    )
    parser.add_argument(
        "--json-output-dir",
        default=None,
        help="JSON 输出目录（覆盖 <output-dir>/json）",
    )
    parser.add_argument(
        "--markdown-output-dir",
        default=None,
        help="Markdown 输出目录（覆盖 <output-dir>/markdown）",
    )
    parser.add_argument(
        "--backend",
        choices=("mineru", "opendataloader"),
        default=None,
        help="抽取后端：mineru（本地 MinerU）或 opendataloader（Java / opendataloader-pdf）",
    )
    parser.add_argument(
        "--mineru-project-root",
        default=None,
        help="MinerU 源码仓库根路径（须含 mineru/）",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="目录输入时是否递归子目录中的 PDF（可与 --no-recursive 显式关闭）",
    )
    parser.add_argument(
        "--mineru-backend",
        default=None,
        help="MinerU CLI 的 -b backend（如 pipeline）",
    )
    parser.add_argument(
        "--mineru-api-url",
        default=None,
        help="常驻 mineru-api 根 URL（可选）；不传则每次临时启动本地 API",
    )
    parser.add_argument(
        "--mineru-model-source",
        choices=("local", "huggingface", "modelscope"),
        default=None,
        help="MinerU 模型来源",
    )
    parser.add_argument(
        "--mineru-tools-config-json",
        default=None,
        help="MinerU 工具 JSON（MINERU_TOOLS_CONFIG_JSON），含 models-dir 等",
    )
    parser.add_argument(
        "--mineru-cli-timeout-sec",
        type=float,
        default=None,
        help="MinerU 子进程最长运行时间（秒）；不设则无上限",
    )
    parser.add_argument(
        "--hybrid",
        choices=("off", "docling-fast", "hancom-ai"),
        default=None,
        help="OpenDataLoader hybrid（仅 backend=opendataloader）",
    )
    parser.add_argument(
        "--hybrid-url",
        default=None,
        help="hybrid 服务根 URL",
    )
    parser.add_argument(
        "--hybrid-mode",
        choices=("auto", "full"),
        default=None,
        help="OpenDataLoader hybrid_mode",
    )
    parser.add_argument(
        "--hybrid-timeout",
        default=None,
        help="hybrid_timeout（毫秒字符串，0 表示不限制）",
    )
    parser.add_argument(
        "--hybrid-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="hybrid 失败时是否回落 Java 原生管线",
    )
    parser.add_argument(
        "--hybrid-health-timeout-sec",
        type=float,
        default=None,
        help="调用 hybrid /health 探测的超时（秒）",
    )
    parser.add_argument(
        "--skip-health-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否跳过 hybrid 启动前的 /health 探测（离线或已知服务不稳定时可设 true）",
    )
    parser.add_argument(
        "--hybrid-force-ocr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否要求 Docling Fast 以全页强制 OCR 运行（须启动 hybrid 时加 --force-ocr，见 scripts/start_docling_hybrid.ps1）",
    )
    parser.add_argument(
        "--hybrid-ocr-lang",
        default=None,
        help="Docling OCR 语言（传给 hybrid 启动脚本），如 ch_sim,en",
    )
    parser.add_argument(
        "--table-method",
        choices=("default", "cluster"),
        default=None,
        help="OpenDataLoader table_method",
    )
    parser.add_argument(
        "--reading-order",
        choices=("xycut", "off"),
        default=None,
        help="OpenDataLoader reading_order",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="静默模式",
    )
    parser.add_argument(
        "--markdown-by-page",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否额外输出按页聚合的 Markdown（*_by_page.md）",
    )
    parser.add_argument(
        "--visual-tagging",
        choices=("off", "clip", "vlm"),
        default=None,
        help="视觉标签：off；clip；vlm（需完整 VLM 配置）",
    )
    parser.add_argument(
        "--visual-min-score",
        type=float,
        default=None,
        help="视觉标签置信度下限",
    )
    parser.add_argument(
        "--vlm-api-base",
        default=None,
        help="VLM OpenAI 兼容根 URL",
    )
    parser.add_argument(
        "--vlm-api-key",
        default=None,
        help="VLM API Key（或环境变量 OPENDATALOADER_VLM_API_KEY）",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="VLM 模型名",
    )
    parser.add_argument(
        "--vlm-timeout-sec",
        type=float,
        default=None,
        help="VLM 请求超时（秒）",
    )
    parser.add_argument(
        "--vlm-chat-path",
        default=None,
        help="chat completions 路径（如 /v1/chat/completions）",
    )
    parser.add_argument(
        "--vlm-system-prompt",
        default=None,
        help="图像分类场景的 system prompt（visual-tagging=vlm）",
    )
    parser.add_argument(
        "--vlm-user-prompt",
        default=None,
        help="图像分类场景的 user prompt",
    )
    parser.add_argument(
        "--vlm-fallback",
        choices=("off", "auto", "force"),
        default=None,
        help="页级 VLM 回退：off；auto；force",
    )
    parser.add_argument(
        "--vlm-fallback-threshold",
        type=int,
        default=None,
        help="auto 模式下触发页级 VLM 的结构化文本长度阈值",
    )
    parser.add_argument(
        "--vlm-fallback-dpi",
        type=int,
        default=None,
        help="页级 VLM 渲染 PDF 的 DPI",
    )
    parser.add_argument(
        "--vlm-page-transcribe-min-timeout-sec",
        type=float,
        default=None,
        help="页级转写 HTTP 超时取下限（秒），实际超时为 max(vlm_timeout_sec, 本值)",
    )
    parser.add_argument(
        "--vlm-page-system-prompt",
        default=None,
        help="页级转写 system prompt",
    )
    parser.add_argument(
        "--vlm-page-user-prompt-template",
        default=None,
        help="页级转写 user 模板，可使用 {page_number}",
    )
    return parser


def _require_vlm_config(args: argparse.Namespace) -> tuple[str, str, float, str | None] | None:
    """若未配置完整 VLM 参数则返回 None。``vlm_api_base`` 为完整 ``http(s)`` URL 时 ``vlm_chat_path`` 可省略。"""
    api_base = getattr(args, "vlm_api_base", None)
    model = getattr(args, "vlm_model", None)
    timeout_sec = getattr(args, "vlm_timeout_sec", None)
    chat_path = getattr(args, "vlm_chat_path", None)
    if not api_base or not model or timeout_sec is None:
        return None
    base_s = str(api_base).strip()
    path_s = str(chat_path).strip() if isinstance(chat_path, str) and chat_path.strip() else None
    if is_http_endpoint_url(base_s):
        # 完整 endpoint 时不再与 vlm_chat_path 拼接，避免误配两段导致 URL 错误
        return base_s, str(model).strip(), float(timeout_sec), None
    if not path_s:
        return None
    return base_s, str(model).strip(), float(timeout_sec), path_s


def main() -> int:
    if sys.version_info < (3, 9):
        print(
            f"错误: 需要 Python 3.9 及以上（当前 {sys.version.split()[0]}）。",
            file=sys.stderr,
        )
        return 1
    config_from_argv, argv_rest = pop_config_path_from_argv(sys.argv[1:])
    parser = _build_parser()
    workspace = Path(__file__).resolve().parent
    cfg_path = config_from_argv
    if cfg_path is None:
        default_pipeline = workspace / "pipeline.json"
        resolved_auto, hint_auto = resolve_pipeline_config_path(default_pipeline)
        cfg_path = resolved_auto
        if hint_auto:
            print(hint_auto, file=sys.stderr)
    if cfg_path is None:
        example = workspace / "config" / "pipeline.example.json"
        print(
            "错误: 未找到 pipeline.json。请将 config/pipeline.example.json 复制为项目根 pipeline.json 并按需修改。",
            file=sys.stderr,
        )
        print(f"模板路径: {example}", file=sys.stderr)
        return 1
    resolved_cfg, cfg_hint = resolve_pipeline_config_path(cfg_path)
    if resolved_cfg is None:
        print(f"配置文件不存在: {cfg_path}", file=sys.stderr)
        return 1
    if cfg_hint:
        print(cfg_hint, file=sys.stderr)
    cfg_path = resolved_cfg
    try:
        # 合并顺序：pipeline.json 为底，环境变量覆盖同名键（便于 .env 指向云端 hybrid），命令行再覆盖。
        merged = merge_defaults(load_config_file(cfg_path), defaults_from_environment())
    except json.JSONDecodeError as exc:
        print(
            f"pipeline.json 解析失败（须为标准 JSON：不要用 // 注释、勿尾随逗号）：{cfg_path}\n{exc}",
            file=sys.stderr,
        )
        return 1
    except UnicodeDecodeError as exc:
        print(
            f"pipeline.json 须以 UTF-8 编码保存（建议无 BOM）。读取失败: {cfg_path}\n{exc}",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as exc:
        print(f"读取配置文件失败: {cfg_path}\n{exc}", file=sys.stderr)
        return 1
    dests = {
        a.dest
        for a in parser._actions
        if getattr(a, "dest", None) and a.dest not in ("help", "version")
    }
    parser.set_defaults(**{k: v for k, v in merged.items() if k in dests})
    args = parser.parse_args(argv_rest)
    _normalize_path_args_after_parse(args=args, workspace=workspace)
    errors = _validate_pipeline_args(args)
    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    try:
        _auto_fill_local_mineru_settings(args=args, workspace=workspace)
        backend = getattr(args, "backend")

        mineru_src: Path | None = None
        if backend == "mineru":
            mpr = getattr(args, "mineru_project_root", "")
            mineru_src = Path(str(mpr).strip()).resolve()
            if not (mineru_src / "mineru").is_dir():
                print(f"MinerU 源码路径无效（缺少 mineru/）: {mineru_src}", file=sys.stderr)
                return 1

        input_path = Path(args.input).resolve()

        pdf_files = _collect_pdf_files(input_path, recursive=args.recursive)
        if not pdf_files:
            print(f"未找到 PDF: {input_path}", file=sys.stderr)
            return 1

        json_out_dir, md_out_dir, output_root = _resolve_output_layout(
            workspace=workspace,
            input_path=input_path,
            output_dir=args.output_dir,
            json_output_dir=args.json_output_dir,
            markdown_output_dir=args.markdown_output_dir,
        )
        json_out_dir.mkdir(parents=True, exist_ok=True)
        md_out_dir.mkdir(parents=True, exist_ok=True)

        mineru_raw_dir = (output_root / "_mineru_raw").resolve()
        odl_work_root = (output_root / "_opendataloader_work").resolve()

        detect_fn = None
        if args.visual_tagging == "vlm":
            cred = _require_vlm_config(args)
            if cred is None:
                print(
                    "使用 --visual-tagging vlm 时必须配置 vlm_api_base 与 vlm_model。",
                    file=sys.stderr,
                )
                return 1
            api_base, model, timeout_sec, chat_path = cred
            detect_fn = build_openai_compatible_vlm_detector(
                api_base=api_base,
                api_key=args.vlm_api_key if isinstance(args.vlm_api_key, str) and args.vlm_api_key else None,
                model=model,
                timeout_sec=timeout_sec,
                system_prompt=args.vlm_system_prompt if isinstance(args.vlm_system_prompt, str) else None,
                user_prompt=args.vlm_user_prompt if isinstance(args.vlm_user_prompt, str) else None,
                chat_completions_path=chat_path,
            )

        page_transcriber = None
        if args.vlm_fallback != "off":
            cred_fb = _require_vlm_config(args)
            if cred_fb is None:
                print(
                    "使用 --vlm-fallback 时必须配置 vlm_api_base 与 vlm_model。",
                    file=sys.stderr,
                )
                return 1
            api_base, model, timeout_sec, chat_path = cred_fb
            min_page = float(args.vlm_page_transcribe_min_timeout_sec)
            page_transcriber = build_openai_compatible_vlm_page_transcriber(
                api_base=api_base,
                api_key=args.vlm_api_key if isinstance(args.vlm_api_key, str) and args.vlm_api_key else None,
                model=model,
                timeout_sec=max(timeout_sec, min_page),
                system_prompt=args.vlm_page_system_prompt if isinstance(args.vlm_page_system_prompt, str) else None,
                user_prompt_template=(
                    args.vlm_page_user_prompt_template
                    if isinstance(args.vlm_page_user_prompt_template, str)
                    else None
                ),
                chat_completions_path=chat_path,
            )

        if not args.quiet:
            print(f"后端: {backend}", flush=True)
            print(f"JSON 输出: {json_out_dir}", flush=True)
            print(f"Markdown 输出: {md_out_dir}", flush=True)
            if backend == "mineru" and mineru_src is not None:
                print(f"MinerU 源码: {mineru_src}", flush=True)
                print(f"MinerU 原始输出: {mineru_raw_dir}", flush=True)
                if not args.mineru_api_url:
                    print(
                        "提示: 未设置 mineru_api_url 时会临时启动 mineru-api；可常驻服务后填写 mineru_api_url。",
                        flush=True,
                    )
            if backend == "opendataloader" and getattr(args, "hybrid", None) != "off":
                if getattr(args, "hybrid_force_ocr", False):
                    print(
                        "提示: hybrid_force_ocr=true 时，Docling Fast 须带 --force-ocr 启动后才生效；"
                        "请执行 .\\scripts\\start_docling_hybrid.ps1（读取 pipeline.json）或重启现有 hybrid 进程。",
                        flush=True,
                    )

        for pdf in pdf_files:
            document: dict
            try:
                if backend == "mineru":
                    assert mineru_src is not None
                    mineru_raw_dir.mkdir(parents=True, exist_ok=True)
                    content_list_path = run_mineru_cli_for_pdf(
                        pdf_file=pdf,
                        output_root=mineru_raw_dir,
                        mineru_project_root=mineru_src,
                        backend=args.mineru_backend,
                        api_url=args.mineru_api_url,
                        model_source=args.mineru_model_source,
                        mineru_tools_config_json=args.mineru_tools_config_json,
                        cli_timeout_sec=getattr(args, "mineru_cli_timeout_sec", None),
                    )
                    document = convert_mineru_content_list_to_document(
                        content_list_path=content_list_path,
                        source_pdf=pdf,
                    )
                    meta = document.get("extraction_meta")
                    if not isinstance(meta, dict):
                        meta = {}
                    meta["backend"] = "mineru"
                    meta["mineru_project_root"] = str(mineru_src)
                    meta["mineru_content_list"] = str(content_list_path)
                    document["extraction_meta"] = meta
                else:
                    # OpenDataLoader/Java 在 Windows 上对「工作目录含中文」不稳；镜像目录使用哈希前缀名保持纯 ASCII。
                    _odl_key = hashlib.sha256(str(pdf.resolve()).encode("utf-8")).hexdigest()[:24]
                    per_dir = (odl_work_root / f"odl_{_odl_key}").resolve()
                    if per_dir.exists():
                        shutil.rmtree(per_dir, ignore_errors=True)
                    json_path_odl, md_native = run_opendataloader_for_pdf(
                        pdf,
                        per_dir,
                        table_method=args.table_method,
                        reading_order=args.reading_order,
                        hybrid=None if args.hybrid == "off" else args.hybrid,
                        hybrid_url=args.hybrid_url if args.hybrid != "off" else None,
                        hybrid_mode=args.hybrid_mode,
                        hybrid_timeout=args.hybrid_timeout,
                        hybrid_fallback=args.hybrid_fallback,
                        quiet=args.quiet,
                        hybrid_health_timeout_sec=float(args.hybrid_health_timeout_sec),
                        skip_hybrid_health_check=bool(args.skip_health_check),
                    )
                    document = load_opendataloader_document(json_path_odl, pdf)
                    # Java 侧原生 .md 通常为连续正文，不含可靠页码分段；需要按页引用时应以 JSON（kids.page number）
                    # 及后续生成的 ``*_by_page.md``（document_to_markdown_by_page）为准。
                    # md_native 在部分环境下可能为 str，须先转为 Path 再调用 .is_file()。
                    md_native_p: Path | None = Path(md_native) if md_native else None
                    if md_native_p is not None and md_native_p.is_file():
                        dest_native = md_out_dir / f"{pdf.stem}_opendataloader_native.md"
                        shutil.copy2(md_native_p, dest_native)
                        meta_odl = document.get("extraction_meta")
                        if isinstance(meta_odl, dict):
                            meta_odl["opendataloader_native_markdown"] = str(dest_native)
            except (MinerUExtractionError, OpenDataLoaderExtractionError) as exc:
                print(f"抽取失败 {pdf}: {exc}", file=sys.stderr)
                return 1

            if args.vlm_fallback != "off" and page_transcriber is not None:
                need = args.vlm_fallback == "force" or document_needs_vlm_fallback(
                    document,
                    threshold=args.vlm_fallback_threshold,
                )
                if need:
                    tmpdir = Path(tempfile.mkdtemp(prefix="vlm_fallback_"))
                    try:
                        page_pngs = render_pdf_pages_to_png(
                            pdf,
                            tmpdir,
                            dpi=max(72, min(240, args.vlm_fallback_dpi)),
                        )

                        def _transcribe_wrap(p: Path, pn: int) -> str:
                            return page_transcriber(p, pn)

                        merge_vlm_page_transcripts(document, page_pngs, _transcribe_wrap)
                    except RuntimeError as exc:
                        print(f"VLM 回退不可用: {exc}", file=sys.stderr)
                        if args.vlm_fallback == "force":
                            return 1
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)

            if args.visual_tagging != "off":
                json_path = json_out_dir / f"{pdf.stem}.json"
                try:
                    with json_path.open("w", encoding="utf-8") as f:
                        json.dump(document, f, ensure_ascii=False, indent=2)
                    hit_count = enrich_document_with_visual_tags(
                        document=document,
                        json_file_path=json_path,
                        detect_fn=detect_fn,
                        min_score=args.visual_min_score,
                    )
                    with json_path.open("w", encoding="utf-8") as f:
                        json.dump(document, f, ensure_ascii=False, indent=2)
                    if not args.quiet and hit_count:
                        print(f"  视觉标签命中: {pdf.name} -> {hit_count}")
                except RuntimeError as exc:
                    print(f"视觉标签失败: {exc}", file=sys.stderr)
                    return 1
                except OSError as exc:
                    print(f"视觉标签读写失败: {exc}", file=sys.stderr)
                    return 1

            json_path_final = json_out_dir / f"{pdf.stem}.json"
            with json_path_final.open("w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)

            md_text = document_to_markdown(document)
            (md_out_dir / f"{pdf.stem}.md").write_text(md_text, encoding="utf-8")
            if args.markdown_by_page:
                md_by_page = document_to_markdown_by_page(document)
                (md_out_dir / f"{pdf.stem}_by_page.md").write_text(md_by_page, encoding="utf-8")

            if not args.quiet:
                extra = f" + {pdf.stem}_by_page.md" if args.markdown_by_page else ""
                print(f"完成: {pdf.name} -> {json_path_final.name} + {pdf.stem}.md{extra}")

        if not args.quiet:
            print(f"全部完成，共 {len(pdf_files)} 个 PDF。")
        return 0
    finally:
        autogen = getattr(args, "_mineru_autogen_tools_json", None)
        if isinstance(autogen, str) and autogen:
            try:
                Path(autogen).unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
