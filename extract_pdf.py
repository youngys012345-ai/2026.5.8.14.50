#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 批量解析：

- **mineru**：在项目配置的本地 MinerU 源码目录下执行 ``python -m mineru.cli.client``，
  将 content_list 转为统一 kids 结构。
- **opendataloader**：调用 ``opendataloader_pdf.convert``（需 Java 11+）。

默认将结果写入项目下 ``output/json`` 与 ``output/markdown``（可用 ``-o`` 指定根目录）。

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

import argparse
import json
import os
import shutil
import tempfile

from document_export import document_to_markdown
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
    if getattr(args, "backend", "mineru") != "mineru":
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


def _auto_fill_mineru_project_root(args: argparse.Namespace, workspace: Path) -> None:
    """若未配置 mineru_project_root，则优先本项目内 ``MinerU/``，再尝试上级目录。"""
    raw = getattr(args, "mineru_project_root", None)
    if isinstance(raw, str) and raw.strip():
        return
    for cand in (workspace / "MinerU", workspace.parent / "MinerU"):
        if (cand / "mineru").is_dir():
            args.mineru_project_root = str(cand.resolve())
            return


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


def _resolve_default_input(workspace: Path) -> Path | None:
    """未传输入时：依次尝试 sample_pdfs、pdfs、历史示例文件名。"""
    for folder_name in ("sample_pdfs", "pdfs"):
        d = workspace / folder_name
        if d.is_dir():
            found = sorted(d.glob("*.pdf"))
            if found:
                return found[0].resolve()
    candidate = workspace / "标讯-启东市吕四港镇大洋港小学综合楼工程的招标公告.pdf"
    if candidate.is_file():
        return candidate.resolve()
    legacy = workspace.parent / "pdfs" / "复杂跨页表格样例.pdf"
    if legacy.is_file():
        print(f"未指定 PDF，已使用样例: {legacy}", file=sys.stderr)
        return legacy.resolve()
    return None


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
    返回 (json_dir, markdown_dir, output_root)。
    """
    if output_dir:
        output_root = Path(output_dir).resolve()
    else:
        output_root = (workspace / "output").resolve()
    json_dir = Path(json_output_dir).resolve() if json_output_dir else (output_root / "json").resolve()
    md_dir = (
        Path(markdown_output_dir).resolve()
        if markdown_output_dir
        else (output_root / "markdown").resolve()
    )
    return json_dir, md_dir, output_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDF 批量抽取（MinerU 源码 / OpenDataLoader），输出 JSON 与 Markdown",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON 配置文件（与环境变量、CLI 合并），示例见 config/pipeline.example.json",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="输入 PDF 或目录（默认：sample_pdfs / pdfs 中首个 PDF）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="输出根目录；其下默认包含 json/ 与 markdown/（未单独指定时）",
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
        default="mineru",
        help="抽取后端：mineru（本地 MinerU 源码）或 opendataloader（Java / opendataloader-pdf）",
    )
    parser.add_argument(
        "--mineru-project-root",
        default=None,
        help="MinerU 源码仓库根路径（须含 mineru/）；默认 ./MinerU 或 ../MinerU",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="目录输入时递归子目录中的 PDF",
    )
    parser.add_argument(
        "--mineru-backend",
        default="pipeline",
        help="MinerU -b backend（默认 pipeline；亦可用 hybrid-auto-engine 等与官方 CLI 一致）",
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
        default="off",
        help="OpenDataLoader hybrid（仅 backend=opendataloader）",
    )
    parser.add_argument(
        "--hybrid-url",
        default="http://127.0.0.1:5002",
        help="hybrid 服务 URL",
    )
    parser.add_argument(
        "--hybrid-mode",
        choices=("auto", "full"),
        default="auto",
        help="OpenDataLoader hybrid_mode",
    )
    parser.add_argument(
        "--hybrid-timeout",
        default="0",
        help="hybrid_timeout（毫秒，0 表示不限制）",
    )
    parser.add_argument(
        "--hybrid-fallback",
        action="store_true",
        help="hybrid 失败时回落 Java 原生管线",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="兼容占位（当前脚本未使用）",
    )
    parser.add_argument(
        "--table-method",
        choices=("default", "cluster"),
        default="cluster",
        help="OpenDataLoader table_method",
    )
    parser.add_argument(
        "--reading-order",
        choices=("xycut", "off"),
        default="xycut",
        help="OpenDataLoader reading_order",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式",
    )
    parser.add_argument(
        "--visual-tagging",
        choices=("off", "clip", "vlm"),
        default="off",
        help="视觉标签：off（默认）；clip；vlm（需配置 VLM）",
    )
    parser.add_argument(
        "--visual-min-score",
        type=float,
        default=0.5,
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
        help="chat completions 路径（默认 /v1/chat/completions）",
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
        default="off",
        help="页级 VLM 回退：off；auto（文本极短时）；force（每页调用，慎用成本）",
    )
    parser.add_argument(
        "--vlm-fallback-threshold",
        type=int,
        default=80,
        help="auto 模式下，结构化文本总长度低于该阈值则触发 VLM 回退",
    )
    parser.add_argument(
        "--vlm-fallback-dpi",
        type=int,
        default=120,
        help="VLM 回退时 PDF 渲染 DPI",
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


def _require_vlm_config(args: argparse.Namespace) -> tuple[str, str, float, str] | None:
    """若未配置完整 VLM 参数则返回 None。"""
    api_base = getattr(args, "vlm_api_base", None)
    model = getattr(args, "vlm_model", None)
    if not api_base or not model:
        return None
    timeout_sec = getattr(args, "vlm_timeout_sec", None)
    if timeout_sec is None:
        timeout_sec = 120.0
    chat_path = getattr(args, "vlm_chat_path", None) or "/v1/chat/completions"
    return api_base, model, float(timeout_sec), chat_path


def main() -> int:
    config_from_argv, argv_rest = pop_config_path_from_argv(sys.argv[1:])
    parser = _build_parser()
    merged = merge_defaults({}, defaults_from_environment())
    cfg_path = config_from_argv
    if cfg_path is not None:
        resolved_cfg, cfg_hint = resolve_pipeline_config_path(cfg_path)
        if resolved_cfg is None:
            print(f"配置文件不存在: {cfg_path}", file=sys.stderr)
            return 1
        if cfg_hint:
            print(cfg_hint, file=sys.stderr)
        cfg_path = resolved_cfg
        merged.update(load_config_file(cfg_path))
    dests = {
        a.dest
        for a in parser._actions
        if getattr(a, "dest", None) and a.dest not in ("help", "version")
    }
    parser.set_defaults(**{k: v for k, v in merged.items() if k in dests})
    args = parser.parse_args(argv_rest)
    workspace = Path(__file__).resolve().parent
    _normalize_path_args_after_parse(args=args, workspace=workspace)
    try:
        _auto_fill_local_mineru_settings(args=args, workspace=workspace)
        backend = getattr(args, "backend", "mineru")

        mineru_src: Path | None = None
        if backend == "mineru":
            _auto_fill_mineru_project_root(args=args, workspace=workspace)
            mpr = getattr(args, "mineru_project_root", None)
            if not isinstance(mpr, str) or not mpr.strip():
                print(
                    "请配置 mineru_project_root：指向本地 MinerU 仓库根目录（含 mineru/），"
                    "或置于本项目 MinerU/、上级 ../MinerU；也可设置环境变量 MINERU_PROJECT_ROOT。",
                    file=sys.stderr,
                )
                return 1
            mineru_src = Path(mpr.strip()).resolve()
            if not (mineru_src / "mineru").is_dir():
                print(f"MinerU 源码路径无效（缺少 mineru/）: {mineru_src}", file=sys.stderr)
                return 1

        if args.input:
            input_path = Path(args.input).resolve()
        else:
            default_input = _resolve_default_input(workspace)
            if default_input is None:
                print(
                    "请指定 PDF 或目录，或将示例 PDF 放入 sample_pdfs/ 目录。",
                    file=sys.stderr,
                )
                return 1
            input_path = default_input

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
            page_transcriber = build_openai_compatible_vlm_page_transcriber(
                api_base=api_base,
                api_key=args.vlm_api_key if isinstance(args.vlm_api_key, str) and args.vlm_api_key else None,
                model=model,
                timeout_sec=max(timeout_sec, 180.0),
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
                    per_dir = (odl_work_root / pdf.stem).resolve()
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
                    )
                    document = load_opendataloader_document(json_path_odl, pdf)
                    if md_native and md_native.is_file():
                        dest_native = md_out_dir / f"{pdf.stem}_opendataloader_native.md"
                        shutil.copy2(md_native, dest_native)
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

            if not args.quiet:
                print(f"完成: {pdf.name} -> {json_path_final.name} + {pdf.stem}.md")

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
