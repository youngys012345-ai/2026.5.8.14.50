#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 MinerU 将 PDF 批量转为 JSON，并适配为现有流程兼容结构。

说明：
- 提取引擎已切换为 MinerU，不再依赖 opendataloader-pdf。
- 产出 JSON 保持现有 query/visual_tagging 流程可消费的结构（kids/table/image 等）。
可选视觉增强（默认关闭；``--visual-tagging clip|vlm``）：识别签名/指印/印章并回填到最近段落或表格单元格。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mineru_adapter import (
    MinerUExtractionError,
    convert_mineru_content_list_to_document,
    run_mineru_cli_for_pdf,
)
from pipeline_config import (
    defaults_from_environment,
    load_config_file,
    merge_defaults,
    pop_config_path_from_argv,
)
from visual_tagging import enrich_document_with_visual_tags
from vlm_client import build_openai_compatible_vlm_detector


def _auto_fill_local_mineru_settings(args: argparse.Namespace, workspace: Path) -> None:
    """自动补全本地模型配置，默认仍优先使用已安装的 MinerU package。"""
    local_cfg = workspace / "config" / "mineru.local.json"
    if args.mineru_tools_config_json is None and local_cfg.is_file():
        args.mineru_tools_config_json = str(local_cfg.resolve())
    if args.mineru_model_source is None and args.mineru_tools_config_json:
        args.mineru_model_source = "local"


def _resolve_default_input() -> Path | None:
    """在未传入输入路径时，尝试回退到项目中的示例 PDF。"""
    here = Path(__file__).resolve().parent
    candidate = here / "标讯-启东市吕四港镇大洋港小学综合楼工程的招标公告.pdf"
    sample = here.parent / "pdfs" / "复杂跨页表格样例.pdf"
    if candidate.is_file():
        return candidate
    if sample.is_file():
        print(
            f"未指定 PDF 且未找到招标公告文件，已使用样例: {sample}",
            file=sys.stderr,
        )
        return sample
    return None


def _collect_pdf_files(input_path: Path, recursive: bool = False) -> list[Path]:
    """收集待处理的 PDF 文件（支持单文件或目录批量）。"""
    if input_path.is_file():
        return [input_path.resolve()]
    if input_path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        return sorted(p.resolve() for p in input_path.glob(pattern) if p.is_file())
    return []


def _resolve_json_output_dir(input_path: Path, output_dir: str | None, json_output_dir: str | None) -> Path:
    """解析 JSON 专用输出目录。"""
    if json_output_dir:
        return Path(json_output_dir).resolve()
    if output_dir:
        return (Path(output_dir).resolve() / "json").resolve()
    base_dir = input_path.parent if input_path.is_file() else input_path
    return (base_dir / "opendataloader_out" / "json").resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDF 批量信息抽取（MinerU 提取 + 现有结构兼容）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON 配置文件路径（可与环境变量叠加；CLI 覆盖配置文件），示例见 config/pipeline.example.json",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="输入路径：可为单个 PDF 或目录（默认：项目内置样例 PDF）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="输出根目录（默认：<输入目录>/opendataloader_out/，为兼容旧流程保留目录名）",
    )
    parser.add_argument(
        "--json-output-dir",
        default=None,
        help="JSON 专用输出目录（默认：<输出根目录>/json/）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="当输入为目录时，是否递归搜索子目录中的 PDF",
    )
    parser.add_argument(
        "--mineru-backend",
        default="pipeline",
        help="MinerU backend（默认 pipeline，可改为 hybrid-http-client / vlm-http-client 等）",
    )
    parser.add_argument(
        "--mineru-api-url",
        default=None,
        help="MinerU API 服务地址（可选，未提供时使用本地 CLI 默认行为）",
    )
    parser.add_argument(
        "--mineru-model-source",
        choices=("local", "huggingface", "modelscope"),
        default=None,
        help="MinerU 模型来源（local/huggingface/modelscope）。建议已下载本地模型时设为 local",
    )
    parser.add_argument(
        "--mineru-tools-config-json",
        default=None,
        help="MinerU 配置文件名或绝对路径（默认读取 ~/mineru.json）",
    )
    parser.add_argument(
        "--mineru-project-dir",
        default=None,
        help="本地 MinerU 源码项目目录（例如 ./MinerU）。提供后优先使用该项目执行提取",
    )
    # 以下参数仅为兼容旧脚本而保留，不参与 MinerU 抽取逻辑。
    parser.add_argument(
        "--hybrid",
        choices=("off", "docling-fast", "hancom-ai"),
        default="off",
        help="已弃用参数（保留兼容，不影响 MinerU 执行；默认 off）",
    )
    parser.add_argument(
        "--hybrid-url",
        default="http://127.0.0.1:5002",
        help="已弃用参数（保留兼容，不影响 MinerU 执行）",
    )
    parser.add_argument(
        "--hybrid-mode",
        choices=("auto", "full"),
        default="auto",
        help="已弃用参数（保留兼容，不影响 MinerU 执行）",
    )
    parser.add_argument(
        "--hybrid-timeout",
        default="0",
        help="已弃用参数（保留兼容，不影响 MinerU 执行）",
    )
    parser.add_argument(
        "--hybrid-fallback",
        action="store_true",
        help="已弃用参数（保留兼容，不影响 MinerU 执行）",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="已弃用参数（保留兼容，不影响 MinerU 执行）",
    )
    parser.add_argument(
        "--table-method",
        choices=("default", "cluster"),
        default="cluster",
        help="已弃用参数（保留兼容，不影响 MinerU 执行）",
    )
    parser.add_argument(
        "--reading-order",
        choices=("xycut", "off"),
        default="xycut",
        help="已弃用参数（保留兼容，不影响 MinerU 执行）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默日志",
    )
    parser.add_argument(
        "--visual-tagging",
        choices=("off", "clip", "vlm"),
        default="off",
        help="视觉标签回填：off=关闭（默认）；clip=本地 CLIP；vlm=OpenAI 兼容视觉 API（需配置 vlm_api_base / vlm_model）",
    )
    parser.add_argument(
        "--visual-min-score",
        type=float,
        default=0.5,
        help="视觉标签最小置信度阈值（默认 0.5）",
    )
    parser.add_argument(
        "--vlm-api-base",
        default=None,
        help="VLM 服务根 URL（OpenAI 兼容），例：https://api.openai.com",
    )
    parser.add_argument(
        "--vlm-api-key",
        default=None,
        help="VLM API Key；也可用环境变量 OPENDATALOADER_VLM_API_KEY",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="VLM 模型名；也可用 OPENDATALOADER_VLM_MODEL",
    )
    parser.add_argument(
        "--vlm-timeout-sec",
        type=float,
        default=None,
        help="单次 VLM 请求超时秒数（默认 120；未配置时可用 OPENDATALOADER_VLM_TIMEOUT_SEC）",
    )
    parser.add_argument(
        "--vlm-chat-path",
        default=None,
        help="chat completions 路径（默认 /v1/chat/completions）",
    )
    parser.add_argument(
        "--vlm-system-prompt",
        default=None,
        help="覆盖默认 system 提示词（高级用法）",
    )
    parser.add_argument(
        "--vlm-user-prompt",
        default=None,
        help="覆盖默认 user 提示词（高级用法）",
    )
    return parser


def main() -> int:
    """解析命令行：支持先从任意位置取出 --config，再与环境变量、配置文件合并默认值。"""
    config_from_argv, argv_rest = pop_config_path_from_argv(sys.argv[1:])
    parser = _build_parser()
    merged = merge_defaults({}, defaults_from_environment())
    cfg_path = config_from_argv
    if cfg_path is not None:
        if not cfg_path.is_file():
            print(f"配置文件不存在: {cfg_path}", file=sys.stderr)
            return 1
        merged.update(load_config_file(cfg_path))
    dests = {
        a.dest
        for a in parser._actions
        if getattr(a, "dest", None) and a.dest not in ("help", "version")
    }
    parser.set_defaults(**{k: v for k, v in merged.items() if k in dests})
    args = parser.parse_args(argv_rest)
    workspace = Path(__file__).resolve().parent
    _auto_fill_local_mineru_settings(args=args, workspace=workspace)

    if args.input:
        input_path = Path(args.input).resolve()
    else:
        default_input = _resolve_default_input()
        if default_input is None:
            print(
                "请传入输入路径（PDF 文件或目录），例如:\n"
                "  python extract_pdf.py \"D:\\\\path\\\\pdf目录\"\n"
                "  python extract_pdf.py \"D:\\\\path\\\\文件.pdf\"",
                file=sys.stderr,
            )
            return 1
        input_path = default_input

    pdf_files = _collect_pdf_files(input_path, recursive=args.recursive)
    if not pdf_files:
        print(f"未找到可处理的 PDF: {input_path}", file=sys.stderr)
        return 1

    detect_fn = None
    if args.visual_tagging == "vlm":
        api_base = getattr(args, "vlm_api_base", None)
        model = getattr(args, "vlm_model", None)
        if not api_base or not model:
            print(
                "使用 --visual-tagging vlm 时必须提供 vlm_api_base 与 vlm_model：\n"
                "  配置文件（vlm_api_base / vlm_model）、或环境变量 "
                "OPENDATALOADER_VLM_API_BASE、OPENDATALOADER_VLM_MODEL、\n"
                "  或命令行 --vlm-api-base、--vlm-model",
                file=sys.stderr,
            )
            return 1
        timeout_sec = getattr(args, "vlm_timeout_sec", None)
        if timeout_sec is None:
            timeout_sec = 120.0
        chat_path = getattr(args, "vlm_chat_path", None) or "/v1/chat/completions"
        sys_prompt = getattr(args, "vlm_system_prompt", None)
        usr_prompt = getattr(args, "vlm_user_prompt", None)
        api_key = getattr(args, "vlm_api_key", None)
        detect_fn = build_openai_compatible_vlm_detector(
            api_base=api_base,
            api_key=api_key if isinstance(api_key, str) and api_key else None,
            model=model,
            timeout_sec=float(timeout_sec),
            system_prompt=sys_prompt if isinstance(sys_prompt, str) else None,
            user_prompt=usr_prompt if isinstance(usr_prompt, str) else None,
            chat_completions_path=chat_path,
        )

    json_out_dir = _resolve_json_output_dir(
        input_path=input_path,
        output_dir=args.output_dir,
        json_output_dir=args.json_output_dir,
    )
    json_out_dir.mkdir(parents=True, exist_ok=True)
    mineru_raw_dir = (json_out_dir / "_mineru_raw").resolve()
    mineru_raw_dir.mkdir(parents=True, exist_ok=True)
    for pdf in pdf_files:
        try:
            content_list_path = run_mineru_cli_for_pdf(
                pdf_file=pdf,
                output_root=mineru_raw_dir,
                backend=args.mineru_backend,
                api_url=args.mineru_api_url,
                model_source=args.mineru_model_source,
                mineru_tools_config_json=args.mineru_tools_config_json,
                mineru_project_dir=args.mineru_project_dir,
            )
            document = convert_mineru_content_list_to_document(
                content_list_path=content_list_path,
                source_pdf=pdf,
            )
            json_path = json_out_dir / f"{pdf.stem}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            if not args.quiet:
                print(f"MinerU 抽取完成: {pdf.name} -> {json_path.name}")
                if args.mineru_project_dir:
                    print(f"  使用本地 MinerU 项目: {args.mineru_project_dir}")
                if args.mineru_model_source:
                    print(f"  模型来源: {args.mineru_model_source}")
        except MinerUExtractionError as exc:
            print(f"MinerU 抽取失败: {pdf}，原因: {exc}", file=sys.stderr)
            return 1

    visual_total_hits = 0
    if args.visual_tagging != "off":
        for pdf in pdf_files:
            json_path = json_out_dir / f"{pdf.stem}.json"
            if not json_path.is_file():
                continue
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    document = json.load(f)
                hit_count = enrich_document_with_visual_tags(
                    document=document,
                    json_file_path=json_path,
                    detect_fn=detect_fn,
                    min_score=args.visual_min_score,
                )
                visual_total_hits += hit_count
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(document, f, ensure_ascii=False, indent=2)
                summary = (
                    document.get("visual_tag_stats", {}).get("summary_sentence")
                    if isinstance(document.get("visual_tag_stats"), dict)
                    else None
                )
                if isinstance(summary, str) and summary:
                    print(f"  视觉摘要 [{json_path.name}]: {summary}")
            except RuntimeError as exc:
                print(f"视觉标签增强失败: {exc}", file=sys.stderr)
                break
            except OSError as exc:
                print(f"视觉标签增强失败（文件读写异常）: {json_path}，原因: {exc}", file=sys.stderr)
                continue

    print(f"完成。共处理 PDF: {len(pdf_files)}")
    print(f"JSON 输出目录: {json_out_dir}")
    if args.visual_tagging != "off":
        print(f"视觉标签回填数量: {visual_total_hits}")
    for pdf in pdf_files:
        print(f"  - {json_out_dir / (pdf.stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
