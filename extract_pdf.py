#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 opendataloader-pdf 将 PDF 批量转为 JSON。

- 纯 Java 管线：适合结构简单的 PDF。
- Hybrid（Docling）：复杂表格/版式建议先启动本地服务再转换（见 start_hybrid_server.ps1）。

单次 convert() 批量处理多个 PDF，仅启动一次 JVM。
可选视觉增强：识别签名/指印/印章并回填到最近段落或表格单元格（本地 CLIP 或 OpenAI 兼容 VLM API）。
Hybrid 模式下 Java 会把页面交给 Docling 服务，需保证 http://127.0.0.1:5002 可用。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from pipeline_config import (
    defaults_from_environment,
    load_config_file,
    merge_defaults,
    pop_config_path_from_argv,
)
from visual_tagging import enrich_document_with_visual_tags
from vlm_client import build_openai_compatible_vlm_detector


DEFAULT_HYBRID_URL = "http://127.0.0.1:5002"


def _hybrid_health_ok(url_base: str, timeout_sec: float = 2.0) -> bool:
    base = url_base.rstrip("/")
    for path in ("/health", "/"):
        try:
            req = urllib.request.Request(f"{base}{path}", method="GET")
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            continue
    return False


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
        description="PDF 批量信息抽取（OpenDataLoader PDF，可选 Hybrid Docling）",
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
        help="输出根目录（默认：<输入目录>/opendataloader_out/）",
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
        "--hybrid",
        choices=("off", "docling-fast", "hancom-ai"),
        default="docling-fast",
        help="Hybrid 后端：docling-fast（默认，需本地 Docling 服务）、hancom-ai、off=纯 Java",
    )
    parser.add_argument(
        "--hybrid-url",
        default=DEFAULT_HYBRID_URL,
        help=f"Hybrid 服务地址（默认 {DEFAULT_HYBRID_URL}；可用环境变量 OPENDATALOADER_HYBRID_URL）",
    )
    parser.add_argument(
        "--hybrid-mode",
        choices=("auto", "full"),
        default="auto",
        help="auto=智能分流；full=全部页面走 hybrid（版式极难时可试 full）",
    )
    parser.add_argument(
        "--hybrid-timeout",
        default="0",
        help="Hybrid 请求超时（毫秒），0 表示不限制；对应 convert(hybrid_timeout=...)",
    )
    parser.add_argument(
        "--hybrid-fallback",
        action="store_true",
        help="hybrid 失败时回退到 Java 管线（默认关闭）",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="不检查 hybrid 服务是否已启动",
    )
    parser.add_argument(
        "--table-method",
        choices=("default", "cluster"),
        default="cluster",
        help="表格检测（仅纯 Java 管线；hybrid 下由 Docling 主导，此选项影响有限）",
    )
    parser.add_argument(
        "--reading-order",
        choices=("xycut", "off"),
        default="xycut",
        help="阅读顺序（仅纯 Java 管线）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默日志",
    )
    parser.add_argument(
        "--visual-tagging",
        choices=("off", "clip", "vlm"),
        default="clip",
        help="视觉标签：off=关闭；clip=本地 CLIP；vlm=OpenAI 兼容视觉 API（需配置 vlm_api_base / vlm_model）",
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

    try:
        import opendataloader_pdf
    except ImportError:
        print("请先安装: pip install -r requirements.txt", file=sys.stderr)
        return 1

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

    hybrid_backend = None if args.hybrid == "off" else args.hybrid

    if hybrid_backend and not args.skip_health_check:
        if not _hybrid_health_ok(args.hybrid_url):
            print(
                "未检测到 Hybrid 服务（请先在一个单独终端启动）：\n"
                "  .\\start_hybrid_server.ps1\n"
                "或:\n"
                "  opendataloader-pdf-hybrid --port 5002 --host 127.0.0.1 --ocr-lang ch_sim,en\n"
                f"期望地址: {args.hybrid_url}/health\n"
                "若服务已在其他端口，请使用: --hybrid-url http://127.0.0.1:端口\n"
                "跳过检查（不推荐）: --skip-health-check",
                file=sys.stderr,
            )
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

    convert_kw = dict(
        input_path=[str(pdf) for pdf in pdf_files],
        output_dir=str(json_out_dir),
        format="json",
        keep_line_breaks=True,
        markdown_page_separator="\n\n--- 第 %page-number% 页 ---\n\n",
        quiet=args.quiet,
    )

    if hybrid_backend:
        convert_kw.update(
            hybrid=hybrid_backend,
            hybrid_mode=args.hybrid_mode,
            hybrid_url=args.hybrid_url,
            hybrid_timeout=str(args.hybrid_timeout),
            hybrid_fallback=args.hybrid_fallback,
        )
    else:
        convert_kw.update(
            reading_order=args.reading_order,
            table_method=args.table_method,
        )

    opendataloader_pdf.convert(**convert_kw)

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
