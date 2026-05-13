#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依据「评审标准_llm_filled.json」类文件：读取每个一级文书块下的「大模型返回结果」作为上下文，
对每个子字段结合「栏目名 + 要求」调用大模型作答，并将回复写入该子字段的「内容」。

前置条件：输入 JSON 须已由 review_standard_llm_fill.py 写入各文书类型的「大模型返回结果」。

合并优先级：命令行参数 优于 pipeline.json 中的 review_field_qa_*。

环境变量（与上一环节共用连接参数）：

- ``LLM_API_BASE``（完整 ``https://…`` Chat Completions POST URL）、``LLM_MODEL``、``LLM_API_KEY``（与 ``review_standard_llm_fill.load_llm_config_from_env`` 相同）；可选 ``LLM_API_KEY_BACKUP1``、``LLM_API_KEY_BACKUP2`` 在主密钥 429/503 时自动轮换（见 ``call_openai_compatible_chat``）。
- ``REVIEW_FIELD_QA_SYSTEM_PROMPT``：本环节专用系统提示；未设置则使用本模块默认评审作答提示。
- 日志级别可用 ``REVIEW_STANDARD_LLM_LOG_LEVEL``（与 fill 脚本共用 configure_logging）。
- 非 dry-run 时，首次调用大模型会在日志中输出**首条** system 提示、user 全文（过长则截断）及**首条**助手回复（过长则截断）。

用法::

    python review_standard_field_qa.py -i 评审标准_llm_filled.json -o 评审标准_answered.json
    python review_standard_field_qa.py --dry-run   # 依赖 pipeline.json 中 review_field_qa_input
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

_workspace = Path(__file__).resolve().parent
if str(_workspace) not in sys.path:
    sys.path.insert(0, str(_workspace))

from step_dotenv import ensure_step_dotenv_loaded  # noqa: E402

ensure_step_dotenv_loaded(_workspace)

from pipeline_config import load_config_file, resolve_pipeline_config_path  # noqa: E402

from review_standard_llm_fill import (  # noqa: E402
    RESULT_FIELD,
    LlmEnvConfig,
    call_openai_compatible_chat,
    configure_logging,
    iter_top_level_sections,
    load_llm_config_from_env,
)
from vlm_client import is_http_endpoint_url  # noqa: E402


def _resolve_input_path(raw: str, workspace: Path) -> Path:
    """
    将配置或命令行中的输入路径解析为绝对路径。
    相对路径：优先当前工作目录下已存在路径，否则项目根（与 review_standard_llm_fill 一致）。
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


def _resolve_output_path(raw: str, workspace: Path) -> Path:
    """输出路径：相对路径锚定项目根。"""
    p = Path(raw.strip())
    if p.is_absolute():
        return p.resolve()
    return (workspace / p).resolve()


CONTENT_FIELD = "内容"
HANDWRITING_FIELD = "是否需要识别手写体"

# 首条 LLM 请求/响应写入日志时的单段最大字符数（避免单条日志过大）
_FIRST_LLM_LOG_MAX_CHARS = 20000


def _preview_for_log(text: str, max_chars: int = _FIRST_LLM_LOG_MAX_CHARS) -> str:
    """用于日志的正文预览；超长时截断并注明总长度。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…（共 {len(text)} 字符，已按 max={max_chars} 截断）"

# 与 REVIEW_FIELD_QA_SYSTEM_PROMPT 二选一（变量优先于本常量）
DEFAULT_FIELD_QA_SYSTEM_PROMPT = (
    "你是行政执法案卷评审助手，以及针对具体栏目的评审要求。请严格依据上下文作答：逐条对照要求说明是否符合、依据何在；"
)


def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def load_field_qa_llm_config() -> LlmEnvConfig:
    """连接参数与 fill 环节相同；系统提示为本环节专用。"""
    base = load_llm_config_from_env()
    qa_sys = _env_first("REVIEW_FIELD_QA_SYSTEM_PROMPT") or DEFAULT_FIELD_QA_SYSTEM_PROMPT
    return replace(base, system_prompt=qa_sys)


def build_field_qa_user_prompt(
    section_title: str,
    llm_extraction: str,
    field_name: str,
    requirements: list[Any],
    handwritten: str | None,
) -> str:
    """
    拼接用户消息：文书类型 + 大模型抽取上下文 + 待评审栏目 + 要求列表。
    """
    lines: list[str] = []
    for item in requirements:
        if isinstance(item, str) and item.strip():
            lines.append(item.strip())
    req_text = "\n".join(lines)
    if not req_text:
        req_text = "（未列出具体条文，请结合上下文对该栏目作合规性判断与说明。）"

    parts: list[str] = [
        f"【文书类型】{section_title}",
        "【该类文书的大模型抽取结果（上下文）】",
        llm_extraction.strip() if llm_extraction else "（上一环节未返回抽取结果或为空）",
        f"【待评审栏目】{field_name}",
    ]
    if handwritten is not None and str(handwritten).strip():
        parts.append(f"【{HANDWRITING_FIELD}】{handwritten.strip()}")
    parts.append("【评审要求】")
    parts.append(req_text)
    parts.append(
        "请仅依据上述材料，逐条对照评审要求判断是否符合要求。请先给出结论，然后再给出简要判断依据。如果信息不全或依据不足，也认为不符合要求"
        "不得臆测材料中不存在的内容。"
    )
    return "\n\n".join(parts)


def fill_field_answers(
    root: dict[str, Any],
    cfg: LlmEnvConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    遍历每个一级文书块及其「字段」子项，将大模型回复写入各子项的「内容」。
    """
    out = json.loads(json.dumps(root, ensure_ascii=False))

    total_calls = 0
    for _title, block in iter_top_level_sections(out):
        fields_obj = block.get("字段")
        if not isinstance(fields_obj, dict):
            continue
        total_calls += sum(1 for _fn, fo in fields_obj.items() if isinstance(fo, dict))

    done = 0
    _LOG.info("[环节:栏目标注] 预计调用次数=%s（每个子字段一次）", total_calls)

    # 仅首条真实请求打印 system + user；仅首次成功返回打印助手正文（便于对照 API 行为）
    first_llm_request_logged = False
    first_llm_response_logged = False

    for section_title, block in iter_top_level_sections(out):
        fields_obj = block.get("字段")
        if not isinstance(fields_obj, dict):
            _LOG.warning("[环节:跳过] 文书「%s」无「字段」对象", section_title)
            continue

        ctx = block.get(RESULT_FIELD)
        if not isinstance(ctx, str) or not ctx.strip():
            _LOG.warning(
                "[环节:上下文为空] 文书「%s」缺少非空的「%s」，仍将尝试仅按栏目与要求作答",
                section_title,
                RESULT_FIELD,
            )
            ctx = ctx if isinstance(ctx, str) else ""

        for field_name, field_obj in fields_obj.items():
            if not isinstance(field_obj, dict):
                continue
            done += 1
            reqs = field_obj.get("要求")
            if not isinstance(reqs, list):
                reqs = []
            hw = field_obj.get(HANDWRITING_FIELD)
            hw_s = str(hw).strip() if hw is not None else None
            user_prompt = build_field_qa_user_prompt(
                section_title, ctx, field_name, reqs, hw_s
            )
            _LOG.info(
                "[环节:栏目] (%s/%s) 文书=%s 栏目=%s 用户消息字符数=%s",
                done,
                total_calls,
                section_title,
                field_name,
                len(user_prompt),
            )
            if dry_run:
                field_obj[CONTENT_FIELD] = "[dry-run 未调用大模型]"
                continue
            if not first_llm_request_logged:
                _LOG.info(
                    "[环节:首条LLM请求] system 提示（%s 字符）:\n%s",
                    len(cfg.system_prompt),
                    _preview_for_log(cfg.system_prompt),
                )
                _LOG.info(
                    "[环节:首条LLM请求] user 消息（%s 字符）:\n%s",
                    len(user_prompt),
                    _preview_for_log(user_prompt),
                )
                first_llm_request_logged = True
            text = call_openai_compatible_chat(cfg, user_prompt)
            if not first_llm_response_logged:
                _LOG.info(
                    "[环节:首条LLM响应] 助手回复（%s 字符）:\n%s",
                    len(text),
                    _preview_for_log(text),
                )
                first_llm_response_logged = True
            field_obj[CONTENT_FIELD] = text

    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="根据大模型抽取结果与评审要求，逐栏目调用大模型写入「内容」。"
    )
    p.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="管线 JSON（默认项目根 pipeline.json），读取 review_field_qa_input / review_field_qa_output。",
    )
    p.add_argument(
        "-i",
        "--input",
        dest="input_path",
        type=Path,
        default=None,
        help="含「大模型返回结果」的 JSON（通常为 *_llm_filled.json）。",
    )
    p.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        default=None,
        help="写出路径；默认 {输入 stem}_answered.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="不请求 API，仅写入占位「内容」",
    )
    p.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="日志级别；默认 INFO 或环境变量 REVIEW_STANDARD_LLM_LOG_LEVEL",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="追加写入日志文件（UTF-8）",
    )
    return p.parse_args(argv)


def _apply_pipeline_field_qa_paths(
    args: argparse.Namespace,
    workspace: Path,
    pipeline_cfg: dict[str, Any],
) -> None:
    if args.input_path is None:
        raw = pipeline_cfg.get("review_field_qa_input")
        if isinstance(raw, str) and raw.strip():
            args.input_path = _resolve_input_path(raw, workspace)
    if args.output_path is None:
        raw = pipeline_cfg.get("review_field_qa_output")
        if isinstance(raw, str) and raw.strip():
            args.output_path = _resolve_output_path(raw, workspace)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace = _workspace
    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(workspace)
    configure_logging(level=args.log_level, log_file=args.log_file)
    if dotenv_missing:
        _LOG.warning(
            "[环节:环境] 未安装 python-dotenv，已跳过 .env；请 pip install python-dotenv"
        )
    elif env_loaded:
        _LOG.info(
            "[环节:环境] 已加载环节变量文件 %s 个",
            len(env_loaded),
        )

    cfg_file = args.config_path
    if cfg_file is None:
        resolved, _ = resolve_pipeline_config_path(workspace / "pipeline.json")
        cfg_file = resolved
    pipeline_cfg: dict[str, Any] = {}
    if cfg_file is not None and cfg_file.is_file():
        try:
            pipeline_cfg = load_config_file(cfg_file)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"错误: 无法读取管线配置: {cfg_file}\n{e}", file=sys.stderr)
            raise SystemExit(1) from e

    if args.input_path is not None:
        args.input_path = _resolve_input_path(str(args.input_path), workspace)
    if args.output_path is not None:
        p = args.output_path
        args.output_path = p if p.is_absolute() else _resolve_output_path(str(p), workspace)

    _apply_pipeline_field_qa_paths(args, workspace, pipeline_cfg)

    if args.input_path is None:
        print(
            "错误: 请使用 -i/--input 指定输入 JSON，或在 pipeline.json 中设置 review_field_qa_input。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    cfg_qa = load_field_qa_llm_config()

    try:
        data = json.loads(args.input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取或解析输入 JSON: {args.input_path}\n{e}", file=sys.stderr)
        raise SystemExit(1) from e
    if not isinstance(data, dict):
        print("错误: JSON 根节点须为对象", file=sys.stderr)
        raise SystemExit(1)

    dry_run = args.dry_run
    if not dry_run and (
        not cfg_qa.api_base
        or not cfg_qa.model
        or not is_http_endpoint_url((cfg_qa.api_base or "").strip())
    ):
        _LOG.warning("[环节:配置] 缺少有效的 LLM_API_BASE（须为完整 https URL）或 LLM_MODEL，自动 dry-run")
        print(
            "警告: 未配置完整的大模型 endpoint URL 或模型名，仅执行 dry-run。",
            file=sys.stderr,
        )
        dry_run = True

    try:
        filled = fill_field_answers(data, cfg_qa, dry_run=dry_run)
    except RuntimeError as e:
        _LOG.exception("[环节:失败] %s", e)
        raise SystemExit(1) from e

    out_path = args.output_path
    if out_path is None:
        out_path = args.input_path.with_name(f"{args.input_path.stem}_answered.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(filled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _LOG.info("[环节:写文件] path=%s", out_path.resolve())
    print(f"已写入: {out_path.resolve()}")

    if dry_run:
        # 构造示例 prompt 长度便于核对
        for sec, blk in iter_top_level_sections(filled):
            fo = blk.get("字段")
            if isinstance(fo, dict) and fo:
                first_name = next(iter(fo.keys()))
                first_obj = fo.get(first_name)
                if isinstance(first_obj, dict):
                    ctx = blk.get(RESULT_FIELD, "")
                    req_list = first_obj.get("要求") if isinstance(first_obj.get("要求"), list) else []
                    sample = build_field_qa_user_prompt(
                        sec,
                        ctx if isinstance(ctx, str) else "",
                        first_name,
                        req_list,
                        first_obj.get(HANDWRITING_FIELD),
                    )
                    print(f"[dry-run] 首个栏目用户消息字符数: {len(sample)}")
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
