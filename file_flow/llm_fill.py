#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件模式第二步：读取 pdf_prepare 生成的工作 JSON，将「文书类型 + 栏目 + 内容 + 问题」拼成用户消息，
与系统提示一并调用大模型，把返回写入该栏目的「回答」。

环境变量（与 ``review_standard_field_qa`` / ``review_standard_llm_fill`` 一致，由项目根 ``.env`` 注入）::

    LLM_API_BASE   完整 Chat Completions POST URL
    LLM_API_KEY、LLM_API_KEY_BACKUP1、LLM_API_KEY_BACKUP2（备用；429/503 时由 ``call_openai_compatible_chat`` 轮询换钥重试）
    LLM_MODEL

本脚本位于 ``file_flow/`` 下，仍通过 ``ensure_step_dotenv_loaded(项目根)`` 加载**仓库根目录**的 ``.env`` /
``环节变量.env``（与 ``step_dotenv`` 约定一致），不依赖当前工作目录是否在子文件夹。

系统提示优先顺序：``FILE_FLOW_SYSTEM_PROMPT`` → ``REVIEW_FIELD_QA_SYSTEM_PROMPT`` → 内置默认。

日志：``REVIEW_STANDARD_LLM_LOG_LEVEL``；非 dry-run 时首条请求会打印 system / user / 助手回复（过长截断，与 field_qa 一致）。

用法::

    python file_flow/llm_fill.py -i file_flow/out/某案_work.json -o file_flow/out/某案_answered.json
    python file_flow/llm_fill.py -i ... --dry-run
    python file_flow/llm_fill.py -i ... -o ... --log-file logs/llm_fill.log
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

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from step_dotenv import ensure_step_dotenv_loaded  # noqa: E402

ensure_step_dotenv_loaded(_ROOT)

from review_standard_llm_fill import (  # noqa: E402
    LlmEnvConfig,
    call_openai_compatible_chat,
    configure_logging,
    iter_top_level_sections,
    load_llm_config_from_env,
)
from vlm_client import is_http_endpoint_url  # noqa: E402

_LOG = logging.getLogger(__name__)

# 首条 LLM 请求/响应写入日志时的单段最大字符数（与 review_standard_field_qa 一致）
_FIRST_LLM_LOG_MAX_CHARS = 20000


def _preview_for_log(text: str, max_chars: int = _FIRST_LLM_LOG_MAX_CHARS) -> str:
    """用于日志的正文预览；超长时截断并注明总长度。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…（共 {len(text)} 字符，已按 max={max_chars} 截断）"


# 可用 FILE_FLOW_SYSTEM_PROMPT 或 REVIEW_FIELD_QA_SYSTEM_PROMPT 覆盖
DEFAULT_SYSTEM = (
    "你是行政执法案卷评审助手。用户会提供某一栏目从案卷中抽取的相关文字、以及需要对照的评审问题。"
    "请严格依据「抽取内容」作答：先给出简要结论，再说明依据；不得臆测材料中不存在的内容。"
    "若材料不足以判断，须明确说明「无法判断」并简述原因。使用简体中文。"
)


def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def load_qa_llm_config() -> LlmEnvConfig:
    base = load_llm_config_from_env()
    sys_prompt = (
        _env_first("FILE_FLOW_SYSTEM_PROMPT", "REVIEW_FIELD_QA_SYSTEM_PROMPT") or DEFAULT_SYSTEM
    )
    return replace(base, system_prompt=sys_prompt)


def build_user_prompt(section_title: str, field_name: str, content: str, question: str) -> str:
    """拼接用户消息：栏目材料 + 评审问题。"""
    c = content.strip() if content else "（抽取内容为空）"
    q = question.strip() if question else "（未配置问题，请结合材料作一般性说明）"
    return (
        f"【文书类型】{section_title}\n\n"
        f"【栏目名称】{field_name}\n\n"
        f"【与案卷相关的抽取内容】\n{c}\n\n"
        f"【评审问题】\n{q}\n\n"
        "请根据「抽取内容」回答「评审问题」。若内容与问题无关，请说明。"
    )


def fill_answers(
    root: dict[str, Any],
    cfg: LlmEnvConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """遍历各文书「字段」，写入「回答」。"""
    out: dict[str, Any] = json.loads(json.dumps(root, ensure_ascii=False))
    meta = out.pop("_file_flow_meta", None)

    total_calls = 0
    for section_title, block in iter_top_level_sections(out):
        if section_title.startswith("_"):
            continue
        fields_obj = block.get("字段")
        if not isinstance(fields_obj, dict):
            continue
        total_calls += sum(1 for _fn, fo in fields_obj.items() if isinstance(fo, dict))

    _LOG.info("[环节:栏目标注] 预计调用次数=%s（每个子字段一次）", total_calls)

    # 仅首条真实请求打印 system + user；仅首次成功返回打印助手正文（与 review_standard_field_qa 一致）
    first_llm_request_logged = False
    first_llm_response_logged = False

    done = 0
    for section_title, block in iter_top_level_sections(out):
        if section_title.startswith("_"):
            continue
        fields_obj = block.get("字段")
        if not isinstance(fields_obj, dict):
            _LOG.warning("[环节:跳过] 文书「%s」无「字段」对象", section_title)
            continue

        for field_name, field_obj in fields_obj.items():
            if not isinstance(field_obj, dict):
                continue
            done += 1
            content = field_obj.get("内容", "")
            if not isinstance(content, str):
                content = str(content)
            question = field_obj.get("问题", "")
            if not isinstance(question, str):
                question = str(question)
            if not content.strip():
                _LOG.warning(
                    "[环节:上下文为空] 文书「%s」栏目「%s」抽取内容为空，仍将仅按问题尝试作答",
                    section_title,
                    field_name,
                )
            user = build_user_prompt(section_title, field_name, content, question)
            _LOG.info(
                "[环节:栏目] (%s/%s) 文书=%s 栏目=%s 用户消息字符数=%s",
                done,
                total_calls,
                section_title,
                field_name,
                len(user),
            )
            if dry_run:
                field_obj["回答"] = "[dry-run 未调用大模型]"
                continue
            if not first_llm_request_logged:
                _LOG.info(
                    "[环节:首条LLM请求] system 提示（%s 字符）:\n%s",
                    len(cfg.system_prompt),
                    _preview_for_log(cfg.system_prompt),
                )
                _LOG.info(
                    "[环节:首条LLM请求] user 消息（%s 字符）:\n%s",
                    len(user),
                    _preview_for_log(user),
                )
                first_llm_request_logged = True
            text = call_openai_compatible_chat(cfg, user)
            if not first_llm_response_logged:
                _LOG.info(
                    "[环节:首条LLM响应] 助手回复（%s 字符）:\n%s",
                    len(text),
                    _preview_for_log(text),
                )
                first_llm_response_logged = True
            field_obj["回答"] = text

    if isinstance(meta, dict):
        out["_file_flow_meta"] = meta
    return out


def _resolve_path(raw: Path, cwd: Path) -> Path:
    if raw.is_absolute():
        return raw.resolve()
    for base in (cwd, _ROOT):
        hit = (base / raw).resolve()
        if hit.is_file():
            return hit
    return (_ROOT / raw).resolve()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="工作 JSON 栏目级大模型填答（文件模式）")
    ap.add_argument("-i", "--input", type=Path, required=True, help="*_work.json 或已装配的 JSON")
    ap.add_argument("-o", "--output", type=Path, required=True, help="写出路径，如 *_answered.json")
    ap.add_argument("--dry-run", action="store_true", help="不请求 API，回答写占位句")
    ap.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="日志级别；默认 INFO 或环境变量 REVIEW_STANDARD_LLM_LOG_LEVEL",
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="追加写入日志文件（UTF-8）",
    )
    ns = ap.parse_args(argv)

    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(_ROOT)
    configure_logging(level=ns.log_level, log_file=ns.log_file)
    if dotenv_missing:
        _LOG.warning(
            "[环节:环境] 未安装 python-dotenv，已跳过 .env；请 pip install python-dotenv"
        )
    elif env_loaded:
        _LOG.info("[环节:环境] 已加载环节变量文件 %s 个", len(env_loaded))

    in_path = _resolve_path(ns.input, Path.cwd())
    out_path = Path(ns.output)
    out_path = out_path.resolve() if out_path.is_absolute() else (Path.cwd() / out_path).resolve()

    if not in_path.is_file():
        print(f"错误: 找不到输入: {in_path}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取或解析 JSON: {e}", file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print("错误: JSON 根须为对象", file=sys.stderr)
        return 1

    cfg = load_qa_llm_config()
    dry_run = ns.dry_run
    if not dry_run and (
        not cfg.api_base
        or not cfg.model
        or not is_http_endpoint_url((cfg.api_base or "").strip())
    ):
        _LOG.warning(
            "[环节:配置] 缺少有效的 LLM_API_BASE（须为完整 http(s) URL）或 LLM_MODEL，自动 dry-run"
        )
        print(
            "警告: 未配置完整的大模型 endpoint URL 或模型名，仅执行 dry-run。",
            file=sys.stderr,
        )
        dry_run = True

    try:
        filled = fill_answers(data, cfg, dry_run=dry_run)
    except RuntimeError as e:
        _LOG.exception("[环节:失败] %s", e)
        return 1

    out_path.write_text(json.dumps(filled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _LOG.info("[环节:写文件] path=%s", out_path.resolve())
    print(f"已写出: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
