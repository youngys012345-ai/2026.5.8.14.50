#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据评审标准 JSON 的一级标题，结合已抽取的 PDF 文本（通常为 extract_pdf 产出的 markdown），
按标题顺序调用大模型，提取对应表格及关联表格的页码与内容，并写回各一级块下的「大模型返回结果」。
每一轮用户消息仅重复拼接同一份 PDF 抽取正文与本轮任务句，不附带前几轮模型输出，以节省上下文。

---------------------------------------------------------------------------
Prompt 模板定义位置（换机迁移时优先核对）
---------------------------------------------------------------------------

1. **系统提示（system）**：本模块常量 ``DEFAULT_SYSTEM_PROMPT``；若设置环境变量
   ``LLM_SYSTEM_PROMPT``，则运行时覆盖该常量。
2. **用户提示（user）**：函数 ``build_table_extraction_user_prompt`` 内的字符串拼接；
   固定结构为「【PDF 抽取结果】+ 全文 + 【本轮任务】+ 提取句式（含一级字段名）」。

---------------------------------------------------------------------------
日志（便于定位环节：配置 / 读文件 / 拼 prompt / 请求 API / 写结果）
---------------------------------------------------------------------------

- 环境变量 ``REVIEW_STANDARD_LLM_LOG_LEVEL``：如 ``DEBUG``、``INFO``（默认 ``INFO``）。
- 命令行 ``--log-level``、``--log-file``：覆盖级别、追加写入日志文件。

大模型通过环境变量配置（不设置则可在 --dry-run 下只生成/预览 prompt，不实际请求）：

- ``LLM_API_BASE``：API 根地址，如 ``https://api.openai.com`` 或兼容服务；兼容 ``OPENAI_API_BASE``。
- ``LLM_API_KEY``：Bearer Token；兼容 ``OPENAI_API_KEY``。
- ``LLM_MODEL``：模型名；兼容 ``OPENAI_MODEL``。
- ``LLM_CHAT_PATH``：对话路径，默认 ``/v1/chat/completions``。
- ``LLM_TIMEOUT_SEC``：秒，默认 ``120``。
- ``LLM_SYSTEM_PROMPT``：系统提示，未设置则使用本模块内默认的表格提取说明。

用法示例::

    set LLM_API_BASE=https://api.example.com/v1
    set LLM_API_KEY=sk-...
    set LLM_MODEL=gpt-4o-mini
    python review_standard_llm_fill.py --json 评审标准.json --markdown output/markdown/某案_by_page.md -o 评审标准_filled.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# 本模块日志器：环节标签便于换机后对照终端输出
_LOG = logging.getLogger(__name__)

# 与同目录其它模块一致，支持非传统启动方式
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from vlm_client import _extract_message_content  # noqa: E402


def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def configure_logging(level: int | str | None = None, log_file: Path | None = None) -> None:
    """
    初始化日志：默认 INFO；可用环境变量 REVIEW_STANDARD_LLM_LOG_LEVEL 指定。
    若传入 log_file，则同时追加写入该文件（便于远程机器留存）。
    """
    if level is None:
        raw = _env_first("REVIEW_STANDARD_LLM_LOG_LEVEL", "LLM_LOG_LEVEL")
        if raw:
            level = raw.upper()
        else:
            level = logging.INFO
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    # force=True：重复入口（如测试多次调用 main）时覆盖旧 handler，避免重复行
    try:
        logging.basicConfig(
            level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True
        )
    except TypeError:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    _LOG.setLevel(level)


# ---------------------------------------------------------------------------
# 系统提示模板：与 LLM_SYSTEM_PROMPT 环境变量二选一（变量优先）
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "你是政务/执法案卷材料分析助手。用户会提供从 PDF 抽取的全文或按页markdown，"
    "请根据其中页码标记与表格结构，定位并提取用户指定名称的文书表格及其在文档中明确关联的表格。"
    "输出须包含：涉及的页码（或页码范围）、表格正文（可用 Markdown 表格或条理清晰的文本复述单元格）。"
    "若文中不存在该表，说明未找到并列出可能相近的标题。"
)


@dataclass(frozen=True)
class LlmEnvConfig:
    """从大模型相关环境变量解析得到的连接参数（仅占位与运行时读取，不在代码里写死密钥）。"""

    api_base: str | None
    api_key: str | None
    model: str | None
    chat_path: str
    timeout_sec: float
    system_prompt: str


def _mask_secret(s: str | None) -> str:
    """日志中脱敏密钥，仅表示是否配置。"""
    if not s:
        return "(未设置)"
    if len(s) <= 8:
        return "已设置(已隐藏)"
    return f"已设置(尾四位 …{s[-4:]})"


def load_llm_config_from_env() -> LlmEnvConfig:
    """从环境变量加载大模型配置（预留接口，便于 CI/本地分别注入）。"""
    chat_path = _env_first("LLM_CHAT_PATH") or "/v1/chat/completions"
    timeout_raw = _env_first("LLM_TIMEOUT_SEC")
    try:
        timeout_sec = float(timeout_raw) if timeout_raw is not None else 120.0
    except ValueError:
        timeout_sec = 120.0
    sys_msg = _env_first("LLM_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
    cfg = LlmEnvConfig(
        api_base=_env_first("LLM_API_BASE", "OPENAI_API_BASE"),
        api_key=_env_first("LLM_API_KEY", "OPENAI_API_KEY"),
        model=_env_first("LLM_MODEL", "OPENAI_MODEL"),
        chat_path=chat_path if chat_path.startswith("/") else f"/{chat_path}",
        timeout_sec=timeout_sec,
        system_prompt=sys_msg,
    )
    _LOG.info(
        "[环节:配置] 已读取环境变量：api_base=%s model=%s chat_path=%s timeout=%s api_key=%s system_prompt来源=%s",
        cfg.api_base or "(未设置)",
        cfg.model or "(未设置)",
        cfg.chat_path,
        cfg.timeout_sec,
        _mask_secret(cfg.api_key),
        "LLM_SYSTEM_PROMPT" if _env_first("LLM_SYSTEM_PROMPT") else "DEFAULT_SYSTEM_PROMPT",
    )
    return cfg


def iter_top_level_sections(data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """遍历根对象下一级且值为对象的条目（视为「一级字段」块）。"""
    for key, val in data.items():
        if isinstance(val, dict):
            yield key, val


def build_table_extraction_user_prompt(
    pdf_extraction_text: str,
    section_title: str,
) -> str:
    """
    构造用户消息：仅拼接 PDF 抽取全文与本轮任务句（每轮相同结构，不附带前几轮模型摘要）。

    用户提示模板即下方字面量拼接（修改措辞只改此处）。
    """
    return (
        "【PDF 抽取结果（全文或按页 markdown，供定位页码与表格）】\n"
        + pdf_extraction_text.strip()
        + f"\n\n【本轮任务】\n提取「{section_title}」这张表以及与这张表相关的表格，返回页码和表格内容。"
    )


def call_openai_compatible_chat(
    cfg: LlmEnvConfig,
    user_text: str,
) -> str:
    """OpenAI 兼容 ``/v1/chat/completions`` 文本对话，返回助手正文字符串。"""
    if not cfg.api_base or not cfg.model:
        raise RuntimeError("缺少 LLM_API_BASE 或 LLM_MODEL（或兼容环境变量），无法调用大模型。")
    base = cfg.api_base.rstrip("/")
    path = cfg.chat_path
    url = f"{base}{path}"
    _LOG.info(
        "[环节:API请求] POST %s model=%s 超时=%ss 用户消息字符数=%s",
        url,
        cfg.model,
        cfg.timeout_sec,
        len(user_text),
    )
    _LOG.debug(
        "[环节:prompt预览] system 前80字=%s",
        (cfg.system_prompt[:80] + "…") if len(cfg.system_prompt) > 80 else cfg.system_prompt,
    )
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_text},
        ],
    }
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        _LOG.error(
            "[环节:API错误] HTTP %s URL=%s 响应正文片段=%s",
            e.code,
            url,
            detail[:500] + ("…" if len(detail) > 500 else ""),
        )
        raise RuntimeError(f"大模型 HTTP 错误 {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        _LOG.error("[环节:API错误] 网络/URL 异常 URL=%s err=%s", url, e)
        raise RuntimeError(f"大模型连接失败: {e}") from e
    parsed = json.loads(body)
    text = _extract_message_content(parsed).strip()
    _LOG.info("[环节:API响应] 助手回复字符数=%s", len(text))
    return text


RESULT_FIELD = "大模型返回结果"


def fill_review_standard_json(
    root: dict[str, Any],
    pdf_extraction_text: str,
    cfg: LlmEnvConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    按一级字段顺序轮询：每轮 prompt 均为「同一份 PDF 全文 + 本轮任务」，调用大模型或 dry-run，
    将返回写入对应块下的 ``大模型返回结果``。
    """
    out = json.loads(json.dumps(root, ensure_ascii=False))  # 深拷贝

    sections = [(t, b) for t, b in iter_top_level_sections(out)]
    total = len(sections)
    _LOG.info("[环节:轮询] 共 %s 个一级字段待处理", total)

    for idx, (title, block) in enumerate(sections, start=1):
        user_prompt = build_table_extraction_user_prompt(pdf_extraction_text, title)
        _LOG.info(
            "[环节:轮询] (%s/%s) 一级字段=%s 用户消息字符数=%s",
            idx,
            total,
            title,
            len(user_prompt),
        )
        if dry_run:
            block[RESULT_FIELD] = "[dry-run 未调用大模型]"
            _LOG.info("[环节:轮询] (%s/%s) dry-run，跳过 API", idx, total)
            continue
        text = call_openai_compatible_chat(cfg, user_prompt)
        block[RESULT_FIELD] = text
        _LOG.info("[环节:轮询] (%s/%s) 已写入字段「%s」", idx, total, RESULT_FIELD)
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按评审标准一级字段轮询大模型，把表格提取结果写入 JSON。"
    )
    p.add_argument(
        "--json",
        dest="json_path",
        required=True,
        type=Path,
        help="评审标准等 JSON 路径（根级多个一级字段块）。",
    )
    p.add_argument(
        "--markdown",
        dest="markdown_path",
        type=Path,
        default=None,
        help="extract_pdf 等生成的 markdown（建议带页码的 _by_page.md）。",
    )
    p.add_argument(
        "--pdf-text-file",
        dest="pdf_text_file",
        type=Path,
        default=None,
        help="若未提供 --markdown，可用纯文本文件作为 PDF 抽取结果。",
    )
    p.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        default=None,
        help="输出 JSON 路径；默认在输入文件名后加 _llm_filled.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="不调用大模型，仅写入占位字段并打印首个 prompt 长度信息。",
    )
    p.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="日志级别；不设则使用环境变量 REVIEW_STANDARD_LLM_LOG_LEVEL，否则为 INFO。",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="同时写入该路径的日志文件（UTF-8）。",
    )
    ns = p.parse_args(argv)
    if ns.markdown_path is None and ns.pdf_text_file is None:
        p.error("请指定 --markdown 或 --pdf-text-file 之一作为 PDF 抽取内容来源")
    return ns


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(level=args.log_level, log_file=args.log_file)

    _LOG.info(
        "[环节:启动] json=%s pdf文本源=%s 输出=%s dry_run=%s",
        args.json_path.resolve(),
        (args.markdown_path or args.pdf_text_file),
        args.output_path or "(默认 _llm_filled)",
        args.dry_run,
    )

    cfg = load_llm_config_from_env()

    text_src = args.markdown_path or args.pdf_text_file
    assert text_src is not None
    try:
        pdf_text = text_src.read_text(encoding="utf-8")
    except OSError as e:
        _LOG.exception("[环节:读文件失败] 无法读取 PDF 抽取文本: %s", text_src)
        raise SystemExit(1) from e
    _LOG.info(
        "[环节:读文件] PDF 抽取文本已载入 path=%s 字符数=%s",
        text_src.resolve(),
        len(pdf_text),
    )

    try:
        data = json.loads(args.json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _LOG.exception("[环节:解析JSON失败] path=%s", args.json_path.resolve())
        raise SystemExit(1) from e
    if not isinstance(data, dict):
        _LOG.error("[环节:解析JSON] 根节点不是对象")
        raise SystemExit("JSON 根节点必须是对象")

    _LOG.info("[环节:解析JSON] 成功，顶层键数量=%s", len(data))

    out_path = args.output_path
    if out_path is None:
        stem = args.json_path.stem
        out_path = args.json_path.with_name(f"{stem}_llm_filled.json")

    if not args.dry_run and (not cfg.api_base or not cfg.model):
        _LOG.warning(
            "[环节:配置] 未设置 LLM_API_BASE / LLM_MODEL，自动按 dry-run 处理（无真实模型调用）"
        )
        print(
            "警告：未配置 LLM_API_BASE / LLM_MODEL，将仅能做 dry-run。"
            "已自动启用 --dry-run 行为（不写真实模型输出）。",
            file=sys.stderr,
        )
        args.dry_run = True

    try:
        filled = fill_review_standard_json(data, pdf_text, cfg, dry_run=args.dry_run)
        out_path.write_text(
            json.dumps(filled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except RuntimeError as e:
        _LOG.exception("[环节:失败] %s", e)
        raise SystemExit(1) from e

    _LOG.info("[环节:写文件] 已写入 path=%s 字节约=%s", out_path.resolve(), out_path.stat().st_size)
    print(f"已写入: {out_path.resolve()}")
    if args.dry_run:
        first_key = next(iter(filled.keys())) if filled else None
        if first_key and isinstance(filled.get(first_key), dict):
            sample = build_table_extraction_user_prompt(pdf_text, first_key)
            print(f"[dry-run] 首个 prompt 字符数: {len(sample)}")
    _LOG.info("[环节:结束] 全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
