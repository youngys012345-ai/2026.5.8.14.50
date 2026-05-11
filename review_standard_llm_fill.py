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

    # 路径写在项目根 pipeline.json 的 review_standard_* 键中时可省略文件参数：
    python review_standard_llm_fill.py

    # 或与管线一致显式指定配置文件后仍可用命令行覆盖单一路径：
    set LLM_API_BASE=https://api.example.com/v1
    set LLM_API_KEY=sk-...
    set LLM_MODEL=gpt-4o-mini
    python review_standard_llm_fill.py --json 评审标准.json --markdown output/markdown/某案_by_page.md -o 评审标准_filled.json

pipeline.json 中与本节相关的键（``load_config_file`` 会读取；``__`` 前缀键为说明，忽略）：

- ``review_standard_json``：评审标准 JSON 路径（相对路径优先当前工作目录，其次项目根）。
- ``review_standard_markdown``：extract_pdf 生成的 markdown（建议 ``*_by_page.md``）。
- ``review_standard_pdf_text_file``：若不使用 markdown，可填纯文本抽取文件；与上一项二选一。
- ``review_standard_output``：写出路径；未配置时默认 ``{评审标准文件名}_llm_filled.json``。

合并优先级：**命令行参数** 优先于 **pipeline.json**（与 extract_pdf 一致）。

---------------------------------------------------------------------------
环节变量文件（.env）
---------------------------------------------------------------------------

启动时会自动加载（需 ``pip install python-dotenv``）：

1. 项目根 ``.env`` — 不覆盖操作系统里已存在的环境变量；
2. 项目根 ``环节变量.env`` — 覆盖上一步中已写入的同名键（便于把大模型密钥单独放此文件）；
3. 当前工作目录下的 ``.env`` / ``环节变量.env`` — 在后加载，后者同名键覆盖前者。

``LLM_API_BASE``、``LLM_MODEL`` 等仍在 ``load_llm_config_from_env()`` 中读取；须在加载 .env **之后**调用，故本脚本在 ``main`` 最先解析完命令行后即加载上述文件。
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

from pipeline_config import load_config_file, resolve_pipeline_config_path  # noqa: E402


def _resolve_input_path(raw: str, workspace: Path) -> Path:
    """
    将配置或命令行中的输入文件路径解析为绝对路径。
    相对路径：优先当前工作目录下存在则取之，否则用项目根（与 extract_pdf 一致）。
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
    """输出路径：相对路径锚定项目根，避免在其他目录启动时写到意外位置。"""
    p = Path(raw.strip())
    if p.is_absolute():
        return p.resolve()
    return (workspace / p).resolve()


def _load_step_env_files(workspace: Path) -> tuple[list[Path], bool]:
    """
    加载环节变量 .env，供 LLM_* / OPENAI_* 等注入 os.environ。
    返回 (已加载文件的绝对路径列表, 是否因未安装 python-dotenv 而跳过)。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return [], True

    loaded: list[Path] = []
    # (路径, override)：先加载通用 .env，再由环节专用文件与当前目录覆盖
    steps: list[tuple[Path, bool]] = [
        (workspace / ".env", False),
        (workspace / "环节变量.env", True),
        (Path.cwd() / ".env", True),
        (Path.cwd() / "环节变量.env", True),
    ]
    for path, override in steps:
        if path.is_file():
            load_dotenv(path, override=override)
            loaded.append(path.resolve())
    return loaded, False


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
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="管线 JSON（默认尝试项目根 pipeline.json）；从中读取 review_standard_* 路径。",
    )
    p.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="评审标准等 JSON 路径；未传则使用配置中的 review_standard_json。",
    )
    p.add_argument(
        "--markdown",
        dest="markdown_path",
        type=Path,
        default=None,
        help="extract_pdf 等生成的 markdown（建议带页码的 _by_page.md）；未传则用 review_standard_markdown。",
    )
    p.add_argument(
        "--pdf-text-file",
        dest="pdf_text_file",
        type=Path,
        default=None,
        help="若未提供 markdown，可用纯文本；未传则用 review_standard_pdf_text_file。",
    )
    p.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        default=None,
        help="输出 JSON 路径；未传则用 review_standard_output，仍缺省则为输入文件名加 _llm_filled.json",
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
    return p.parse_args(argv)


def _apply_pipeline_paths_to_args(
    args: argparse.Namespace,
    workspace: Path,
    pipeline_cfg: dict[str, Any],
) -> None:
    """命令行已给出的路径不覆盖；仅从 pipeline 补齐缺失项。"""
    if args.json_path is None:
        raw = pipeline_cfg.get("review_standard_json")
        if isinstance(raw, str) and raw.strip():
            args.json_path = _resolve_input_path(raw, workspace)
    if args.markdown_path is None:
        raw = pipeline_cfg.get("review_standard_markdown")
        if isinstance(raw, str) and raw.strip():
            args.markdown_path = _resolve_input_path(raw, workspace)
    if args.pdf_text_file is None:
        raw = pipeline_cfg.get("review_standard_pdf_text_file")
        if isinstance(raw, str) and raw.strip():
            args.pdf_text_file = _resolve_input_path(raw, workspace)
    if args.output_path is None:
        raw = pipeline_cfg.get("review_standard_output")
        if isinstance(raw, str) and raw.strip():
            args.output_path = _resolve_output_path(raw, workspace)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace = _repo_root
    env_loaded, dotenv_missing = _load_step_env_files(workspace)
    configure_logging(level=args.log_level, log_file=args.log_file)
    if dotenv_missing:
        _LOG.warning(
            "[环节:环境] 未安装 python-dotenv，已跳过 .env 加载；请执行 pip install python-dotenv"
        )
    elif env_loaded:
        _LOG.info(
            "[环节:环境] 已从环节变量文件载入 %s 个: %s",
            len(env_loaded),
            ", ".join(str(p) for p in env_loaded),
        )
    else:
        _LOG.info(
            "[环节:环境] 未找到项目根或当前目录下的 .env / 环节变量.env（可忽略若仅用系统环境变量）"
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
            _LOG.exception("[环节:配置] 无法加载 %s: %s", cfg_file, e)
            print(f"错误: 无法读取管线配置: {cfg_file}\n{e}", file=sys.stderr)
            raise SystemExit(1) from e
        _LOG.info("[环节:配置] 已载入 pipeline 片段 path=%s 键数=%s", cfg_file.resolve(), len(pipeline_cfg))
    elif cfg_file is not None:
        _LOG.warning("[环节:配置] 未找到配置文件，已忽略: %s", cfg_file)

    if args.json_path is not None:
        args.json_path = _resolve_input_path(str(args.json_path), workspace)
    if args.markdown_path is not None:
        args.markdown_path = _resolve_input_path(str(args.markdown_path), workspace)
    if args.pdf_text_file is not None:
        args.pdf_text_file = _resolve_input_path(str(args.pdf_text_file), workspace)
    if args.output_path is not None:
        p = args.output_path
        args.output_path = p if p.is_absolute() else _resolve_output_path(str(p), workspace)

    _apply_pipeline_paths_to_args(args, workspace, pipeline_cfg)

    if args.json_path is None:
        print(
            "错误: 未指定评审标准 JSON。请在命令行使用 --json，"
            "或在 pipeline.json 中设置 review_standard_json。",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.markdown_path is None and args.pdf_text_file is None:
        print(
            "错误: 未指定 PDF 抽取文本来源。请使用 --markdown 或 --pdf-text-file，"
            "或在 pipeline.json 中设置 review_standard_markdown / review_standard_pdf_text_file。",
            file=sys.stderr,
        )
        raise SystemExit(1)

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
