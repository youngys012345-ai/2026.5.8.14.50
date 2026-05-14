#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_flow 包内自包含副本：环境变量与 JSON 管线配置相关工具（与仓库根 ``pipeline_config.py`` 逻辑一致）。

说明：``file_flow.pipeline_merge.load_merged_pipeline_config`` **仅**调用 ``load_config_file`` 读取磁盘 ``pipeline.json``，
不与 ``defaults_from_environment()`` 做字典合并；密钥等请用 ``file_flow/.env`` 或系统环境。

下列「合并优先级」适用于仓库根 ``extract_pdf`` 等仍使用 ``merge_defaults(..., defaults_from_environment())`` 的场景：
pipeline.json → 环境变量（OPENDATALOADER_* 等）→ 命令行参数。

业务默认值写在 ``file_flow/pipeline.json``；部署相关覆盖放在 ``file_flow/.env`` 或环境变量。

未设置 ``OPENDATALOADER_VLM_*`` 时，VLM 的 ``vlm_api_base`` 可从 ``LLM_API_BASE`` 读取（须为完整
``http(s)`` Chat Completions POST URL）；``LLM_MODEL``、``LLM_API_KEY`` 分别映射到
``vlm_model``、``vlm_api_key``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# 与 extract_pdf 中参数对应的可配置键（便于日后扩展，勿随意删改键名）
CONFIG_KEYS = frozenset(
    {
        "backend",
        "hybrid",
        "hybrid_url",
        "hybrid_mode",
        "hybrid_fallback",
        "hybrid_timeout",
        "skip_health_check",
        "table_method",
        "reading_order",
        "quiet",
        "visual_tagging",
        "visual_min_score",
        "vlm_api_base",
        "vlm_api_key",
        "vlm_model",
        "vlm_timeout_sec",
        "vlm_chat_path",
        "vlm_system_prompt",
        "vlm_user_prompt",
        "recursive",
        "output_dir",
        "json_output_dir",
        "markdown_output_dir",
        "vlm_fallback",
        "vlm_fallback_threshold",
        "vlm_fallback_dpi",
        "vlm_page_system_prompt",
        "vlm_page_user_prompt_template",
        "mineru_backend",
        "mineru_api_url",
        "mineru_model_source",
        "mineru_tools_config_json",
        "mineru_cli_timeout_sec",
        "mineru_project_root",
        "input",
        "hybrid_health_timeout_sec",
        "vlm_page_transcribe_min_timeout_sec",
        "markdown_by_page",
        "hybrid_force_ocr",
        "hybrid_ocr_lang",
        # 评审标准 LLM 环节（review_standard_llm_fill.py）
        "review_standard_json",
        "review_standard_markdown",
        "review_standard_pdf_text_file",
        "review_standard_output",
        # 评审栏目问答（review_standard_field_qa.py）
        "review_field_qa_input",
        "review_field_qa_output",
        # file_flow（仅依赖本仓库 pipeline.json + 本模块）
        "file_flow_pdf_dir",
        "file_flow_schema_json",
        "file_flow_out_dir",
        "file_flow_llm_api_base",
        "file_flow_llm_model",
        "file_flow_llm_timeout_sec",
        "file_flow_llm_system_prompt",
        "file_flow_llm_input",
        "file_flow_llm_output",
        "file_flow_schema_extract_system_prompt",
        "file_flow_steps",
        "file_flow_llm_extract",
        "file_flow_auto_batch",
        "file_flow_render_html_input",
        "file_flow_render_html_output",
        "file_flow_render_title",
        # file_flow：按 standards 清单评审（standards_llm_review.py）
        "file_flow_standards_json",
        "file_flow_standards_review_system_prompt",
        "file_flow_review_work_input",
        "file_flow_review_result_output",
        "file_flow_review_attach_schema_digest",
        "file_flow_pdf_text_backend",
        "file_flow_pdf_fallback_pymupdf",
        # 产出 JSON 文件名后缀（不含 .json），默认 _work / _answered / _review
        "file_flow_suffix_work",
        "file_flow_suffix_answered",
        "file_flow_suffix_review",
    }
)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str | None) -> str | None:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def defaults_from_environment() -> dict[str, Any]:
    """从环境变量读取管线默认值（预留与脚本、容器编排对接）。"""
    d: dict[str, Any] = {}
    # MinerU 参数
    mb = _env_str("MINERU_BACKEND", None) or _env_str("OPENDATALOADER_MINERU_BACKEND", None)
    if mb is not None:
        d["mineru_backend"] = mb
    mu = _env_str("MINERU_API_URL", None) or _env_str("OPENDATALOADER_MINERU_API_URL", None)
    if mu is not None:
        d["mineru_api_url"] = mu
    mms = _env_str("MINERU_MODEL_SOURCE", None) or _env_str("OPENDATALOADER_MINERU_MODEL_SOURCE", None)
    if mms is not None:
        d["mineru_model_source"] = mms
    mtcj = _env_str("MINERU_TOOLS_CONFIG_JSON", None) or _env_str("OPENDATALOADER_MINERU_TOOLS_CONFIG_JSON", None)
    if mtcj is not None:
        d["mineru_tools_config_json"] = mtcj
    mcts = _env_str("MINERU_CLI_TIMEOUT_SEC", None) or _env_str(
        "OPENDATALOADER_MINERU_CLI_TIMEOUT_SEC", None
    )
    if mcts is not None and mcts != "":
        try:
            d["mineru_cli_timeout_sec"] = float(mcts)
        except ValueError:
            pass

    mpr = _env_str("MINERU_PROJECT_ROOT", None) or _env_str("OPENDATALOADER_MINERU_PROJECT_ROOT", None)
    if mpr is not None:
        d["mineru_project_root"] = mpr

    hu = _env_str("OPENDATALOADER_HYBRID_URL", None)
    if hu is not None:
        d["hybrid_url"] = hu
    hm = _env_str("OPENDATALOADER_HYBRID_MODE", None)
    if hm is not None:
        d["hybrid_mode"] = hm
    hb = _env_str("OPENDATALOADER_HYBRID_BACKEND", None)
    if hb is not None:
        d["hybrid"] = hb
    if os.environ.get("OPENDATALOADER_HYBRID_FALLBACK"):
        d["hybrid_fallback"] = _env_bool("OPENDATALOADER_HYBRID_FALLBACK", False)
    ht = _env_str("OPENDATALOADER_HYBRID_TIMEOUT", None)
    if ht is not None:
        d["hybrid_timeout"] = ht
    hhs = _env_str("OPENDATALOADER_HYBRID_HEALTH_TIMEOUT_SEC", None)
    if hhs is not None and hhs != "":
        try:
            d["hybrid_health_timeout_sec"] = float(hhs)
        except ValueError:
            pass
    if os.environ.get("OPENDATALOADER_SKIP_HEALTH_CHECK"):
        d["skip_health_check"] = _env_bool("OPENDATALOADER_SKIP_HEALTH_CHECK", False)

    # 视觉增强（CLIP / VLM）
    vt = _env_str("OPENDATALOADER_VISUAL_TAGGING", None)
    if vt is not None:
        d["visual_tagging"] = vt
    if os.environ.get("OPENDATALOADER_VISUAL_MIN_SCORE"):
        d["visual_min_score"] = _env_float("OPENDATALOADER_VISUAL_MIN_SCORE", 0.5)
    # VLM（OpenAI 兼容视觉接口）
    vb = _env_str("OPENDATALOADER_VLM_API_BASE", None)
    if vb is not None:
        d["vlm_api_base"] = vb
    vk = _env_str("OPENDATALOADER_VLM_API_KEY", None)
    if vk is not None:
        d["vlm_api_key"] = vk
    vm = _env_str("OPENDATALOADER_VLM_MODEL", None)
    if vm is not None:
        d["vlm_model"] = vm
    if os.environ.get("OPENDATALOADER_VLM_TIMEOUT_SEC"):
        d["vlm_timeout_sec"] = _env_float("OPENDATALOADER_VLM_TIMEOUT_SEC", 120.0)
    vcp = _env_str("OPENDATALOADER_VLM_CHAT_PATH", None)
    if vcp is not None:
        d["vlm_chat_path"] = vcp
    vsp = _env_str("OPENDATALOADER_VLM_SYSTEM_PROMPT", None)
    if vsp is not None:
        d["vlm_system_prompt"] = vsp
    vup = _env_str("OPENDATALOADER_VLM_USER_PROMPT", None)
    if vup is not None:
        d["vlm_user_prompt"] = vup
    # 与评审 LLM 环节共用 LLM_*（仅当未配置 OPENDATALOADER_VLM_* 对应项时）
    if "vlm_api_base" not in d:
        llm_b = _env_str("LLM_API_BASE", None)
        if llm_b is not None:
            d["vlm_api_base"] = llm_b
    if "vlm_api_key" not in d:
        llm_k = _env_str("LLM_API_KEY", None)
        if llm_k is not None:
            d["vlm_api_key"] = llm_k
    if "vlm_model" not in d:
        llm_m = _env_str("LLM_MODEL", None)
        if llm_m is not None:
            d["vlm_model"] = llm_m
    # 输出与遍历
    od = _env_str("OPENDATALOADER_OUTPUT_DIR", None)
    if od is not None:
        d["output_dir"] = od
    pdf_in = _env_str("OPENDATALOADER_INPUT", None) or _env_str("OPENDATALOADER_PDF", None)
    if pdf_in is not None:
        d["input"] = pdf_in
    ob = _env_str("OPENDATALOADER_BACKEND", None)
    if ob is not None:
        d["backend"] = ob
    jd = _env_str("OPENDATALOADER_JSON_OUTPUT_DIR", None)
    if jd is not None:
        d["json_output_dir"] = jd
    md_out = _env_str("OPENDATALOADER_MARKDOWN_OUTPUT_DIR", None)
    if md_out is not None:
        d["markdown_output_dir"] = md_out
    vf = _env_str("OPENDATALOADER_VLM_FALLBACK", None)
    if vf is not None:
        d["vlm_fallback"] = vf
    vft = _env_str("OPENDATALOADER_VLM_FALLBACK_THRESHOLD", None)
    if vft is not None and vft != "":
        try:
            d["vlm_fallback_threshold"] = int(vft)
        except ValueError:
            pass
    if os.environ.get("OPENDATALOADER_RECURSIVE"):
        d["recursive"] = _env_bool("OPENDATALOADER_RECURSIVE", False)
    if os.environ.get("OPENDATALOADER_QUIET"):
        d["quiet"] = _env_bool("OPENDATALOADER_QUIET", False)
    if os.environ.get("OPENDATALOADER_MARKDOWN_BY_PAGE"):
        d["markdown_by_page"] = _env_bool("OPENDATALOADER_MARKDOWN_BY_PAGE", False)
    if os.environ.get("OPENDATALOADER_HYBRID_FORCE_OCR"):
        d["hybrid_force_ocr"] = _env_bool("OPENDATALOADER_HYBRID_FORCE_OCR", False)
    hol = _env_str("OPENDATALOADER_HYBRID_OCR_LANG", None)
    if hol is not None:
        d["hybrid_ocr_lang"] = hol

    vpmt = _env_str("OPENDATALOADER_VLM_PAGE_TRANSCRIBE_MIN_TIMEOUT_SEC", None)
    if vpmt is not None and vpmt != "":
        try:
            d["vlm_page_transcribe_min_timeout_sec"] = float(vpmt)
        except ValueError:
            pass
    return d


def load_config_file(path: Path) -> dict[str, Any]:
    """加载 JSON 配置文件，忽略未知键并做类型约束。"""
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件必须是 JSON 对象: {path}")
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in CONFIG_KEYS:
            continue
        if v is None:
            continue
        out[k] = v
    return out


def merge_defaults(
    base: dict[str, Any],
    *overlays: dict[str, Any],
) -> dict[str, Any]:
    """合并多组默认值，后面的覆盖前面的。"""
    merged = dict(base)
    for layer in overlays:
        for k, v in layer.items():
            merged[k] = v
    return merged


def resolve_pipeline_config_path(requested: Path) -> tuple[Path | None, str | None]:
    """
    解析实际存在的管线配置文件路径。
    若用户拼错常见文件名（如 pipline.json），且同目录下存在修正后的文件，则改用后者。
    返回 (存在的路径, 给 stderr 的简短中文提示)；找不到则 (None, None)。
    """
    if requested.is_file():
        return requested.resolve(), None
    parent = requested.parent
    name = requested.name
    # 常见拼写：pipline -> pipeline
    if name.lower() == "pipline.json":
        alt = parent / "pipeline.json"
        if alt.is_file():
            return alt.resolve(), f"提示: 未找到 {name}，已改用 {alt.name}"
    return None, None


def pop_config_path_from_argv(argv: list[str]) -> tuple[Path | None, list[str]]:
    """
    从命令行参数中取出 --config，返回 (配置文件路径, 其余参数)。
    支持 --config path 与 --config=path，可出现在任意位置。
    """
    out: list[str] = []
    config_path: Path | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = Path(argv[i + 1])
            i += 2
            continue
        if argv[i].startswith("--config="):
            config_path = Path(argv[i].split("=", 1)[1])
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return config_path, out
