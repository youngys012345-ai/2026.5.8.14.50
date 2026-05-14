#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_flow 管线编排：读取 ``file_flow/pipeline.json``（或 ``--config`` 指定路径）中的允许键，并按配置顺序以 **Python 调用** 各步骤，
无需在终端分别敲 ``pdf_prepare`` / ``llm_fill`` / ``render_html``。

**推荐业务顺序**（与 ``file_flow_steps`` 默认一致）：

1. ``pdf_prepare``：PDF 全文 → 按 ``schema.json`` 生成 ``*_work.json``；若 ``file_flow_llm_extract=true``，
   按每字段 **field_name + description** 调用大模型从全文摘录写入 ``content``。
2. ``llm_fill``：按 **文书名称、字段名、字段说明、content** 与 ``standards.json`` 同下标的 **standard**（【评审问题】）拼接后调用大模型，写入 ``answer``。
3. ``standards_review``：按 ``standards.json`` 顶层数组逐项（``category`` / ``subcategory`` / ``content`` / ``standard`` 等）结合案卷摘录调用大模型，写入 ``standards_review``。
4. ``render_html``：渲染 HTML。

- ``load_merged_pipeline_config``：仅从磁盘 ``pipeline.json`` 加载已声明的键，**不**与仓库根配置、也不与 ``defaults_from_environment()`` 合并；密钥等请用 ``file_flow/.env`` 或系统环境变量（由各子模块直接读 ``os.environ``）。
- ``run_file_flow``：根据 ``file_flow_steps`` 依次执行；``file_flow_auto_batch`` 未出现在配置中时**默认开启**，
  对 ``file_flow_out_dir`` 下符合后缀规则的 JSON 批量执行 llm_fill、standards_review、render_html。
  产出文件名后缀由 ``file_flow_suffix_work`` / ``file_flow_suffix_answered`` / ``file_flow_suffix_review`` 控制（默认 ``_work`` / ``_answered`` / ``_review``）。

用法（在包含 ``file_flow`` 包的上级目录执行，一般为仓库根）::

    python -m file_flow.pipeline_merge
    python run_file_flow.py
    python -m file_flow.pipeline_merge --config my_pipeline.json --dry-run

或在代码中::

    from file_flow.pipeline_merge import run_file_flow
    run_file_flow()

编排键（写在 ``file_flow/pipeline.json``，仅列核心项）::

    file_flow_steps: ["pdf_prepare", "llm_fill", "standards_review", "render_html"]
    file_flow_pdf_dir / file_flow_schema_json / file_flow_standards_json / file_flow_out_dir
    file_flow_auto_batch: true
    file_flow_llm_extract: true
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

from .naming import (
    answered_glob_pattern,
    file_flow_stem_suffixes,
    replace_work_json_name_with_answered,
    review_glob_pattern,
    review_json_filename_for_base,
    stem_base_from_stage_stem,
    work_glob_pattern,
)
from .pipeline_config import (
    load_config_file,
    resolve_pipeline_config_path,
)

DEFAULT_FILE_FLOW_STEPS: tuple[str, ...] = (
    "pdf_prepare",
    "llm_fill",
    "standards_review",
    "render_html",
)


def file_flow_root() -> Path:
    """``file_flow`` 包所在目录（本级根，不依赖仓库根）。"""
    return Path(__file__).resolve().parent


def load_merged_pipeline_config(pipeline_path: Path | None) -> dict[str, Any]:
    """
    仅从 ``pipeline.json`` 读取 ``pipeline_config.CONFIG_KEYS`` 中允许的键。

    不与仓库根 ``pipeline.json``、不与 ``defaults_from_environment()`` 做字典合并；
    未找到有效配置文件时返回空字典（各步骤按 pipeline 缺省逻辑或环境变量自行处理）。
    """
    if pipeline_path is None or not pipeline_path.is_file():
        return {}
    return load_config_file(pipeline_path)


def resolve_pipeline_disk_path(workspace: Path, config_path: Path | None) -> Path | None:
    """
    解析管线 JSON 磁盘路径。

    ``config_path`` 为 ``None`` 时使用 ``{workspace}/pipeline.json``（``workspace`` 一般为 ``file_flow`` 目录）。
    """
    if config_path is None:
        cand = workspace / "pipeline.json"
        resolved, hint = resolve_pipeline_config_path(cand)
        if hint:
            _LOG.warning("%s", hint)
        return resolved
    p = Path(config_path)
    cand = p.resolve() if p.is_absolute() else (workspace / p).resolve()
    resolved, hint = resolve_pipeline_config_path(cand)
    if hint:
        _LOG.warning("%s", hint)
    return resolved


def _resolve_path(raw: Path | str, workspace: Path, cwd: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    hit = (cwd / p).resolve()
    if hit.exists():
        return hit
    return (workspace / p).resolve()


def _out_dir_from_merged(merged: dict[str, Any], workspace: Path, cwd: Path) -> Path:
    mo = merged.get("file_flow_out_dir")
    if isinstance(mo, str) and mo.strip():
        return _resolve_path(mo.strip(), workspace, cwd)
    return _resolve_path("out", workspace, cwd)


def parse_file_flow_steps(merged: dict[str, Any]) -> list[str]:
    raw = merged.get("file_flow_steps")
    if isinstance(raw, list) and raw:
        steps = [str(x).strip() for x in raw if str(x).strip()]
        if steps:
            return steps
    return list(DEFAULT_FILE_FLOW_STEPS)


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def file_flow_auto_batch_on(merged: dict[str, Any]) -> bool:
    """未配置 ``file_flow_auto_batch`` 时默认开启，便于一键编排。"""
    if "file_flow_auto_batch" not in merged:
        return True
    return _truthy(merged.get("file_flow_auto_batch"))


def _single_glob_hit(out_dir: Path, pattern: str) -> Path | None:
    xs = sorted(out_dir.glob(pattern))
    return xs[0] if len(xs) == 1 else None


def _step_kw(step: str, step_overrides: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if not step_overrides:
        return {}
    got = step_overrides.get(step)
    return dict(got) if isinstance(got, dict) else {}


def run_file_flow(
    workspace: Path | None = None,
    config_path: Path | None = None,
    cwd: Path | None = None,
    merged: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    step_overrides: dict[str, dict[str, Any]] | None = None,
) -> int:
    """
    按 ``merged`` 中的 ``file_flow_steps`` 依次执行 file_flow 各子步骤。

    ``step_overrides`` 示例：``{"pdf_prepare": {"llm_extract": True}, "llm_fill": {"log_level": "DEBUG"}}``
    """
    ws = workspace or file_flow_root()
    cwd = cwd or Path.cwd()
    from .llm_openai import configure_logging  # noqa: PLC0415
    from .step_dotenv import ensure_step_dotenv_loaded  # noqa: PLC0415

    configure_logging()
    _LOG.info("编排启动 | workspace=%s | cwd=%s", ws.resolve(), cwd.resolve())

    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(ws)
    if dotenv_missing:
        _LOG.warning(
            "未安装 python-dotenv，已跳过 .env；请 pip install python-dotenv（requirements.txt 已列出）"
        )
    elif env_loaded:
        for p in env_loaded:
            _LOG.info("已加载环境文件: %s", p)
    else:
        _LOG.info(
            "未从 %s 及其上一级目录找到任何 .env / 环节变量.env（可将 LLM_* 写在仓库根 .env 或本目录 .env）",
            ws.resolve(),
        )

    disk = resolve_pipeline_disk_path(ws, config_path)
    if merged is None:
        merged = load_merged_pipeline_config(disk if disk is not None and disk.is_file() else None)

    if disk is not None and disk.is_file():
        _LOG.info("管线配置文件: %s", disk.resolve())
    else:
        _LOG.warning(
            "未找到管线 JSON，merged 为空字典。期望路径: %s 或传入 --config",
            (ws / "pipeline.json").resolve(),
        )

    steps = parse_file_flow_steps(merged)
    auto_batch = file_flow_auto_batch_on(merged)
    out_dir = _out_dir_from_merged(merged, ws, cwd)
    sw, sa, sr = file_flow_stem_suffixes(merged)

    _LOG.info("将执行步骤: %s", steps)
    _LOG.info(
        "运行参数 | dry_run=%s | file_flow_auto_batch=%s | out_dir=%s | glob: work=%s answered=%s review=%s",
        dry_run,
        auto_batch,
        out_dir.resolve(),
        work_glob_pattern(merged),
        answered_glob_pattern(merged),
        review_glob_pattern(merged),
    )
    _LOG.info(
        "路径后缀 | work=%r answered=%r review=%r | file_flow_llm_extract=%s",
        sw,
        sa,
        sr,
        merged.get("file_flow_llm_extract"),
    )

    for step in steps:
        if step == "pdf_prepare":
            from . import pdf_prepare  # noqa: PLC0415

            _LOG.info("--- 开始步骤: pdf_prepare ---")
            kw = _step_kw("pdf_prepare", step_overrides)
            dr = bool(kw.pop("dry_run", dry_run))
            code = pdf_prepare.run_pdf_prepare(
                merged,
                workspace=ws,
                cwd=cwd,
                pdf_dir=kw.pop("pdf_dir", None),
                schema=kw.pop("schema", None),
                out_dir=kw.pop("out_dir", None),
                llm_extract=kw.pop("llm_extract", None),
                dry_run=dr,
                log_level=kw.pop("log_level", None),
                log_file=kw.pop("log_file", None),
            )
            if code != 0:
                _LOG.error("步骤 pdf_prepare 失败 exit_code=%s", code)
                return code
            _LOG.info("步骤 pdf_prepare 完成 exit_code=0")
            continue

        if step == "llm_fill":
            from . import llm_fill  # noqa: PLC0415

            _LOG.info("--- 开始步骤: llm_fill | auto_batch=%s ---", auto_batch)
            kw = _step_kw("llm_fill", step_overrides)
            dr = bool(kw.pop("dry_run", dry_run))
            wg = work_glob_pattern(merged)
            if auto_batch:
                work_files = sorted(out_dir.glob(wg))
                if not work_files:
                    sw = file_flow_stem_suffixes(merged)[0]
                    _LOG.error(
                        "llm_fill 批量：目录 %s 内无匹配 %s 的文件（请检查 file_flow_out_dir 与 file_flow_suffix_work）",
                        out_dir.resolve(),
                        wg,
                    )
                    return 1
                _LOG.info("llm_fill 批量：共 %s 个输入文件，glob=%s", len(work_files), wg)
                for wf in work_files:
                    answered = wf.with_name(replace_work_json_name_with_answered(wf.name, merged))
                    _LOG.info("llm_fill: %s -> %s", wf.name, answered.name)
                    code = llm_fill.run_llm_fill(
                        merged,
                        workspace=ws,
                        cwd=cwd,
                        input_path=wf,
                        output_path=answered,
                        dry_run=dr,
                        log_level=kw.get("log_level"),
                        log_file=kw.get("log_file"),
                    )
                    if code != 0:
                        _LOG.error("llm_fill 处理 %s 失败 exit_code=%s", wf.name, code)
                        return code
                _LOG.info("步骤 llm_fill 完成（批量）")
                continue

            inp = kw.pop("input_path", kw.pop("input", None))
            outp = kw.pop("output_path", kw.pop("output", None))
            if inp is None:
                mi = merged.get("file_flow_llm_input")
                if isinstance(mi, str) and mi.strip():
                    inp = Path(mi.strip())
                else:
                    hit = _single_glob_hit(out_dir, wg)
                    if hit is None:
                        _LOG.error(
                            "llm_fill：请配置 file_flow_llm_input，或开启 file_flow_auto_batch，"
                            "或保证 %s 内恰好一个匹配 glob=%s 的文件",
                            out_dir.resolve(),
                            wg,
                        )
                        return 1
                    inp = hit
                    _LOG.info("llm_fill 自动发现单文件输入: %s", inp.resolve())
            if outp is None:
                mo = merged.get("file_flow_llm_output")
                if isinstance(mo, str) and mo.strip():
                    outp = Path(mo.strip())
                else:
                    _, sa, _ = file_flow_stem_suffixes(merged)
                    base = stem_base_from_stage_stem(Path(inp).stem, merged)
                    outp = Path(inp).with_name(f"{base}{sa}.json")
            _LOG.info("llm_fill 单文件模式 输入=%s 输出=%s", inp.resolve(), Path(outp).resolve())
            code = llm_fill.run_llm_fill(
                merged,
                workspace=ws,
                cwd=cwd,
                input_path=inp,
                output_path=outp,
                dry_run=dr,
                log_level=kw.pop("log_level", None),
                log_file=kw.pop("log_file", None),
            )
            if code != 0:
                _LOG.error("步骤 llm_fill 失败 exit_code=%s", code)
                return code
            _LOG.info("步骤 llm_fill 完成 exit_code=0")
            continue

        if step == "standards_review":
            from . import standards_llm_review  # noqa: PLC0415

            _LOG.info("--- 开始步骤: standards_review | auto_batch=%s ---", auto_batch)
            kw = _step_kw("standards_review", step_overrides)
            dr = bool(kw.pop("dry_run", dry_run))
            std_path = kw.pop("standards", None)
            if std_path:
                _LOG.info("standards_review 使用 pipeline/入参 standards: %s", std_path)
            if auto_batch:
                ag = answered_glob_pattern(merged)
                cand = sorted(out_dir.glob(ag))
                if not cand:
                    cand = sorted(out_dir.glob(work_glob_pattern(merged)))
                if not cand:
                    _LOG.error(
                        "standards_review 批量：目录 %s 内无匹配 %s 或 %s 的文件",
                        out_dir.resolve(),
                        ag,
                        work_glob_pattern(merged),
                    )
                    return 1
                _LOG.info("standards_review 批量：共 %s 个输入", len(cand))
                for wf in cand:
                    base = stem_base_from_stage_stem(wf.stem, merged)
                    out_rev = wf.with_name(review_json_filename_for_base(base, merged))
                    _LOG.info("standards_review: %s -> %s", wf.name, out_rev.name)
                    code = standards_llm_review.run_standards_review(
                        merged,
                        workspace=ws,
                        cwd=cwd,
                        work_input=wf,
                        standards_path=std_path,
                        output_path=out_rev,
                        dry_run=dr,
                        log_level=kw.get("log_level"),
                        log_file=kw.get("log_file"),
                    )
                    if code != 0:
                        _LOG.error("standards_review 处理 %s 失败 exit_code=%s", wf.name, code)
                        return code
                _LOG.info("步骤 standards_review 完成（批量）")
                continue

            win = kw.pop("work_input", kw.pop("input", None))
            sout = kw.pop("output_path", kw.pop("output", None))
            if win is None:
                for key in ("file_flow_review_work_input", "file_flow_llm_output", "file_flow_llm_input"):
                    v = merged.get(key)
                    if isinstance(v, str) and v.strip():
                        win = Path(v.strip())
                        break
                if win is None:
                    hit = _single_glob_hit(out_dir, answered_glob_pattern(merged)) or _single_glob_hit(
                        out_dir, work_glob_pattern(merged)
                    )
                    if hit is None:
                        _LOG.error(
                            "standards_review：请配置 file_flow_review_work_input 等键，"
                            "或保证 %s 内恰好一个 answered/work 匹配（glob answered=%s work=%s）",
                            out_dir.resolve(),
                            answered_glob_pattern(merged),
                            work_glob_pattern(merged),
                        )
                        return 1
                    win = hit
                    _LOG.info("standards_review 自动发现输入: %s", win.resolve())
            if sout is None:
                mo = merged.get("file_flow_review_result_output")
                if isinstance(mo, str) and mo.strip():
                    sout = Path(mo.strip())
                else:
                    base = stem_base_from_stage_stem(Path(win).stem, merged)
                    sout = Path(win).with_name(review_json_filename_for_base(base, merged))
            _LOG.info(
                "standards_review 单文件模式 输入=%s 输出=%s",
                Path(win).resolve(),
                Path(sout).resolve(),
            )
            code = standards_llm_review.run_standards_review(
                merged,
                workspace=ws,
                cwd=cwd,
                work_input=win,
                standards_path=std_path,
                output_path=sout,
                dry_run=dr,
                log_level=kw.pop("log_level", None),
                log_file=kw.pop("log_file", None),
            )
            if code != 0:
                _LOG.error("步骤 standards_review 失败 exit_code=%s", code)
                return code
            _LOG.info("步骤 standards_review 完成 exit_code=0")
            continue

        if step == "render_html":
            from . import render_html  # noqa: PLC0415

            _LOG.info("--- 开始步骤: render_html | auto_batch=%s ---", auto_batch)
            kw = _step_kw("render_html", step_overrides)
            if auto_batch:
                rg = review_glob_pattern(merged)
                rev_files = sorted(out_dir.glob(rg))
                if rev_files:
                    batch_inputs = rev_files
                else:
                    ans = sorted(out_dir.glob(answered_glob_pattern(merged)))
                    batch_inputs = ans if ans else sorted(out_dir.glob(work_glob_pattern(merged)))
                if not batch_inputs:
                    _LOG.error(
                        "render_html 批量：目录 %s 内无 review/answered/work 匹配（glob review=%s answered=%s work=%s）",
                        out_dir.resolve(),
                        rg,
                        answered_glob_pattern(merged),
                        work_glob_pattern(merged),
                    )
                    return 1
                _LOG.info("render_html 批量：共 %s 个输入", len(batch_inputs))
                for af in batch_inputs:
                    html_out = af.with_suffix(".html")
                    _LOG.info("render_html: %s -> %s", af.name, html_out.name)
                    code = render_html.run_render_html(
                        merged,
                        workspace=ws,
                        cwd=cwd,
                        input_path=af,
                        output_path=html_out,
                        title=kw.get("title"),
                    )
                    if code != 0:
                        _LOG.error("render_html 处理 %s 失败 exit_code=%s", af.name, code)
                        return code
                _LOG.info("步骤 render_html 完成（批量）")
                continue

            ip = kw.pop("input_path", kw.pop("input", None))
            op = kw.pop("output_path", kw.pop("output", None))
            if ip is None:
                for key in ("file_flow_render_html_input", "file_flow_review_result_output", "file_flow_llm_output"):
                    v = merged.get(key)
                    if isinstance(v, str) and v.strip():
                        ip = Path(v.strip())
                        break
                if ip is None:
                    hit = (
                        _single_glob_hit(out_dir, review_glob_pattern(merged))
                        or _single_glob_hit(out_dir, answered_glob_pattern(merged))
                        or _single_glob_hit(out_dir, work_glob_pattern(merged))
                    )
                    if hit is None:
                        _LOG.error(
                            "render_html：请配置 file_flow_render_html_input，或保证 %s 内恰好一个 JSON 匹配 "
                            "review/answered/work 的 glob",
                            out_dir.resolve(),
                        )
                        return 1
                    ip = hit
                    _LOG.info("render_html 自动发现输入: %s", ip.resolve())
            if op is None:
                mo = merged.get("file_flow_render_html_output")
                if isinstance(mo, str) and mo.strip():
                    op = Path(mo.strip())
                else:
                    op = Path(ip).with_suffix(".html")
            _LOG.info("render_html 单文件模式 输入=%s 输出=%s", Path(ip).resolve(), Path(op).resolve())
            code = render_html.run_render_html(
                merged,
                workspace=ws,
                cwd=cwd,
                input_path=ip,
                output_path=op,
                title=kw.pop("title", None),
            )
            if code != 0:
                _LOG.error("步骤 render_html 失败 exit_code=%s", code)
                return code
            _LOG.info("步骤 render_html 完成 exit_code=0")
            continue

        _LOG.warning("未知步骤，已跳过: %s", step)
    _LOG.info("全部步骤完成，exit_code=0")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按 pipeline.json 编排执行 file_flow 全流程")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="管线 JSON；默认使用 file_flow 目录下的 pipeline.json",
    )
    ap.add_argument("--dry-run", action="store_true", help="各 LLM 步骤尽量 dry-run")
    ns = ap.parse_args(argv)
    return run_file_flow(workspace=file_flow_root(), config_path=ns.config, dry_run=ns.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
