#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_flow 管线编排：合并 ``pipeline.json``（默认使用 ``file_flow`` 包目录下的 ``pipeline.json``）与环境默认值，并按配置顺序以 **Python 调用** 各步骤，
无需在终端分别敲 ``pdf_prepare`` / ``llm_fill`` / ``render_html``。

- ``load_merged_pipeline_config``：读盘合并（``pipeline.json`` 为底，环境覆盖同名键）。
- ``run_file_flow``：根据 ``file_flow_steps`` 依次执行；``file_flow_auto_batch`` 未出现在配置中时**默认开启**，
  对 ``file_flow_out_dir`` 下符合后缀规则的 JSON 批量执行 llm_fill、standards_review、render_html。
  产出文件名后缀由 ``file_flow_suffix_work`` / ``file_flow_suffix_answered`` / ``file_flow_suffix_review`` 控制（默认 ``_work`` / ``_answered`` / ``_review``）。

用法（在包含 ``file_flow`` 包的上级目录执行）::

    python -m file_flow.pipeline_merge
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
import sys
from pathlib import Path
from typing import Any

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
    defaults_from_environment,
    load_config_file,
    merge_defaults,
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
    合并：先读 ``pipeline.json`` 中允许的键，再叠 ``defaults_from_environment()``（后者覆盖同名键）。
    """
    if pipeline_path is None or not pipeline_path.is_file():
        return defaults_from_environment()
    return merge_defaults(load_config_file(pipeline_path), defaults_from_environment())


def resolve_pipeline_disk_path(workspace: Path, config_path: Path | None) -> Path | None:
    """
    解析管线 JSON 磁盘路径。

    ``config_path`` 为 ``None`` 时使用 ``{workspace}/pipeline.json``（``workspace`` 一般为 ``file_flow`` 目录）。
    """
    if config_path is None:
        cand = workspace / "pipeline.json"
        resolved, hint = resolve_pipeline_config_path(cand)
        if hint:
            print(hint, file=sys.stderr)
        return resolved
    p = Path(config_path)
    cand = p.resolve() if p.is_absolute() else (workspace / p).resolve()
    resolved, hint = resolve_pipeline_config_path(cand)
    if hint:
        print(hint, file=sys.stderr)
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
    from .step_dotenv import ensure_step_dotenv_loaded  # noqa: PLC0415

    env_loaded, dotenv_missing = ensure_step_dotenv_loaded(ws)
    if dotenv_missing:
        print(
            "警告: 未安装 python-dotenv，已跳过 .env；请 pip install python-dotenv",
            file=sys.stderr,
        )
    elif env_loaded:
        print(f"已加载环境文件: {len(env_loaded)} 个")

    disk = resolve_pipeline_disk_path(ws, config_path)
    if merged is None:
        merged = load_merged_pipeline_config(disk if disk is not None and disk.is_file() else None)

    from .llm_openai import configure_logging  # noqa: PLC0415

    configure_logging()

    steps = parse_file_flow_steps(merged)
    print(f"[编排] 将执行步骤: {steps}")

    auto_batch = file_flow_auto_batch_on(merged)
    out_dir = _out_dir_from_merged(merged, ws, cwd)

    for step in steps:
        if step == "pdf_prepare":
            from . import pdf_prepare  # noqa: PLC0415

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
                return code
            continue

        if step == "llm_fill":
            from . import llm_fill  # noqa: PLC0415

            kw = _step_kw("llm_fill", step_overrides)
            dr = bool(kw.pop("dry_run", dry_run))
            wg = work_glob_pattern(merged)
            if auto_batch:
                work_files = sorted(out_dir.glob(wg))
                if not work_files:
                    sw = file_flow_stem_suffixes(merged)[0]
                    print(f"[编排] llm_fill 批量：目录内无 *{sw}.json: {out_dir}", file=sys.stderr)
                    return 1
                for wf in work_files:
                    answered = wf.with_name(replace_work_json_name_with_answered(wf.name, merged))
                    print(f"[编排] llm_fill 批量: {wf.name} -> {answered.name}")
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
                        return code
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
                        sw = file_flow_stem_suffixes(merged)[0]
                        print(
                            f"[编排] llm_fill：请配置 file_flow_llm_input，或开启批量，"
                            f"或保证 {out_dir} 内**恰好一个** *{sw}.json",
                            file=sys.stderr,
                        )
                        return 1
                    inp = hit
            if outp is None:
                mo = merged.get("file_flow_llm_output")
                if isinstance(mo, str) and mo.strip():
                    outp = Path(mo.strip())
                else:
                    _, sa, _ = file_flow_stem_suffixes(merged)
                    base = stem_base_from_stage_stem(Path(inp).stem, merged)
                    outp = Path(inp).with_name(f"{base}{sa}.json")
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
                return code
            continue

        if step == "standards_review":
            from . import standards_llm_review  # noqa: PLC0415

            kw = _step_kw("standards_review", step_overrides)
            dr = bool(kw.pop("dry_run", dry_run))
            std_path = kw.pop("standards", None)
            if auto_batch:
                ag = answered_glob_pattern(merged)
                cand = sorted(out_dir.glob(ag))
                if not cand:
                    cand = sorted(out_dir.glob(work_glob_pattern(merged)))
                if not cand:
                    sw, sa, _ = file_flow_stem_suffixes(merged)
                    print(
                        f"[编排] standards_review 批量：目录内无 *{sa}.json 或 *{sw}.json: {out_dir}",
                        file=sys.stderr,
                    )
                    return 1
                for wf in cand:
                    base = stem_base_from_stage_stem(wf.stem, merged)
                    out_rev = wf.with_name(review_json_filename_for_base(base, merged))
                    print(f"[编排] standards_review 批量: {wf.name} -> {out_rev.name}")
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
                        return code
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
                        sw, sa, _ = file_flow_stem_suffixes(merged)
                        print(
                            f"[编排] standards_review：请配置输入键，或保证 {out_dir} 内**恰好一个**"
                            f" *{sa}.json 或 *{sw}.json",
                            file=sys.stderr,
                        )
                        return 1
                    win = hit
            if sout is None:
                mo = merged.get("file_flow_review_result_output")
                if isinstance(mo, str) and mo.strip():
                    sout = Path(mo.strip())
                else:
                    base = stem_base_from_stage_stem(Path(win).stem, merged)
                    sout = Path(win).with_name(review_json_filename_for_base(base, merged))
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
                return code
            continue

        if step == "render_html":
            from . import render_html  # noqa: PLC0415

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
                    sw, sa, sr = file_flow_stem_suffixes(merged)
                    print(
                        f"[编排] render_html 批量：目录内无 *{sr}.json / *{sa}.json / *{sw}.json: {out_dir}",
                        file=sys.stderr,
                    )
                    return 1
                for af in batch_inputs:
                    html_out = af.with_suffix(".html")
                    print(f"[编排] render_html 批量: {af.name} -> {html_out.name}")
                    code = render_html.run_render_html(
                        merged,
                        workspace=ws,
                        cwd=cwd,
                        input_path=af,
                        output_path=html_out,
                        title=kw.get("title"),
                    )
                    if code != 0:
                        return code
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
                        sw, sa, sr = file_flow_stem_suffixes(merged)
                        print(
                            f"[编排] render_html：请配置 file_flow_render_html_input，或保证 {out_dir} 内**恰好一个**"
                            f" *{sr}.json / *{sa}.json / *{sw}.json",
                            file=sys.stderr,
                        )
                        return 1
                    ip = hit
            if op is None:
                mo = merged.get("file_flow_render_html_output")
                if isinstance(mo, str) and mo.strip():
                    op = Path(mo.strip())
                else:
                    op = Path(ip).with_suffix(".html")
            code = render_html.run_render_html(
                merged,
                workspace=ws,
                cwd=cwd,
                input_path=ip,
                output_path=op,
                title=kw.pop("title", None),
            )
            if code != 0:
                return code
            continue

        print(f"[编排] 未知步骤，已跳过: {step}", file=sys.stderr)

    print("[编排] 全部步骤完成")
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
