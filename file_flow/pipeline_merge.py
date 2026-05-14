#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_flow 管线编排：合并 ``pipeline.json``（默认优先 ``file_flow/pipeline.json``）与环境默认值，并按配置顺序以 **Python 调用** 各步骤，
无需在终端分别敲 ``pdf_prepare`` / ``llm_fill`` / ``render_html``。

- ``load_merged_pipeline_config``：读盘合并（``pipeline.json`` 为底，环境覆盖同名键）。
- ``run_file_flow``：根据 ``file_flow_steps`` 依次执行；支持 ``file_flow_auto_batch`` 对目录内
  多个 ``*_work.json`` / ``*_answered.json`` 批量 llm_fill、``*_review.json`` 批量 render。

用法::

    python file_flow/pipeline_merge.py
    python file_flow/pipeline_merge.py --config my_pipeline.json --dry-run

或在代码中::

    from file_flow.pipeline_merge import run_file_flow
    run_file_flow()

编排键（写在 ``pipeline.json``）::

    file_flow_steps: ["pdf_prepare", "llm_fill", "standards_review", "render_html"]
    file_flow_auto_batch: true
    file_flow_llm_extract: true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

DEFAULT_FILE_FLOW_STEPS: tuple[str, ...] = (
    "pdf_prepare",
    "llm_fill",
    "standards_review",
    "render_html",
)


def repo_root() -> Path:
    """``file_flow`` 的上一级目录（仓库根）。"""
    return Path(__file__).resolve().parent.parent


def ensure_repo_importable() -> Path:
    """将仓库根加入 ``sys.path``，以便 ``import pipeline_config``。"""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_merged_pipeline_config(pipeline_path: Path | None) -> dict[str, Any]:
    """
    合并：先读 ``pipeline.json`` 中允许的键，再叠 ``defaults_from_environment()``（后者覆盖同名键）。
    """
    ensure_repo_importable()
    from pipeline_config import (  # noqa: PLC0415
        defaults_from_environment,
        load_config_file,
        merge_defaults,
    )

    if pipeline_path is None or not pipeline_path.is_file():
        return defaults_from_environment()
    return merge_defaults(load_config_file(pipeline_path), defaults_from_environment())


def resolve_pipeline_disk_path(workspace: Path, config_path: Path | None) -> Path | None:
    """
    解析管线 JSON 磁盘路径。

    ``config_path`` 为 ``None`` 时：**优先** ``{workspace}/file_flow/pipeline.json``（与 ``file_flow`` 内脚本同级），
    不存在时再尝试 ``{workspace}/pipeline.json``。
    """
    ensure_repo_importable()
    from pipeline_config import resolve_pipeline_config_path  # noqa: PLC0415

    if config_path is None:
        ff = workspace / "file_flow" / "pipeline.json"
        if ff.is_file():
            resolved, hint = resolve_pipeline_config_path(ff)
            if hint:
                print(hint, file=sys.stderr)
            return resolved
        resolved, hint = resolve_pipeline_config_path(workspace / "pipeline.json")
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
    return _resolve_path("file_flow/out", workspace, cwd)


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
    ws = workspace or repo_root()
    cwd = cwd or Path.cwd()
    from file_flow.step_dotenv import ensure_step_dotenv_loaded  # noqa: PLC0415

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

    from file_flow.llm_openai import configure_logging  # noqa: PLC0415

    configure_logging()

    steps = parse_file_flow_steps(merged)
    print(f"[编排] 将执行步骤: {steps}")

    auto_batch = _truthy(merged.get("file_flow_auto_batch"))
    out_dir = _out_dir_from_merged(merged, ws, cwd)

    for step in steps:
        if step == "pdf_prepare":
            from file_flow import pdf_prepare  # noqa: PLC0415

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
            from file_flow import llm_fill  # noqa: PLC0415

            kw = _step_kw("llm_fill", step_overrides)
            dr = bool(kw.pop("dry_run", dry_run))
            if auto_batch:
                work_files = sorted(out_dir.glob("*_work.json"))
                if not work_files:
                    print(f"[编排] llm_fill 批量：目录内无 *_work.json: {out_dir}", file=sys.stderr)
                    return 1
                for wf in work_files:
                    answered = wf.with_name(wf.name.replace("_work.json", "_answered.json", 1))
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

            code = llm_fill.run_llm_fill(
                merged,
                workspace=ws,
                cwd=cwd,
                input_path=kw.pop("input_path", kw.pop("input", None)),
                output_path=kw.pop("output_path", kw.pop("output", None)),
                dry_run=dr,
                log_level=kw.pop("log_level", None),
                log_file=kw.pop("log_file", None),
            )
            if code != 0:
                return code
            continue

        if step == "standards_review":
            from file_flow import standards_llm_review  # noqa: PLC0415

            kw = _step_kw("standards_review", step_overrides)
            dr = bool(kw.pop("dry_run", dry_run))
            if auto_batch:
                cand = sorted(out_dir.glob("*_answered.json"))
                if not cand:
                    cand = sorted(out_dir.glob("*_work.json"))
                if not cand:
                    print(
                        f"[编排] standards_review 批量：目录内无 *_answered.json 或 *_work.json: {out_dir}",
                        file=sys.stderr,
                    )
                    return 1
                for wf in cand:
                    stem = wf.stem
                    if stem.endswith("_answered"):
                        base = stem[: -len("_answered")]
                    elif stem.endswith("_work"):
                        base = stem[: -len("_work")]
                    else:
                        base = stem
                    out_rev = wf.with_name(f"{base}_review.json")
                    print(f"[编排] standards_review 批量: {wf.name} -> {out_rev.name}")
                    code = standards_llm_review.run_standards_review(
                        merged,
                        workspace=ws,
                        cwd=cwd,
                        work_input=wf,
                        standards_path=kw.pop("standards", None),
                        output_path=out_rev,
                        dry_run=dr,
                        log_level=kw.get("log_level"),
                        log_file=kw.get("log_file"),
                    )
                    if code != 0:
                        return code
                continue

            code = standards_llm_review.run_standards_review(
                merged,
                workspace=ws,
                cwd=cwd,
                work_input=kw.pop("work_input", kw.pop("input", None)),
                standards_path=kw.pop("standards", None),
                output_path=kw.pop("output_path", kw.pop("output", None)),
                dry_run=dr,
                log_level=kw.pop("log_level", None),
                log_file=kw.pop("log_file", None),
            )
            if code != 0:
                return code
            continue

        if step == "render_html":
            from file_flow import render_html  # noqa: PLC0415

            kw = _step_kw("render_html", step_overrides)
            if auto_batch:
                rev_files = sorted(out_dir.glob("*_review.json"))
                if rev_files:
                    batch_inputs = rev_files
                else:
                    ans = sorted(out_dir.glob("*_answered.json"))
                    batch_inputs = ans if ans else sorted(out_dir.glob("*_work.json"))
                if not batch_inputs:
                    print(
                        f"[编排] render_html 批量：目录内无 *_review.json / *_answered.json / *_work.json: {out_dir}",
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

            code = render_html.run_render_html(
                merged,
                workspace=ws,
                cwd=cwd,
                input_path=kw.pop("input_path", kw.pop("input", None)),
                output_path=kw.pop("output_path", kw.pop("output", None)),
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
        help="管线 JSON；默认优先 file_flow/pipeline.json，否则项目根 pipeline.json",
    )
    ap.add_argument("--dry-run", action="store_true", help="各 LLM 步骤尽量 dry-run")
    ns = ap.parse_args(argv)
    return run_file_flow(workspace=repo_root(), config_path=ns.config, dry_run=ns.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
