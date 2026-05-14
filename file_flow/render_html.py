#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件模式第三步（评测视图）：将工作/填答后的 JSON 渲染为单文件 HTML。

每个栏目为左右分栏：左侧为「原文 / content 摘录」，右侧为「评审要点」与「填答 answer」，两侧顶部对齐，
长文随整页滚动（不在栏内做固定视口截断）。

字段左侧展示 ``content`` 摘录；右侧「评审要点」来自 ``related_review_items`` 或 ``description``；
「填答」来自 ``answer``。缺少填答时显示「（未填答）」。

用法（在包含 ``file_flow`` 包的上级目录执行）::

    python -m file_flow.render_html -i out/某案_answered.json -o out/某案_review.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

_FILE_FLOW_DIR = Path(__file__).resolve().parent


def _field_review_prompt(field_obj: dict[str, Any]) -> str:
    items = field_obj.get("related_review_items")
    if isinstance(items, list):
        lines = [str(x).strip() for x in items if str(x).strip()]
        if lines:
            return "\n".join(lines)
    d = field_obj.get("description")
    if isinstance(d, str) and d.strip():
        return d.strip()
    if d is not None:
        return str(d).strip()
    return ""


def _field_content(field_obj: dict[str, Any]) -> str:
    co = field_obj.get("content")
    if isinstance(co, str):
        return co
    if co is None:
        return ""
    return str(co)


def _field_answer(field_obj: dict[str, Any]) -> str:
    a = field_obj.get("answer")
    if isinstance(a, str) and a.strip():
        return a.strip()
    return "（未填答）"


def _html_escape_field(val: Any) -> str:
    if val is None:
        return ""
    return html.escape(str(val))


def _render_standards_review_section(data: dict[str, Any]) -> str:
    """若有 ``standards_review``，渲染清单评审块（供最终可视化）。"""
    sr = data.get("standards_review")
    if not isinstance(sr, dict):
        return ""
    items = sr.get("items")
    if not isinstance(items, list) or not items:
        return ""
    sp = sr.get("standards_path", "")
    gen = sr.get("generated_at", "")
    meta_bits = []
    if sp:
        meta_bits.append(f"标准文件：<code>{_html_escape_field(sp)}</code>")
    if gen:
        meta_bits.append(f"生成时间：<code>{_html_escape_field(gen)}</code>")
    meta_html = f"<p class='sr-meta'>{' &nbsp;|&nbsp; '.join(meta_bits)}</p>" if meta_bits else ""

    cards: list[str] = []
    for idx, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue
        ra = it.get("review_answer")
        ra_s = html.escape(str(ra).strip() if isinstance(ra, str) else (str(ra) if ra is not None else ""))
        cards.append(
            f"<article class='field sr-card'>"
            f"<header class='field-head'><h4>评审清单第 {idx} 项</h4></header>"
            f"<dl class='sr-dl'>"
            f"<dt>类别</dt><dd><pre class='pre-src'>{_html_escape_field(it.get('category'))}</pre></dd>"
            f"<dt>子类</dt><dd><pre class='pre-src'>{_html_escape_field(it.get('subcategory'))}</pre></dd>"
            f"<dt>条目说明</dt><dd><pre class='pre-src'>{_html_escape_field(it.get('content'))}</pre></dd>"
            f"<dt>评审标准（须对照）</dt><dd><pre class='pre-question'>{_html_escape_field(it.get('standard'))}</pre></dd>"
            f"<dt>分值 / 扣分 / 编号</dt><dd>"
            f"{_html_escape_field(it.get('score'))} / {_html_escape_field(it.get('penalty'))} / {_html_escape_field(it.get('number'))}"
            f"</dd>"
            f"<dt>评审结论</dt><dd><pre class='pre-answer'>{ra_s}</pre></dd>"
            f"</dl></article>"
        )
    return (
        f"<section class='doc standards-review'><h2>按 standards 清单评审</h2>"
        f"{meta_html}<div class='fields'>{''.join(cards)}</div></section>"
    )


def render_review_html(data: dict[str, Any], title: str = "案卷评审结果浏览") -> str:
    """生成完整 HTML 文档字符串。"""
    blocks: list[str] = []
    meta = data.get("_file_flow_meta")
    if isinstance(meta, dict):
        meta_rows = "".join(
            f"<tr><th>{html.escape(str(k))}</th><td><code>{html.escape(str(v))}</code></td></tr>"
            for k, v in meta.items()
        )
        blocks.append(f"<section class='meta'><h2>元信息</h2><table>{meta_rows}</table></section>")

    doc_types = data.get("document_types")
    if isinstance(doc_types, list) and doc_types:
        for doc in doc_types:
            if not isinstance(doc, dict):
                continue
            doc_title = str(doc.get("document_name") or doc.get("document_type") or "文书").strip()
            must = doc.get("required", "")
            must_s = html.escape(str(must)) if must not in (None, "") else ""
            fields_list = doc.get("fields")
            if not isinstance(fields_list, list):
                continue
            field_cards: list[str] = []
            for field_obj in fields_list:
                if not isinstance(field_obj, dict):
                    continue
                fn_raw = field_obj.get("field_name") or "（未命名字段）"
                fn = html.escape(str(fn_raw))
                desc_raw = field_obj.get("description")
                desc_html = ""
                if isinstance(desc_raw, str) and desc_raw.strip():
                    desc_html = (
                        "<p class='field-desc'><strong>字段说明</strong>："
                        f"{html.escape(desc_raw.strip())}</p>"
                    )
                elif desc_raw is not None and str(desc_raw).strip():
                    desc_html = (
                        "<p class='field-desc'><strong>字段说明</strong>："
                        f"{html.escape(str(desc_raw).strip())}</p>"
                    )
                q = html.escape(_field_review_prompt(field_obj))
                c = html.escape(_field_content(field_obj))
                a = html.escape(_field_answer(field_obj))
                dt_note = field_obj.get("data_type")
                dt_s = ""
                if dt_note is not None and str(dt_note).strip():
                    dt_s = f"<p class='hw'><strong>数据类型</strong>：{html.escape(str(dt_note).strip())}</p>"
                field_cards.append(
                    f"<article class='field'>"
                    f"<header class='field-head'><h4>{fn}</h4>{desc_html}{dt_s}</header>"
                    f"<div class='field-split'>"
                    f"<div class='col-source'><h5>原文（抽取内容）</h5><pre class='pre-source'>{c}</pre></div>"
                    f"<div class='col-qa'>"
                    f"<div class='qa-block'><h5>评审要点</h5><pre class='pre-question'>{q}</pre></div>"
                    f"<div class='qa-block qa-answer'><h5>填答</h5><pre class='pre-answer'>{a}</pre></div>"
                    f"</div></div></article>"
                )
            sn = html.escape(doc_title)
            badge = f"<span class='badge'>{must_s}</span>" if must_s else ""
            blocks.append(
                f"<section class='doc'><h3>{sn} {badge}</h3><div class='fields'>{''.join(field_cards)}</div></section>"
            )

    sr_html = _render_standards_review_section(data)
    if sr_html:
        blocks.append(sr_html)

    body = "\n".join(blocks)
    esc_title = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{esc_title}</title>
  <style>
    body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 1rem 1.5rem; background: #f6f7f9; color: #1a1a1a; }}
    h1 {{ font-size: 1.25rem; border-bottom: 2px solid #2563eb; padding-bottom: 0.35rem; }}
    h2 {{ font-size: 1.05rem; margin-top: 1.25rem; }}
    h3 {{ font-size: 1.1rem; margin: 0 0 0.75rem 0; color: #1e3a8a; }}
    h4 {{ margin: 0; font-size: 1rem; }}
    .field-desc {{ margin: 0.35rem 0 0 0; font-size: 0.88rem; color: #334155; line-height: 1.45; }}
    h5 {{ margin: 0 0 0.35rem 0; font-size: 0.8rem; color: #64748b; letter-spacing: 0.02em; }}
    section.doc {{ background: #fff; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;
      box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    section.meta {{ background: #eff6ff; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.9rem; }}
    section.meta table {{ border-collapse: collapse; width: 100%; }}
    section.meta th {{ text-align: left; padding: 0.25rem 0.5rem; width: 8rem; color: #334155; }}
    section.meta td {{ padding: 0.25rem; }}
    .badge {{ font-size: 0.75rem; background: #e0e7ff; color: #3730a3; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.35rem; }}
    article.field {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; background: #fafafa; }}
    article.field:last-child {{ margin-bottom: 0; }}
    .field-head {{ margin-bottom: 0.65rem; }}
    .field-head h4 {{ display: inline-block; vertical-align: middle; }}
    /* 左右分栏：顶部对齐，随页面整体滚动（不设栏内 max-height 截断） */
    .field-split {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 1rem 1.25rem;
      align-items: start;
    }}
    @media (max-width: 900px) {{
      .field-split {{ grid-template-columns: 1fr; }}
    }}
    .col-source, .col-qa {{ min-width: 0; }}
    .col-qa {{ display: flex; flex-direction: column; gap: 0.75rem; }}
    .qa-block {{ min-width: 0; }}
    article.field pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 0.88rem;
      line-height: 1.5;
      border-radius: 6px;
      padding: 0.55rem 0.7rem;
      overflow: visible;
    }}
    .pre-source {{
      background: #fff;
      border: 1px solid #e2e8f0;
    }}
    .pre-question {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
    }}
    .pre-answer {{
      background: #f0fdf4;
      border: 1px solid #86efac;
    }}
    p.hw {{ margin: 0.35rem 0 0 0; font-size: 0.85rem; color: #92400e; }}
    section.standards-review h2 {{ font-size: 1.05rem; color: #0f766e; margin-bottom: 0.5rem; }}
    p.sr-meta {{ font-size: 0.85rem; color: #475569; margin: 0 0 0.75rem 0; }}
    dl.sr-dl {{ margin: 0; display: grid; grid-template-columns: 8rem 1fr; gap: 0.35rem 0.75rem; font-size: 0.88rem; }}
    dl.sr-dl dt {{ color: #64748b; font-weight: 600; }}
    dl.sr-dl dd {{ margin: 0; min-width: 0; }}
    .sr-card pre.pre-src {{ background: #fff; border: 1px solid #e2e8f0; white-space: pre-wrap; word-break: break-word; padding: 0.45rem 0.55rem; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>{esc_title}</h1>
{body}
</body>
</html>
"""


def _resolve_path(raw: Path, cwd: Path, workspace: Path) -> Path:
    if raw.is_absolute():
        return raw.resolve()
    for base in (cwd, workspace):
        hit = (base / raw).resolve()
        if hit.is_file():
            return hit
    return (workspace / raw).resolve()


def run_render_html(
    merged: dict[str, Any],
    *,
    workspace: Path,
    cwd: Path,
    input_path: Path | None = None,
    output_path: Path | None = None,
    title: str | None = None,
) -> int:
    """可编程入口：JSON → HTML。"""
    in_raw = input_path
    if in_raw is None:
        for key in ("file_flow_render_html_input", "file_flow_review_result_output"):
            v = merged.get(key)
            if isinstance(v, str) and v.strip():
                in_raw = Path(v.strip())
                break
    if in_raw is None:
        print(
            "错误: 未指定 render 输入，请设 file_flow_render_html_input / file_flow_review_result_output",
            file=sys.stderr,
        )
        return 1

    out_raw = output_path
    if out_raw is None:
        mo = merged.get("file_flow_render_html_output")
        if isinstance(mo, str) and mo.strip():
            out_raw = Path(mo.strip())
        else:
            p = Path(in_raw)
            out_raw = p.with_suffix(".html")

    title_s = title
    if title_s is None:
        t = merged.get("file_flow_render_title")
        title_s = str(t).strip() if isinstance(t, str) and t.strip() else "案卷评审结果浏览"

    in_path = _resolve_path(Path(in_raw), cwd, workspace)
    out_path = Path(out_raw)
    out_path = out_path.resolve() if out_path.is_absolute() else (cwd / out_path).resolve()

    if not in_path.is_file():
        print(f"错误: 找不到输入: {in_path}", file=sys.stderr)
        return 1
    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取或解析 JSON: {e}", file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print("错误: JSON 根须为对象", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_review_html(data, title=title_s), encoding="utf-8")
    print(f"已写出: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="将评审 JSON 渲染为静态 HTML")
    ap.add_argument("-i", "--input", type=Path, required=True, help="*_review.json 或 *_work.json（含 standards_review 时优先 review）")
    ap.add_argument("-o", "--output", type=Path, required=True, help="输出 .html 路径")
    ap.add_argument("--title", default="案卷评审结果浏览", help="页面标题")
    ns = ap.parse_args(argv)
    return run_render_html(
        {},
        workspace=_FILE_FLOW_DIR,
        cwd=Path.cwd(),
        input_path=ns.input,
        output_path=ns.output,
        title=ns.title if ns.title else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
