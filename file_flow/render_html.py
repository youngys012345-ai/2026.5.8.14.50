#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件模式第三步（评测视图）：将工作/填答后的 JSON 渲染为单文件 HTML。

每个栏目为左右分栏：左侧为「原文/抽取内容」，右侧为「问题」与「回答」，两侧顶部对齐，
长文随整页滚动（不在栏内做固定视口截断）。

若某栏缺少「问题」，则从「要求」列表现场拼接展示；缺少「回答」时显示为「（未填答）」。

用法::

    python file_flow/render_html.py -i file_flow/out/某案_answered.json -o file_flow/out/某案_review.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent


def _requirements_to_question(reqs: Any) -> str:
    if isinstance(reqs, list):
        return "\n".join(str(x).strip() for x in reqs if str(x).strip())
    if reqs is None:
        return ""
    return str(reqs).strip()


def _field_question(field_obj: dict[str, Any]) -> str:
    q = field_obj.get("问题")
    if isinstance(q, str) and q.strip():
        return q.strip()
    return _requirements_to_question(field_obj.get("要求"))


def _field_content(field_obj: dict[str, Any]) -> str:
    c = field_obj.get("内容")
    if isinstance(c, str):
        return c
    if c is None:
        return ""
    return str(c)


def _field_answer(field_obj: dict[str, Any]) -> str:
    a = field_obj.get("回答")
    if isinstance(a, str) and a.strip():
        return a.strip()
    return "（未填答）"


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

    for section_name, block in data.items():
        if section_name.startswith("_"):
            continue
        if not isinstance(block, dict):
            continue
        must = block.get("是否必须", "")
        must_s = html.escape(str(must)) if must else ""
        fields_obj = block.get("字段")
        if not isinstance(fields_obj, dict):
            continue

        field_cards: list[str] = []
        for field_name, field_obj in fields_obj.items():
            if not isinstance(field_obj, dict):
                continue
            fn = html.escape(str(field_name))
            q = html.escape(_field_question(field_obj))
            c = html.escape(_field_content(field_obj))
            a = html.escape(_field_answer(field_obj))
            hw = field_obj.get("是否需要识别手写体")
            hw_note = ""
            if hw is not None and str(hw).strip():
                hw_note = f"<p class='hw'><strong>手写体</strong>：{html.escape(str(hw).strip())}</p>"
            field_cards.append(
                f"<article class='field'>"
                f"<header class='field-head'><h4>{fn}</h4>{hw_note}</header>"
                f"<div class='field-split'>"
                f"<div class='col-source'><h5>原文（抽取内容）</h5><pre class='pre-source'>{c}</pre></div>"
                f"<div class='col-qa'>"
                f"<div class='qa-block'><h5>问题</h5><pre class='pre-question'>{q}</pre></div>"
                f"<div class='qa-block qa-answer'><h5>回答</h5><pre class='pre-answer'>{a}</pre></div>"
                f"</div></div></article>"
            )

        sn = html.escape(str(section_name))
        badge = f"<span class='badge'>{must_s}</span>" if must_s else ""
        blocks.append(
            f"<section class='doc'><h3>{sn} {badge}</h3><div class='fields'>{''.join(field_cards)}</div></section>"
        )

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
  </style>
</head>
<body>
  <h1>{esc_title}</h1>
{body}
</body>
</html>
"""


def _resolve_path(raw: Path, cwd: Path) -> Path:
    if raw.is_absolute():
        return raw.resolve()
    for base in (cwd, _ROOT):
        hit = (base / raw).resolve()
        if hit.is_file():
            return hit
    return (_ROOT / raw).resolve()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="将评审 JSON 渲染为静态 HTML")
    ap.add_argument("-i", "--input", type=Path, required=True, help="工作 JSON 或 *_answered.json")
    ap.add_argument("-o", "--output", type=Path, required=True, help="输出 .html 路径")
    ap.add_argument("--title", default="案卷评审结果浏览", help="页面标题")
    ns = ap.parse_args(argv)
    in_path = _resolve_path(ns.input, Path.cwd())
    out_path = Path(ns.output)
    out_path = out_path.resolve() if out_path.is_absolute() else (Path.cwd() / out_path).resolve()

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
    html_doc = render_review_html(data, title=ns.title)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"已写出: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
