#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印 file_flow 两阶段「将发给大模型的单条 user 正文」示例，便于人工核对（不含真实 API 调用）。

在仓库根执行::

    python -m file_flow.prompt_stage_samples
"""

from __future__ import annotations

SAMPLE_PDF = "（示例 PDF 正文，仅作占位）\n当事人：张三。立案日期：2024-01-01。"


def build_sample_prompts() -> dict[str, str]:
    """返回 schema 摘录与 standards_review 的完整 user 提示（与线上一致：环节指令 + --- + 结构化正文）。"""
    merged: dict = {}

    from file_flow.schema_llm_extract import build_public_context, build_schema_extract_full_user_prompt
    from file_flow.standards_llm_review import (
        build_focused_field_context,
        build_standards_review_full_user_prompt,
    )

    pub = build_public_context("示例文书", "文书级说明占位")
    stage1 = build_schema_extract_full_user_prompt(
        merged,
        pub,
        "示例字段",
        "字段说明占位",
        SAMPLE_PDF,
    )
    work = {
        "schema_version": "1",
        "document_types": [
            {
                "document_name": "示例文书",
                "fields": [{"field_name": "示例字段", "content": "从上一步摘录得到的 content 占位文本。"}],
            }
        ],
    }
    row = {
        "category": "程序",
        "subcategory": "立案",
        "content": "标准条目说明占位",
        "standard": "须对照判断的 standard 占位条文",
    }
    field_items = {
        "1.1": {
            "documents": [
                {"document_name": "示例文书", "field_names": ["示例字段"]},
            ],
        },
    }
    ctx = build_focused_field_context(work, field_items, "1.1")
    stage2 = build_standards_review_full_user_prompt(merged, row, focused_context=ctx)

    return {
        "1_schema_extract": stage1,
        "2_standards_review": stage2,
    }


def main() -> None:
    for name, text in build_sample_prompts().items():
        bar = "=" * 20 + f" {name} " + "=" * 20
        print("\n" + bar + "\n")
        print(text)


if __name__ == "__main__":
    main()
