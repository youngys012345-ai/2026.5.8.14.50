"""document_export 模块单元测试。"""

from document_export import document_to_markdown_by_page


def test_markdown_by_page_groups_top_level_kids() -> None:
    doc = {
        "file name": "demo.pdf",
        "number of pages": 2,
        "title": "测试文档",
        "kids": [
            {"type": "paragraph", "page number": 1, "content": "第一页正文"},
            {"type": "heading", "heading level": 2, "page number": 2, "content": "第二节"},
            {"type": "paragraph", "page number": 2, "content": "第二页段落"},
        ],
    }
    md = document_to_markdown_by_page(doc)
    assert "## 第 1 页 · Page 1" in md
    assert "## 第 2 页 · Page 2" in md
    assert "第一页正文" in md
    assert "第二页段落" in md
    assert "**本页标题（推断）:** 第二节" in md
    assert "<!-- page_index: 1 -->" in md
    assert "<!-- llm_layout: by-page -->" in md


def test_markdown_by_page_fills_empty_declared_pages() -> None:
    doc = {
        "file name": "x.pdf",
        "number of pages": 3,
        "title": "",
        "kids": [{"type": "paragraph", "page number": 1, "content": "only p1"}],
    }
    md = document_to_markdown_by_page(doc)
    assert "## 第 3 页 · Page 3" in md
    assert "本页无结构化文本块" in md
