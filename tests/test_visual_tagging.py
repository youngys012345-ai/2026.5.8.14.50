from pathlib import Path

from visual_tagging import build_visual_summary_sentence, enrich_document_with_visual_tags


def test_enrich_document_assigns_tag_to_nearest_table_cell(tmp_path: Path) -> None:
    image_file = tmp_path / "doc_images" / "imageFile1.png"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_bytes(b"fake")

    document = {
        "file name": "doc.pdf",
        "kids": [
            {
                "type": "table",
                "page number": 1,
                "rows": [
                    {
                        "cells": [
                            {
                                "type": "table cell",
                                "page number": 1,
                                "bounding box": [10, 10, 110, 50],
                                "kids": [{"type": "paragraph", "content": "签字栏"}],
                            }
                        ]
                    }
                ],
            },
            {
                "type": "image",
                "page number": 1,
                "bounding box": [12, 12, 30, 30],
                "source": "doc_images/imageFile1.png",
            },
        ],
    }

    def detector(_: Path) -> tuple[str, float]:
        return ("手写签名", 0.98)

    enrich_document_with_visual_tags(
        document=document,
        json_file_path=tmp_path / "doc.json",
        detect_fn=detector,
    )

    cell = document["kids"][0]["rows"][0]["cells"][0]
    assert cell["visual_tags"] == ["手写签名"]
    assert document["kids"][1]["visual_anchor_mode"] == "table_row_overlap"
    assert document["visual_tag_stats"]["counts"]["手写签名"] == 1
    assert document["visual_tag_stats"]["summary_sentence"] == "存在1个人的手写签名。"


def test_enrich_document_assigns_tag_to_nearest_paragraph(tmp_path: Path) -> None:
    image_file = tmp_path / "doc_images" / "imageFile2.png"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_bytes(b"fake")

    document = {
        "file name": "doc.pdf",
        "kids": [
            {
                "type": "paragraph",
                "page number": 2,
                "bounding box": [20, 20, 200, 60],
                "content": "此处是申请人信息。",
            },
            {
                "type": "image",
                "page number": 2,
                "bounding box": [25, 25, 40, 40],
                "source": "doc_images/imageFile2.png",
            },
        ],
    }

    def detector(_: Path) -> tuple[str, float]:
        return ("印章", 0.88)

    enrich_document_with_visual_tags(
        document=document,
        json_file_path=tmp_path / "doc.json",
        detect_fn=detector,
    )

    paragraph = document["kids"][0]
    assert paragraph["visual_tags"] == ["印章"]
    assert document["kids"][1]["visual_anchor_mode"] == "paragraph_overlap"
    assert document["visual_tag_stats"]["counts"]["印章"] == 1
    assert document["visual_tag_stats"]["summary_sentence"] == "存在1个印章。"


def test_build_visual_summary_sentence_formats_multiple_types() -> None:
    sentence = build_visual_summary_sentence({"手写签名": 2, "指印": 1, "印章": 0})
    assert sentence == "存在2个人的手写签名和1个指印。"


def test_row_overlap_tags_all_cells(tmp_path: Path) -> None:
    """图片仅与第二列单元格相交时，整行单元格均写入标签。"""
    image_file = tmp_path / "doc_images" / "img.png"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_bytes(b"x")

    document = {
        "file name": "demo.pdf",
        "kids": [
            {
                "type": "table",
                "page number": 1,
                "rows": [
                    {
                        "cells": [
                            {
                                "page number": 1,
                                "bounding box": [0, 0, 50, 50],
                                "kids": [{"content": "姓名"}],
                            },
                            {
                                "page number": 1,
                                "bounding box": [60, 0, 120, 50],
                                "kids": [{"content": "签字"}],
                            },
                        ]
                    }
                ],
            },
            {
                "type": "image",
                "page number": 1,
                "bounding box": [70, 10, 90, 30],
                "source": "doc_images/img.png",
            },
        ],
    }

    def detector(_: Path) -> tuple[str, float]:
        return ("印章", 0.9)

    enrich_document_with_visual_tags(
        document=document,
        json_file_path=tmp_path / "doc.json",
        detect_fn=detector,
    )

    cells = document["kids"][0]["rows"][0]["cells"]
    assert cells[0]["visual_tags"] == ["印章"]
    assert cells[1]["visual_tags"] == ["印章"]
    assert document["kids"][1]["visual_anchor_mode"] == "table_row_overlap"


def test_fallback_nearest_when_no_overlap(tmp_path: Path) -> None:
    """与表格不相交时回退到最近锚点。"""
    image_file = tmp_path / "doc_images" / "img2.png"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_bytes(b"x")

    document = {
        "file name": "demo.pdf",
        "kids": [
            {
                "type": "table",
                "page number": 1,
                "rows": [
                    {
                        "cells": [
                            {
                                "page number": 1,
                                "bounding box": [10, 10, 40, 40],
                                "kids": [{"content": "远栏"}],
                            }
                        ]
                    }
                ],
            },
            {
                "type": "image",
                "page number": 1,
                "bounding box": [500, 500, 520, 520],
                "source": "doc_images/img2.png",
            },
        ],
    }

    def detector(_: Path) -> tuple[str, float]:
        return ("指印", 0.95)

    enrich_document_with_visual_tags(
        document=document,
        json_file_path=tmp_path / "doc.json",
        detect_fn=detector,
    )

    assert document["kids"][0]["rows"][0]["cells"][0]["visual_tags"] == ["指印"]
    assert document["kids"][1]["visual_anchor_mode"] == "nearest"


def test_document_markdown_preserves_body_and_appends_visual_section() -> None:
    """导出 Markdown 时正文不变，仅在文末追加视觉摘要块。"""
    from document_export import document_to_markdown

    doc = {
        "file name": "a.pdf",
        "number of pages": 1,
        "kids": [{"type": "paragraph", "page number": 1, "content": "合同正文一行"}],
        "visual_tag_stats": {
            "total": 2,
            "counts": {"印章": 2},
            "summary_sentence": "存在2个印章。",
        },
    }
    md = document_to_markdown(doc)
    assert md.index("合同正文一行") < md.index("### 视觉标签摘要")
    assert "### 视觉标签摘要" in md
    assert "存在2个印章。" in md
    assert "<!-- visual_tag_total: 2 -->" in md
