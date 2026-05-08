from query_extracted_json import _fill_template_for_document, _load_template_json


def test_fill_template_content_and_visual_summary() -> None:
    document = {
        "file name": "demo.pdf",
        "title": "立案登记表",
        "visual_tag_stats": {
            "summary_sentence": "存在1个人的手写签名。",
        },
        "kids": [
            {
                "type": "heading",
                "page number": 1,
                "bounding box": [40, 700, 180, 730],
                "content": "立案登记表",
            },
            {
                "type": "table",
                "page number": 1,
                "rows": [
                    {
                        "cells": [
                            {
                                "bounding box": [40, 640, 160, 670],
                                "kids": [{"content": "案件来源"}],
                            },
                            {
                                "bounding box": [170, 640, 360, 670],
                                "kids": [{"content": "行政检查中发现"}],
                            },
                        ]
                    }
                ],
            },
            {
                "type": "paragraph",
                "page number": 1,
                "bounding box": [40, 590, 420, 620],
                "content": "经办机构负责人意见：建议立案。",
                "visual_tags": ["手写签名", "印章"],
            },
        ],
    }

    template = {
        "立案登记表": {
            "是否必须": "必须",
            "字段": {
                "案件来源": {
                    "是否需要识别手写体": "否",
                    "要求": [],
                    "内容": "",
                },
                "经办机构负责人意见": {
                    "是否需要识别手写体": "是",
                    "要求": [],
                    "内容": "",
                },
            },
        }
    }

    result = _fill_template_for_document(document=document, template=template)
    fields = result["立案登记表"]["字段"]
    # 小标题后区块，不含小标题本身与整行重复拼接
    assert fields["案件来源"]["内容"] == "行政检查中发现"
    assert (
        fields["经办机构负责人意见"]["内容"]
        == "建议立案。；视觉标签：手写签名、印章"
    )


def test_bbox_extract_only_right_or_below_text() -> None:
    """字段值只允许从目标右侧或下侧近邻提取，不取左侧或上侧。"""
    document = {
        "file name": "demo.pdf",
        "title": "立案登记表",
        "kids": [
            {"type": "heading", "page number": 1, "bounding box": [40, 700, 180, 730], "content": "立案登记表"},
            {"type": "paragraph", "page number": 1, "bounding box": [100, 500, 200, 530], "content": "案由"},
            {"type": "paragraph", "page number": 1, "bounding box": [20, 500, 90, 530], "content": "左侧不应命中"},
            {"type": "paragraph", "page number": 1, "bounding box": [100, 540, 200, 570], "content": "上方不应命中"},
            {"type": "paragraph", "page number": 1, "bounding box": [210, 500, 360, 530], "content": "涉嫌无证经营"},
            {"type": "paragraph", "page number": 1, "bounding box": [100, 450, 220, 480], "content": "下方备选"},
        ],
    }
    template = {
        "立案登记表": {
            "是否必须": "必须",
            "字段": {
                "案由": {"是否需要识别手写体": "否", "要求": [], "内容": ""},
            },
        }
    }
    result = _fill_template_for_document(document=document, template=template)
    assert result["立案登记表"]["字段"]["案由"]["内容"] == "涉嫌无证经营"


def test_level1_title_matched_without_heading_type() -> None:
    """一级标题出现在非 heading 节点（如段落）时也应命中。"""
    document = {
        "file name": "demo.pdf",
        "title": "其他",
        "kids": [
            {
                "type": "paragraph",
                "page number": 1,
                "content": "立案登记表（副本）",
            },
            {
                "type": "table",
                "page number": 1,
                "rows": [
                    {
                        "cells": [
                            {"kids": [{"content": "案由"}]},
                            {"kids": [{"content": "涉嫌无证经营"}]},
                        ]
                    }
                ],
            },
        ],
    }
    template = {
        "立案登记表": {
            "是否必须": "必须",
            "字段": {
                "案由": {"是否需要识别手写体": "否", "要求": [], "内容": ""},
            },
        }
    }
    result = _fill_template_for_document(document=document, template=template)
    assert result["立案登记表"]["字段"]["案由"]["内容"] == "涉嫌无证经营"


def test_level1_slash_either_alternative_fills_fields() -> None:
    """一级键含 / 时，任一分支在 heading 命中即可填字段。"""
    document = {
        "file name": "demo.pdf",
        "title": "其他",
        "kids": [
            {"type": "heading", "page number": 1, "content": "登记表副本"},
            {
                "type": "table",
                "page number": 1,
                "rows": [
                    {
                        "cells": [
                            {"kids": [{"content": "案件来源"}]},
                            {"kids": [{"content": "移送"}]},
                        ]
                    }
                ],
            },
        ],
    }
    template = {
        "立案登记表/登记表副本": {
            "是否必须": "必须",
            "字段": {
                "案件来源": {
                    "是否需要识别手写体": "否",
                    "要求": [],
                    "内容": "",
                }
            },
        }
    }

    result = _fill_template_for_document(document=document, template=template)
    assert result["立案登记表/登记表副本"]["字段"]["案件来源"]["内容"] == "移送"


def test_level1_slash_none_match_all_missing() -> None:
    document = {
        "file name": "demo.pdf",
        "title": "无关",
        "kids": [
            {"type": "heading", "page number": 1, "content": "附录"},
            {"type": "paragraph", "page number": 1, "content": "案件来源：测试"},
        ],
    }
    template = {
        "登记表甲/登记表乙": {
            "是否必须": "必须",
            "字段": {
                "案件来源": {
                    "是否需要识别手写体": "否",
                    "要求": [],
                    "内容": "",
                }
            },
        }
    }

    result = _fill_template_for_document(document=document, template=template)
    assert result["登记表甲/登记表乙"]["字段"]["案件来源"]["内容"] == "内容缺失"


def test_fill_template_marks_missing_when_level1_title_not_found() -> None:
    document = {
        "file name": "demo.pdf",
        "title": "其他文书",
        "kids": [
            {"type": "heading", "page number": 1, "content": "其他文书"},
            {"type": "paragraph", "page number": 1, "content": "案件来源：测试"},
        ],
    }
    template = {
        "立案登记表": {
            "是否必须": "必须",
            "字段": {
                "案件来源": {
                    "是否需要识别手写体": "否",
                    "要求": [],
                    "内容": "",
                }
            },
        }
    }

    result = _fill_template_for_document(document=document, template=template)
    assert result["立案登记表"]["字段"]["案件来源"]["内容"] == "内容缺失"


def test_load_template_json_valid_object(tmp_path) -> None:
    path = tmp_path / "template.json"
    path.write_text(
        '{"一级": {"是否必须": "必须", "字段": {"A": {"是否需要识别手写体": "否", "要求": [], "内容": ""}}}}',
        encoding="utf-8",
    )
    loaded = _load_template_json(path)
    assert "一级" in loaded
