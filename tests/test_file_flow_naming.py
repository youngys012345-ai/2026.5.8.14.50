# -*- coding: utf-8 -*-
"""file_flow/naming：pipeline 后缀与文件名生成。"""

from __future__ import annotations

from file_flow.naming import (
    file_flow_stem_suffixes,
    replace_work_json_name_with_answered,
    review_json_filename_for_base,
    stem_base_from_stage_stem,
    work_json_filename_for_stem,
)


def test_default_suffixes() -> None:
    sw, sa, sr = file_flow_stem_suffixes({})
    assert sw == "_work" and sa == "_answered" and sr == "_review"


def test_custom_suffixes() -> None:
    m = {
        "file_flow_suffix_work": "_draft",
        "file_flow_suffix_answered": "_ans",
        "file_flow_suffix_review": "_rev",
    }
    assert work_json_filename_for_stem("case", m) == "case_draft.json"
    assert replace_work_json_name_with_answered("case_draft.json", m) == "case_ans.json"
    assert stem_base_from_stage_stem("case_ans", m) == "case"
    assert review_json_filename_for_base("case", m) == "case_rev.json"
