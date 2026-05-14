# -*- coding: utf-8 -*-
"""file_flow.prompt_stage_samples：两阶段 user 提示可重复生成。"""

from __future__ import annotations

from file_flow.prompt_stage_samples import build_sample_prompts


def test_sample_prompts_two_stages() -> None:
    d = build_sample_prompts()
    assert set(d.keys()) == {"1_schema_extract", "2_standards_review"}
    for v in d.values():
        assert "---" in v
        assert len(v) > 80
