"""pipeline_config 模块的单元测试。"""

import json
from pathlib import Path

from pipeline_config import (
    load_config_file,
    merge_defaults,
    pop_config_path_from_argv,
)


def test_pop_config_path_from_argv() -> None:
    p, rest = pop_config_path_from_argv(["a.pdf", "--config", "c.json", "--quiet"])
    assert p == Path("c.json")
    assert rest == ["a.pdf", "--quiet"]


def test_pop_config_path_equals_form() -> None:
    p, rest = pop_config_path_from_argv(["--config=cfg.json", "x.pdf"])
    assert p == Path("cfg.json")
    assert rest == ["x.pdf"]


def test_load_config_file_skips_unknown_and_null(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps(
            {
                "_comment": "ignored",
                "hybrid_url": "http://example:9999",
                "output_dir": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    data = load_config_file(path)
    assert data == {"hybrid_url": "http://example:9999"}


def test_merge_defaults_order() -> None:
    m = merge_defaults({"a": 1}, {"a": 2}, {"b": 3})
    assert m == {"a": 2, "b": 3}
