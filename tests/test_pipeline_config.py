"""pipeline_config 模块的单元测试。"""

import json
from pathlib import Path

from pipeline_config import (
    defaults_from_environment,
    load_config_file,
    merge_defaults,
    pop_config_path_from_argv,
    resolve_pipeline_config_path,
)


def test_resolve_pipeline_config_path_fixes_pipline_typo(tmp_path: Path) -> None:
    good = tmp_path / "pipeline.json"
    good.write_text("{}", encoding="utf-8")
    bad = tmp_path / "pipline.json"
    resolved, hint = resolve_pipeline_config_path(bad)
    assert resolved == good.resolve()
    assert hint is not None


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
                "mineru_project_root": "..\\MinerU",
                "output_dir": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    data = load_config_file(path)
    assert data == {"mineru_project_root": "..\\MinerU"}


def test_merge_defaults_order() -> None:
    m = merge_defaults({"a": 1}, {"a": 2}, {"b": 3})
    assert m == {"a": 2, "b": 3}


def test_defaults_from_environment_with_mineru_keys(monkeypatch) -> None:
    monkeypatch.setenv("MINERU_BACKEND", "pipeline")
    monkeypatch.setenv("MINERU_API_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "local")
    monkeypatch.setenv("MINERU_TOOLS_CONFIG_JSON", "config/mineru.local.json")
    monkeypatch.setenv("MINERU_PROJECT_ROOT", "D:\\MinerU")

    data = defaults_from_environment()
    assert data["mineru_backend"] == "pipeline"
    assert data["mineru_api_url"] == "http://127.0.0.1:8000"
    assert data["mineru_model_source"] == "local"
    assert data["mineru_tools_config_json"] == "config/mineru.local.json"
    assert data["mineru_project_root"] == "D:\\MinerU"
