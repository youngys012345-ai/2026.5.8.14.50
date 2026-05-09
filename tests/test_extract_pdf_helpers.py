import argparse
import json
from pathlib import Path

from extract_pdf import (
    _auto_fill_local_mineru_settings,
    _auto_fill_mineru_project_root,
    _collect_pdf_files,
    _mineru_pipeline_models_parent,
    _resolve_output_layout,
)


def test_collect_pdf_files_for_directory(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.pdf").write_text("x", encoding="utf-8")

    non_recursive = _collect_pdf_files(tmp_path, recursive=False)
    recursive = _collect_pdf_files(tmp_path, recursive=True)

    assert len(non_recursive) == 1
    assert non_recursive[0].name == "a.pdf"
    assert len(recursive) == 2
    assert {p.name for p in recursive} == {"a.pdf", "c.pdf"}


def test_resolve_output_layout_defaults_and_override(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    input_dir = tmp_path / "pdfs"
    input_dir.mkdir()
    file_path = input_dir / "demo.pdf"
    file_path.write_text("x", encoding="utf-8")

    jd, md, root = _resolve_output_layout(
        workspace=workspace,
        input_path=file_path,
        output_dir=None,
        json_output_dir=None,
        markdown_output_dir=None,
    )
    assert root == (workspace / "output").resolve()
    assert jd == (root / "json").resolve()
    assert md == (root / "markdown").resolve()

    jd2, md2, root2 = _resolve_output_layout(
        workspace=workspace,
        input_path=file_path,
        output_dir=str(tmp_path / "out"),
        json_output_dir=None,
        markdown_output_dir=None,
    )
    assert root2 == (tmp_path / "out").resolve()
    assert jd2 == (tmp_path / "out" / "json").resolve()

    jd3, md3, root3 = _resolve_output_layout(
        workspace=workspace,
        input_path=file_path,
        output_dir=str(tmp_path / "out"),
        json_output_dir=str(tmp_path / "jonly"),
        markdown_output_dir=str(tmp_path / "monly"),
    )
    assert jd3 == (tmp_path / "jonly").resolve()
    assert md3 == (tmp_path / "monly").resolve()


def test_auto_fill_local_mineru_settings(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    local_cfg = workspace / "config" / "mineru.local.json"
    local_cfg.write_text('{"models-dir": {"pipeline": "x"}}', encoding="utf-8")
    args = argparse.Namespace(
        mineru_tools_config_json=None,
        mineru_model_source=None,
    )

    _auto_fill_local_mineru_settings(args=args, workspace=workspace)

    assert args.mineru_tools_config_json == str(local_cfg.resolve())
    assert args.mineru_model_source == "local"


def test_mineru_pipeline_models_parent_prefers_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "MinerU"
    (repo / "models").mkdir(parents=True)
    assert _mineru_pipeline_models_parent(repo) == repo.resolve()


def test_mineru_pipeline_models_parent_nested_pipeline(tmp_path: Path) -> None:
    repo = tmp_path / "MinerU"
    nested = repo / "pipeline" / "models"
    nested.mkdir(parents=True)
    assert _mineru_pipeline_models_parent(repo) == (repo / "pipeline").resolve()


def test_auto_fill_prefers_mineru_repo_models_over_legacy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mineru_repo = workspace / "MinerU"
    (mineru_repo / "mineru").mkdir(parents=True)
    (mineru_repo / "models").mkdir()
    (mineru_repo / "models" / ".keep").write_text("", encoding="utf-8")

    legacy = workspace / "local_models" / "mineru" / "pipeline" / "models"
    legacy.mkdir(parents=True)
    (legacy / ".keep").write_text("", encoding="utf-8")

    args = argparse.Namespace(
        mineru_tools_config_json=None,
        mineru_model_source=None,
        mineru_project_root=str(mineru_repo.resolve()),
        backend="mineru",
    )
    try:
        _auto_fill_local_mineru_settings(args=args, workspace=workspace)
        assert args.mineru_model_source == "local"
        assert args.mineru_tools_config_json
        cfg = json.loads(Path(args.mineru_tools_config_json).read_text(encoding="utf-8"))
        assert cfg["models-dir"]["pipeline"] == str(mineru_repo.resolve())
    finally:
        p = getattr(args, "_mineru_autogen_tools_json", None)
        if p:
            Path(p).unlink(missing_ok=True)


def test_auto_fill_generates_tools_when_pipeline_models_present(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    models_dir = workspace / "local_models" / "mineru" / "pipeline" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / ".keep").write_text("", encoding="utf-8")
    args = argparse.Namespace(
        mineru_tools_config_json=None,
        mineru_model_source=None,
        backend="mineru",
    )
    try:
        _auto_fill_local_mineru_settings(args=args, workspace=workspace)
        assert args.mineru_model_source == "local"
        assert args.mineru_tools_config_json
        cfg = json.loads(Path(args.mineru_tools_config_json).read_text(encoding="utf-8"))
        assert cfg["models-dir"]["pipeline"] == str(
            (workspace / "local_models" / "mineru" / "pipeline").resolve()
        )
    finally:
        p = getattr(args, "_mineru_autogen_tools_json", None)
        if p:
            Path(p).unlink(missing_ok=True)


def test_auto_fill_mineru_project_root_from_parent_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "opendataloader_quickstart"
    workspace.mkdir(parents=True)
    mineru_repo = tmp_path / "MinerU"
    (mineru_repo / "mineru").mkdir(parents=True)
    args = argparse.Namespace(mineru_project_root=None)
    _auto_fill_mineru_project_root(args=args, workspace=workspace)
    assert args.mineru_project_root == str(mineru_repo.resolve())


def test_auto_fill_skips_autogen_when_user_wants_hub(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    models_dir = workspace / "local_models" / "mineru" / "pipeline" / "models"
    models_dir.mkdir(parents=True)
    args = argparse.Namespace(
        mineru_tools_config_json=None,
        mineru_model_source="huggingface",
        backend="mineru",
    )
    _auto_fill_local_mineru_settings(args=args, workspace=workspace)
    assert args.mineru_tools_config_json is None
    assert args.mineru_model_source == "huggingface"
