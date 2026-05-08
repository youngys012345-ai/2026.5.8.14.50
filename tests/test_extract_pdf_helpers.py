from pathlib import Path

from extract_pdf import _collect_pdf_files, _resolve_json_output_dir


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


def test_resolve_json_output_dir_default_and_override(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdfs"
    input_dir.mkdir()
    file_path = input_dir / "demo.pdf"
    file_path.write_text("x", encoding="utf-8")

    default_out = _resolve_json_output_dir(file_path, output_dir=None, json_output_dir=None)
    output_dir_out = _resolve_json_output_dir(file_path, output_dir=str(tmp_path / "out"), json_output_dir=None)
    custom_out = _resolve_json_output_dir(
        file_path,
        output_dir=str(tmp_path / "out"),
        json_output_dir=str(tmp_path / "custom-json"),
    )

    assert default_out == (input_dir / "opendataloader_out" / "json").resolve()
    assert output_dir_out == (tmp_path / "out" / "json").resolve()
    assert custom_out == (tmp_path / "custom-json").resolve()
