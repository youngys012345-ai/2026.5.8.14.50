import json
from pathlib import Path
import sys

from mineru_adapter import convert_mineru_content_list_to_document, run_mineru_cli_for_pdf


def test_convert_content_list_v1_to_document(tmp_path: Path) -> None:
    content_path = tmp_path / "demo_content_list.json"
    content_path.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "text": "立案登记表",
                    "text_level": 1,
                    "bbox": [10, 20, 300, 60],
                    "page_idx": 0,
                },
                {
                    "type": "text",
                    "text": "案件来源",
                    "bbox": [10, 100, 100, 130],
                    "page_idx": 0,
                },
                {
                    "type": "text",
                    "text": "行政检查中发现",
                    "bbox": [120, 100, 300, 130],
                    "page_idx": 0,
                },
                {
                    "type": "table",
                    "table_body": "<table><tr><td>案由</td><td>涉嫌无证经营</td></tr></table>",
                    "bbox": [10, 200, 500, 260],
                    "page_idx": 0,
                },
                {
                    "type": "image",
                    "img_path": "images/a.jpg",
                    "bbox": [400, 80, 500, 180],
                    "page_idx": 0,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    doc = convert_mineru_content_list_to_document(
        content_list_path=content_path,
        source_pdf=tmp_path / "demo.pdf",
    )

    assert doc["file name"] == "demo.pdf"
    assert doc["number of pages"] == 1
    assert any(item.get("type") == "heading" for item in doc["kids"])
    assert any(item.get("type") == "table" for item in doc["kids"])
    image_nodes = [item for item in doc["kids"] if item.get("type") == "image"]
    assert len(image_nodes) == 1
    assert Path(image_nodes[0]["source"]).name == "a.jpg"


def test_convert_content_list_v2_to_document(tmp_path: Path) -> None:
    content_path = tmp_path / "demo_content_list_v2.json"
    content_path.write_text(
        json.dumps(
            [
                [
                    {
                        "type": "title",
                        "content": {
                            "title_content": [
                                {"type": "text", "content": "立案登记表"},
                            ],
                            "level": 1,
                        },
                        "bbox": [10, 20, 300, 60],
                    },
                    {
                        "type": "paragraph",
                        "content": {
                            "paragraph_content": [
                                {"type": "text", "content": "案件来源：移送"},
                            ],
                        },
                        "bbox": [10, 100, 300, 130],
                    },
                ],
                [
                    {
                        "type": "table",
                        "content": {
                            "table_body": "<table><tr><td>项目名称</td><td>综合楼工程</td></tr></table>",
                        },
                        "bbox": [10, 200, 500, 260],
                    }
                ],
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    doc = convert_mineru_content_list_to_document(
        content_list_path=content_path,
        source_pdf=tmp_path / "demo.pdf",
    )

    assert doc["title"] == "立案登记表"
    assert doc["number of pages"] == 2
    heading_nodes = [item for item in doc["kids"] if item.get("type") == "heading"]
    assert len(heading_nodes) == 1
    assert heading_nodes[0]["content"] == "立案登记表"
    table_nodes = [item for item in doc["kids"] if item.get("type") == "table"]
    assert len(table_nodes) == 1
    assert table_nodes[0]["rows"][0]["cells"][0]["kids"][0]["content"] == "项目名称"


def test_run_mineru_cli_uses_api_url_for_pipeline_backend(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []
    fake_output = tmp_path / "demo_content_list.json"
    fake_output.write_text("[]", encoding="utf-8")

    def _fake_run(cmd: list[str], check: bool, env=None, cwd=None) -> None:
        calls.append(cmd)

    monkeypatch.setattr("mineru_adapter.subprocess.run", _fake_run)
    monkeypatch.setattr("mineru_adapter.locate_mineru_content_list", lambda output_root, pdf_file: fake_output)

    out = run_mineru_cli_for_pdf(
        pdf_file=tmp_path / "demo.pdf",
        output_root=tmp_path,
        backend="pipeline",
        api_url="http://127.0.0.1:8000",
    )

    assert out == fake_output
    assert "--api-url" in calls[0]
    assert "-u" not in calls[0]


def test_run_mineru_cli_uses_u_for_http_client_backend(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []
    fake_output = tmp_path / "demo_content_list.json"
    fake_output.write_text("[]", encoding="utf-8")

    def _fake_run(cmd: list[str], check: bool, env=None, cwd=None) -> None:
        calls.append(cmd)

    monkeypatch.setattr("mineru_adapter.subprocess.run", _fake_run)
    monkeypatch.setattr("mineru_adapter.locate_mineru_content_list", lambda output_root, pdf_file: fake_output)

    out = run_mineru_cli_for_pdf(
        pdf_file=tmp_path / "demo.pdf",
        output_root=tmp_path,
        backend="hybrid-http-client",
        api_url="http://127.0.0.1:30000",
    )

    assert out == fake_output
    assert "-u" in calls[0]


def test_run_mineru_cli_sets_model_env(tmp_path: Path, monkeypatch) -> None:
    captured_env = {}
    fake_output = tmp_path / "demo_content_list.json"
    fake_output.write_text("[]", encoding="utf-8")

    def _fake_run(cmd: list[str], check: bool, env=None, cwd=None) -> None:
        if env:
            captured_env.update(env)

    monkeypatch.setattr("mineru_adapter.subprocess.run", _fake_run)
    monkeypatch.setattr("mineru_adapter.locate_mineru_content_list", lambda output_root, pdf_file: fake_output)

    run_mineru_cli_for_pdf(
        pdf_file=tmp_path / "demo.pdf",
        output_root=tmp_path,
        backend="pipeline",
        model_source="local",
        mineru_tools_config_json=str(tmp_path / "mineru.local.json"),
    )

    assert captured_env["MINERU_MODEL_SOURCE"] == "local"
    assert captured_env["MINERU_TOOLS_CONFIG_JSON"].endswith("mineru.local.json")


def test_run_mineru_cli_uses_local_project(tmp_path: Path, monkeypatch) -> None:
    captured = {"cmd": None, "cwd": None, "env": None}
    fake_output = tmp_path / "demo_content_list.json"
    fake_output.write_text("[]", encoding="utf-8")

    def _fake_run(cmd: list[str], check: bool, env=None, cwd=None) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr("mineru_adapter.subprocess.run", _fake_run)
    monkeypatch.setattr("mineru_adapter.locate_mineru_content_list", lambda output_root, pdf_file: fake_output)

    project_dir = tmp_path / "MinerU"
    project_dir.mkdir()
    run_mineru_cli_for_pdf(
        pdf_file=tmp_path / "demo.pdf",
        output_root=tmp_path,
        mineru_project_dir=str(project_dir),
    )

    assert captured["cmd"][:3] == [sys.executable, "-m", "mineru.cli.client"]
    assert captured["cwd"] == str(project_dir.resolve())
    assert str(project_dir.resolve()) in str(captured["env"].get("PYTHONPATH", ""))
