import json
from pathlib import Path

from run_pipeline import merge_extracted_jsons


def test_merge_extracted_jsons_collects_documents(tmp_path: Path) -> None:
    json_dir = tmp_path / "jsons"
    json_dir.mkdir()
    (json_dir / "a.json").write_text(
        json.dumps({"file name": "a.pdf", "kids": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    (json_dir / "b.json").write_text(
        json.dumps({"file name": "b.pdf", "kids": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    (json_dir / "bad.json").write_text("{bad", encoding="utf-8")

    merged_output = tmp_path / "output" / "merged.json"
    merged_count = merge_extracted_jsons(json_dir=json_dir, merged_output=merged_output, recursive=False)

    assert merged_count == 2
    payload = json.loads(merged_output.read_text(encoding="utf-8"))
    assert payload["meta"]["file_count"] == 3
    assert payload["meta"]["merged_count"] == 2
    assert len(payload["documents"]) == 2
    assert payload["documents"][0]["source_pdf"] == "a.pdf"
