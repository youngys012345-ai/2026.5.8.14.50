"""vlm_client：解析逻辑与 OpenAI 风格响应抽取的单元测试。"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from vlm_client import (
    _extract_message_content,
    build_openai_compatible_vlm_detector,
    join_openai_compatible_endpoint_url,
    parse_vlm_classification_text,
)


def test_parse_plain_json() -> None:
    out = parse_vlm_classification_text('{"label":"印章","confidence":0.87}')
    assert out == ("印章", 0.87)


def test_parse_json_in_markdown_fence() -> None:
    text = '说明如下：\n```json\n{"label": "signature", "confidence": 0.9}\n```'
    out = parse_vlm_classification_text(text)
    assert out == ("手写签名", 0.9)


def test_parse_clamps_confidence() -> None:
    out = parse_vlm_classification_text('{"label":"指印","confidence":2}')
    assert out == ("指印", 1.0)


def test_extract_message_content_string() -> None:
    resp = {"choices": [{"message": {"content": '{"label":"印章","confidence":0.5}'}}]}
    assert _extract_message_content(resp) == '{"label":"印章","confidence":0.5}'


def test_extract_message_content_multipart_text() -> None:
    resp = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": '{"label":"印章",'},
                        {"type": "text", "text": '"confidence":0.5}'},
                    ]
                }
            }
        ]
    }
    merged = _extract_message_content(resp)
    assert '"confidence":0.5}' in merged


def test_join_openai_compatible_endpoint_url_base_and_path() -> None:
    assert (
        join_openai_compatible_endpoint_url("https://api.openai.com/v1", "/chat/completions")
        == "https://api.openai.com/v1/chat/completions"
    )
    assert (
        join_openai_compatible_endpoint_url("https://host/", "v1/chat/completions")
        == "https://host/v1/chat/completions"
    )


def test_join_openai_compatible_endpoint_url_full_url_in_path() -> None:
    u = "https://other/v1/chat/completions"
    assert join_openai_compatible_endpoint_url("https://ignored", u) == u


def test_join_openai_compatible_endpoint_url_only_base_no_path() -> None:
    """路径为空时 ``api_base`` 视为完整 POST URL。"""
    u = "https://api.example/v1/chat/completions"
    assert join_openai_compatible_endpoint_url(u, None) == u
    assert join_openai_compatible_endpoint_url(u, "") == u


def test_build_detector_calls_api(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # 最小 PNG 文件头片段即可用于 mime

    fake_resp = {
        "choices": [
            {"message": {"content": '{"label":"手写签名","confidence":0.95}'}}
        ]
    }
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value.read.return_value = json.dumps(fake_resp).encode("utf-8")
    mock_cm.__exit__.return_value = None

    with patch("vlm_client.urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
        detect = build_openai_compatible_vlm_detector(
            api_base="https://example.com/v1/chat/completions",
            api_key="sk-test",
            model="gpt-4o-mini",
            timeout_sec=30.0,
        )
        label, score = detect(img)

    assert label == "手写签名"
    assert abs(score - 0.95) < 1e-6
    mock_urlopen.assert_called_once()
    call_kw = mock_urlopen.call_args
    req = call_kw[0][0]
    assert req.full_url == "https://example.com/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-test"
