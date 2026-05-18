# -*- coding: utf-8 -*-
"""file_flow.opendataloader_adapter 中 client_base_url_for_local_service 单元测试。"""

from __future__ import annotations

from file_flow.opendataloader_adapter import client_base_url_for_local_service


def test_client_base_url_empty_unchanged() -> None:
    assert client_base_url_for_local_service("") == ""
    assert client_base_url_for_local_service("   ") == ""


def test_client_base_url_localhost_unchanged() -> None:
    assert client_base_url_for_local_service("http://127.0.0.1:5002/") == "http://127.0.0.1:5002/"
    assert client_base_url_for_local_service("http://localhost:5002") == "http://localhost:5002"


def test_client_base_url_zero_rewritten() -> None:
    assert client_base_url_for_local_service("http://0.0.0.0:5002") == "http://127.0.0.1:5002"
    assert client_base_url_for_local_service("http://0.0.0.0:5002/health") == "http://127.0.0.1:5002/health"


def test_client_base_url_zero_no_port() -> None:
    assert client_base_url_for_local_service("http://0.0.0.0") == "http://127.0.0.1"


def test_client_base_url_userinfo_preserved() -> None:
    assert (
        client_base_url_for_local_service("http://u:p@0.0.0.0:5002/x")
        == "http://u:p@127.0.0.1:5002/x"
    )
