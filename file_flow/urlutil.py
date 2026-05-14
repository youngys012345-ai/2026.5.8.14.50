# -*- coding: utf-8 -*-
"""与 file_flow 调用外部 HTTP 服务相关的 URL 工具。"""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def client_base_url_for_local_service(url: str) -> str:
    """
    将 ``http://0.0.0.0:端口`` 规范为 ``http://127.0.0.1:端口``。

    ``0.0.0.0`` 仅表示进程绑定在所有接口上，**不能**作为本机客户端的目标主机；
    本机访问应使用 ``127.0.0.1`` 或 ``localhost``。
    """
    s = (url or "").strip()
    if not s:
        return s
    p = urlparse(s)
    h = (p.hostname or "").lower()
    if h != "0.0.0.0":
        return s
    if p.port:
        netloc = f"127.0.0.1:{p.port}"
    else:
        netloc = "127.0.0.1"
    if p.username:
        pw = f":{p.password}" if p.password else ""
        netloc = f"{p.username}{pw}@{netloc}"
    return urlunparse((p.scheme or "http", netloc, p.path or "", p.params or "", p.query, p.fragment))
