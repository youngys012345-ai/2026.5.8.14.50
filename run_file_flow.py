#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从仓库根启动 file_flow 全流程编排。

等价命令::

    python -m file_flow.pipeline_merge

本脚本便于不习惯 ``-m`` 或写错路径时使用；请勿在 ``file_flow/`` 目录内单独执行
``python pipeline_merge.py``（会因包相对导入失败）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from file_flow.pipeline_merge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
