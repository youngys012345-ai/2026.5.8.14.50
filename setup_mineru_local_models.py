#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过 HuggingFace 国内镜像下载 MinerU 模型到本地目录，并生成本地调用配置。

默认行为：
1. 从 opendatalab/PDF-Extract-Kit-1.0 下载 pipeline 模型；
2. 下载目录使用用户指定的本地文件夹；
3. 生成 MinerU 配置文件，写入 models-dir.pipeline；
4. 提示 extract_pdf.py 所需参数（--mineru-model-source local）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "opendatalab/PDF-Extract-Kit-1.0"
DEFAULT_MIRROR = "https://hf-mirror.com"


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _merge_models_dir(config: dict[str, Any], model_type: str, model_root: Path) -> dict[str, Any]:
    new_cfg = dict(config)
    models_dir = new_cfg.get("models-dir")
    if not isinstance(models_dir, dict):
        models_dir = {}
    models_dir[model_type] = str(model_root.resolve())
    new_cfg["models-dir"] = models_dir
    return new_cfg


def _download_pipeline_models(repo_id: str, local_dir: Path, endpoint: str) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    # 使用本地目录直写并禁用符号链接，避免 Windows 权限导致 WinError 1314。
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        endpoint=endpoint,
        allow_patterns=["models/*", "models/**/*"],
    )
    return local_dir.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 MinerU 模型到本地目录（HF 国内镜像）")
    parser.add_argument(
        "--model-dir",
        default="local_models/mineru/pipeline",
        help="本地模型目录（默认 local_models/mineru/pipeline）",
    )
    parser.add_argument(
        "--config-path",
        default="config/mineru.local.json",
        help="生成的 MinerU 配置文件路径（默认 config/mineru.local.json）",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace 仓库（默认 {DEFAULT_REPO_ID}）",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=DEFAULT_MIRROR,
        help=f"HF 镜像地址（默认 {DEFAULT_MIRROR}）",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    model_dir = (workspace / args.model_dir).resolve()
    config_path = (workspace / args.config_path).resolve()

    # 设置镜像端点，便于 huggingface_hub 内部请求统一走国内镜像。
    os.environ["HF_ENDPOINT"] = args.hf_endpoint

    try:
        model_root = _download_pipeline_models(
            repo_id=args.repo_id,
            local_dir=model_dir,
            endpoint=args.hf_endpoint,
        )
    except Exception as exc:
        print(f"模型下载失败: {exc}")
        return 1

    old_cfg = _load_json_if_exists(config_path)
    new_cfg = _merge_models_dir(old_cfg, model_type="pipeline", model_root=model_root)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(new_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("模型下载完成。")
    print(f"- 本地模型目录: {model_root}")
    print(f"- MinerU 配置文件: {config_path}")
    print("调用方式示例：")
    print(
        f'  python extract_pdf.py "你的PDF目录" '
        f'--mineru-model-source local --mineru-tools-config-json "{config_path}"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
