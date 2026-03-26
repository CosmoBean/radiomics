#!/usr/bin/env python3
"""Download the MU-Glioma-Post processed dataset from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the processed MU-Glioma-Post dataset from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="sbandred/mu-glioma-post-processed",
        help="Hugging Face dataset repository to pull.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face revision (branch/commit/tag).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("processed"),
        help="Local directory where the dataset files will be written.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face access token (falls back to HF_TOKEN).",
    )
    parser.add_argument(
        "--allow-symlinks",
        action="store_true",
        help="Allow the Hugging Face client to create symlinks instead of copying.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface-hub is required; run `pip install huggingface-hub` before retrying."
        ) from exc

    dataset_root = args.dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id}@{args.revision} to {dataset_root}")
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=dataset_root,
        local_dir_use_symlinks=args.allow_symlinks,
        token=token,
        repo_type="dataset",
    )

    print("Download complete; processed files are ready under:", dataset_root)


if __name__ == "__main__":
    main()
