from __future__ import annotations

import argparse
import json

from src.preparation import prepare_spider


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Spider train/dev JSONL files and schemas.")
    parser.add_argument("--data_dir", default="data/raw/spider", help="Raw Spider directory with train_spider.json, dev.json and tables.json.")
    parser.add_argument("--output_dir", default="data/processed/spider", help="Processed Spider output directory.")
    parser.add_argument(
        "--source",
        choices=["auto", "none", "local", "zip", "hf"],
        default="auto",
        help="How to acquire raw Spider if --data_dir is missing. auto uses --source_path/SPIDER_SOURCE_PATH, then Hugging Face.",
    )
    parser.add_argument("--source_path", default=None, help="Existing Spider directory or archive to import before preparation.")
    parser.add_argument("--hf_repo", default="dreamerdeo/multispider", help="Hugging Face dataset repo used by --source hf/auto.")
    parser.add_argument("--cache_dir", default=".cache/datasets/spider", help="Cache directory for Hugging Face dataset snapshots.")
    parser.add_argument("--force_download", action="store_true", help="Replace --data_dir when importing/downloading raw Spider.")
    args = parser.parse_args()
    metadata = prepare_spider(
        args.data_dir,
        args.output_dir,
        source=args.source,
        source_path=args.source_path,
        hf_repo=args.hf_repo,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
