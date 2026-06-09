from __future__ import annotations

import argparse
import json

from src.tp2.preparation import prepare_spider


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Spider train/dev JSONL files and schemas.")
    parser.add_argument("--data_dir", required=True, help="Raw Spider directory with train_spider.json, dev.json and tables.json.")
    parser.add_argument("--output_dir", default="data/processed/spider", help="Processed Spider output directory.")
    args = parser.parse_args()
    metadata = prepare_spider(args.data_dir, args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
