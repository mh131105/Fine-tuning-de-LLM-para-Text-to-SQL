from __future__ import annotations

import argparse
import json

from src.tp2.preparation import prepare_mmlu


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the deterministic MMLU 150-question suite.")
    parser.add_argument("--config", required=True, help="Evaluation YAML config.")
    parser.add_argument("--mock", action="store_true", help="Create a tiny deterministic local suite without Hugging Face downloads.")
    parser.add_argument("--limit_per_category", type=int, default=None, help="Limit per category for smoke tests.")
    args = parser.parse_args()
    metadata = prepare_mmlu(args.config, mock=args.mock, limit_per_category=args.limit_per_category)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
