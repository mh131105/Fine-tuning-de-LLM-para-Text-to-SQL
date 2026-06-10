from __future__ import annotations

import argparse
import json

from src.evaluation import run_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Spider and MMLU benchmarks and write summaries.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True, help="Output directory for the model/experiment being evaluated.")
    parser.add_argument("--limit", type=int, default=None, help="Optional shared limit for smoke tests.")
    parser.add_argument("--mock", action="store_true", help="Use gold answers as model outputs for smoke tests.")
    args = parser.parse_args()
    summary = run_benchmarks(args.config, args.model_path, limit=args.limit, mock=args.mock)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
