from __future__ import annotations

import argparse
import json

from src.tp2.evaluation import evaluate_mmlu


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MMLU suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default=None, help="Experiment output directory containing an adapter, or baseline output dir.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use gold answers as model outputs for smoke tests.")
    args = parser.parse_args()
    metrics = evaluate_mmlu(args.config, model_path=args.model_path, output_dir=args.output_dir, limit=args.limit, mock=args.mock)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
