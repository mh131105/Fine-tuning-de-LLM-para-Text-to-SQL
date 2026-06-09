import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.evaluate_spider import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/baseline")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    evaluate(args.config, adapter_path=None, output_dir=args.output_dir, limit=args.limit)
