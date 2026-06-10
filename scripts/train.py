from __future__ import annotations

import argparse
import json

from src.training import train_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA/QLoRA on Spider train.")
    parser.add_argument("--config", required=True, help="Training YAML config.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--dry_run", action="store_true", help="Validate data/config formatting without loading the model.")
    args = parser.parse_args()
    logs = train_lora(args.config, max_steps=args.max_steps, dry_run=args.dry_run)
    print(json.dumps(logs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
