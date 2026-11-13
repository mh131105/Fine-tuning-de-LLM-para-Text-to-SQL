"""Carrega o adapter Exp F e gera SQL com o prompt zero-shot do projeto."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import extract_sql
from src.prompts import build_spider_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="mh131105/qwen2.5-3b-text-to-sql-lora-exp-f")
    parser.add_argument("--revision", default="main", help="Revisão do adapter no Hub.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--schema", default="singer(singer_id, name, country)")
    parser.add_argument("--question", default="How many singers are from each country?")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens deve ser maior que zero")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    config = PeftConfig.from_pretrained(args.adapter, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, revision=args.revision)
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        revision=config.revision,
        dtype=dtype,
    )
    model = PeftModel.from_pretrained(base, args.adapter, revision=args.revision).to(device).eval()
    prompt = build_spider_prompt({"question": args.question}, args.schema)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    raw_output = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    sql = extract_sql(raw_output)
    if not sql:
        raise SystemExit("O modelo não retornou uma consulta SQL reconhecível.")
    print(sql)


if __name__ == "__main__":
    main()
