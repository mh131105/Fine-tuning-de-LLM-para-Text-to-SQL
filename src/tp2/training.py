from __future__ import annotations

import time
import os
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import copy_config, ensure_dir, load_yaml
from .data import load_spider_schemas, load_spider_train, save_json, serialize_schema
from .logging_utils import save_environment_snapshot
from .model import build_lora_config, build_quantization_config, load_tokenizer
from .prompts import build_spider_prompt
from .reproducibility import set_global_seed


def format_spider_for_sft(examples: list[dict[str, Any]], schemas: dict[str, Any], few_shot_count: int = 0) -> list[dict[str, str]]:
    few_shot = []
    if few_shot_count:
        for example in examples[:few_shot_count]:
            enriched = dict(example)
            enriched["schema_text"] = serialize_schema(schemas[example["db_id"]])
            few_shot.append(enriched)

    rows: list[dict[str, str]] = []
    for example in examples:
        schema = schemas.get(example["db_id"], {"db_id": example["db_id"], "tables": [], "foreign_keys": []})
        prompt = build_spider_prompt(example, serialize_schema(schema), few_shot)
        answer = (example.get("gold_sql") or example.get("query") or "").strip()
        rows.append({"text": f"{prompt} {answer}"})
    return rows


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for raw_part in version.replace("-", ".").split("."):
        if raw_part.isdigit():
            parts.append(int(raw_part))
        else:
            break
    return tuple(parts)


def _validate_training_runtime() -> None:
    try:
        torchao_version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return
    if _version_tuple(torchao_version) < (0, 16, 0):
        raise ImportError(
            "Found incompatible torchao version "
            f"{torchao_version}. PEFT requires torchao>=0.16.0 when torchao is installed. "
            "Run `pip install -U torchao==0.17.0` or `pip uninstall -y torchao`, then restart the runtime."
        )


def train_lora(config_path: str | Path, max_steps: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    paths = config.get("paths", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    output_dir = ensure_dir(paths.get("output_dir", f"outputs/{config.get('experiment_name', 'experiment')}"))
    copy_config(config_path, output_dir, "training_config.yaml")
    save_environment_snapshot(output_dir, seed=seed, model_name=model_cfg.get("name"))

    spider_data_dir = Path(paths.get("data_dir", "data/processed/spider"))
    examples = load_spider_train(spider_data_dir)
    schemas = load_spider_schemas(spider_data_dir)
    dataset_rows = format_spider_for_sft(examples, schemas, int(training_cfg.get("few_shot_count", 0)))
    save_json(output_dir / "dataset_preview.json", {"num_rows": len(dataset_rows), "first_row": dataset_rows[0] if dataset_rows else None})

    if dry_run:
        logs = {"dry_run": True, "train_rows": len(dataset_rows), "max_steps": max_steps}
        save_json(output_dir / "train_logs.json", logs)
        return logs

    _validate_training_runtime()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, TrainingArguments
    from trl import SFTTrainer

    dataset = Dataset.from_list(dataset_rows)
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-3B-Instruct")
    tokenizer = load_tokenizer(model_name, cache_dir=paths.get("model_cache_dir"))

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": model_cfg.get("device_map", "auto"),
        "cache_dir": paths.get("model_cache_dir"),
        "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
    }
    quantization_config = build_quantization_config(config)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    if training_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if (config.get("training", {}).get("method") or "").lower() == "qlora" or quantization_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    lora_config = build_lora_config(config)
    common_args = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training_cfg.get("learning_rate", 1e-4)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 1)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "logging_steps": int(training_cfg.get("logging_steps", 10)),
        "save_strategy": training_cfg.get("save_strategy", "epoch"),
        "bf16": bool(training_cfg.get("bf16", False)),
        "fp16": bool(training_cfg.get("fp16", False)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", False)),
        "report_to": training_cfg.get("report_to", "none"),
        "max_steps": max_steps if max_steps is not None else int(training_cfg.get("max_steps", -1)),
    }

    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    if SFTConfig is not None:
        try:
            args = SFTConfig(
                **common_args,
                dataset_text_field="text",
                max_length=int(training_cfg.get("max_seq_length", 2048)),
            )
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                peft_config=lora_config,
                processing_class=tokenizer,
            )
        except TypeError:
            args = TrainingArguments(**common_args)
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                peft_config=lora_config,
                processing_class=tokenizer,
            )
    else:
        args = TrainingArguments(**common_args)
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            peft_config=lora_config,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=int(training_cfg.get("max_seq_length", 2048)),
        )

    started = time.perf_counter()
    result = trainer.train()
    runtime = time.perf_counter() - started
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    logs = {
        "dry_run": False,
        "train_rows": len(dataset_rows),
        "runtime_seconds": runtime,
        "train_result": getattr(result, "metrics", {}),
    }
    save_json(output_dir / "train_logs.json", logs)
    return logs
