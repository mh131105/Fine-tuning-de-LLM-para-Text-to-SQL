from __future__ import annotations

import time
import os
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import copy_config, ensure_dir, load_yaml
from .data import load_spider_schemas, load_spider_train, save_json, serialize_schema
from .logging_utils import save_environment_snapshot
from .model import _torch_dtype, build_lora_config, build_quantization_config, load_tokenizer
from .prompts import build_spider_prompt
from .reproducibility import set_global_seed


def _completion_with_eos(answer: str, eos_token: str | None) -> str:
    completion = f" {answer.strip()}" if answer.strip() else ""
    if eos_token and not completion.endswith(eos_token):
        completion = f"{completion}{eos_token}"
    return completion


def format_spider_for_sft(
    examples: list[dict[str, Any]],
    schemas: dict[str, Any],
    few_shot_count: int = 0,
    eos_token: str | None = None,
) -> list[dict[str, str]]:
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
        rows.append({"prompt": prompt, "completion": _completion_with_eos(answer, eos_token)})
    return rows


def tokenize_sft_rows(
    rows: list[dict[str, str]],
    tokenizer: Any,
    max_length: int | None = None,
) -> list[dict[str, list[int]]]:
    tokenized_rows: list[dict[str, list[int]]] = []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + completion_ids
        completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        if max_length is not None:
            input_ids = input_ids[:max_length]
            completion_mask = completion_mask[:max_length]
        tokenized_rows.append({"input_ids": input_ids, "completion_mask": completion_mask})
    return tokenized_rows


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


def _validate_effective_batch_size(training_cfg: dict[str, Any]) -> None:
    if "effective_batch_size" not in training_cfg:
        return
    batch_size = int(training_cfg.get("per_device_train_batch_size", 1))
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    expected = batch_size * accumulation
    configured = int(training_cfg["effective_batch_size"])
    if configured != expected:
        raise ValueError(
            "invalid_config_schema: training.effective_batch_size must equal "
            "per_device_train_batch_size * gradient_accumulation_steps "
            f"({configured} != {batch_size} * {accumulation})"
        )


def _training_eos_token(training_cfg: dict[str, Any], tokenizer: Any | None = None) -> str | None:
    configured = training_cfg.get("eos_token")
    if configured is not None:
        return str(configured)
    if tokenizer is not None:
        return getattr(tokenizer, "eos_token", None)
    return "<|im_end|>"


def _configure_torch_performance(training_cfg: dict[str, Any]) -> None:
    try:
        import torch
    except ImportError:
        return
    if bool(training_cfg.get("tf32", False)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    precision = training_cfg.get("float32_matmul_precision")
    if precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(precision))


def train_lora(config_path: str | Path, max_steps: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    paths = config.get("paths", {})
    training_cfg = config.get("training", {})
    _validate_effective_batch_size(training_cfg)
    model_cfg = config.get("model", {})
    output_dir = ensure_dir(paths.get("output_dir", f"outputs/{config.get('experiment_name', 'experiment')}"))
    copy_config(config_path, output_dir, "training_config.yaml")
    save_environment_snapshot(output_dir, seed=seed, model_name=model_cfg.get("name"))

    spider_data_dir = Path(paths.get("data_dir", "data/processed/spider"))
    examples = load_spider_train(spider_data_dir)
    schemas = load_spider_schemas(spider_data_dir)

    if dry_run:
        dataset_rows = format_spider_for_sft(
            examples,
            schemas,
            int(training_cfg.get("few_shot_count", 0)),
            eos_token=_training_eos_token(training_cfg),
        )
        save_json(output_dir / "dataset_preview.json", {"num_rows": len(dataset_rows), "first_row": dataset_rows[0] if dataset_rows else None})
        logs = {"dry_run": True, "train_rows": len(dataset_rows), "max_steps": max_steps}
        save_json(output_dir / "train_logs.json", logs)
        return logs

    _validate_training_runtime()
    _configure_torch_performance(training_cfg)

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, TrainingArguments
    from trl import SFTTrainer

    model_name = model_cfg.get("name", "Qwen/Qwen2.5-3B-Instruct")
    tokenizer = load_tokenizer(model_name, cache_dir=paths.get("model_cache_dir"))
    eos_token = _training_eos_token(training_cfg, tokenizer)
    dataset_rows = format_spider_for_sft(
        examples,
        schemas,
        int(training_cfg.get("few_shot_count", 0)),
        eos_token=eos_token,
    )
    save_json(output_dir / "dataset_preview.json", {"num_rows": len(dataset_rows), "first_row": dataset_rows[0] if dataset_rows else None})
    tokenized_rows = tokenize_sft_rows(
        dataset_rows,
        tokenizer,
        max_length=int(training_cfg.get("max_seq_length", 2048)),
    )
    dataset = Dataset.from_list(tokenized_rows)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": model_cfg.get("device_map", "auto"),
        "cache_dir": paths.get("model_cache_dir"),
        "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
    }
    torch_dtype = _torch_dtype(model_cfg.get("torch_dtype"))
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
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
    run_name = training_cfg.get("run_name") or config.get("run_name")
    if run_name:
        common_args["run_name"] = str(run_name)
    if "save_total_limit" in training_cfg:
        common_args["save_total_limit"] = int(training_cfg["save_total_limit"])
    if "load_best_model_at_end" in training_cfg:
        common_args["load_best_model_at_end"] = bool(training_cfg["load_best_model_at_end"])
    if "optim" in training_cfg:
        common_args["optim"] = str(training_cfg["optim"])
    if "tf32" in training_cfg:
        common_args["tf32"] = bool(training_cfg["tf32"])
    if "dataloader_num_workers" in training_cfg:
        common_args["dataloader_num_workers"] = int(training_cfg["dataloader_num_workers"])
    if "dataloader_pin_memory" in training_cfg:
        common_args["dataloader_pin_memory"] = bool(training_cfg["dataloader_pin_memory"])
    if "dataloader_prefetch_factor" in training_cfg:
        common_args["dataloader_prefetch_factor"] = int(training_cfg["dataloader_prefetch_factor"])
    if "group_by_length" in training_cfg:
        common_args["group_by_length"] = bool(training_cfg["group_by_length"])

    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    if SFTConfig is not None:
        try:
            args = SFTConfig(
                **common_args,
                max_length=int(training_cfg.get("max_seq_length", 2048)),
                completion_only_loss=bool(training_cfg.get("completion_only_loss", True)),
                eos_token=eos_token,
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
