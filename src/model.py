from __future__ import annotations

from pathlib import Path
from typing import Any


def _torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    import torch

    lookup = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return lookup.get(str(dtype_name).lower())


def load_tokenizer(model_name_or_path: str, cache_dir: str | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_quantization_config(config: dict[str, Any]):
    quant_cfg = config.get("quantization") or {}
    model_cfg = config.get("model") or {}
    if not quant_cfg and model_cfg.get("quantization") != "qlora_4bit":
        return None
    if not quant_cfg.get("load_in_4bit", True):
        return None
    import torch
    from transformers import BitsAndBytesConfig

    dtype = quant_cfg.get("bnb_4bit_compute_dtype", "float16")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=getattr(torch, dtype),
    )


def load_base_model(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM

    model_cfg = config.get("model") or {}
    paths_cfg = config.get("paths") or {}
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-3B-Instruct")
    cache_dir = paths_cfg.get("model_cache_dir")
    dtype = _torch_dtype(model_cfg.get("torch_dtype"))
    if dtype is None:
        dtype = torch.bfloat16 if model_cfg.get("bf16", False) else None
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation"),
        quantization_config=build_quantization_config(config),
    )


def load_model_with_adapter(config: dict[str, Any], adapter_path: str | Path):
    from peft import PeftModel

    model = load_base_model(config)
    return PeftModel.from_pretrained(model, str(adapter_path))


def find_adapter_path(model_path: str | Path | None) -> Path | None:
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.exists():
        return None
    candidates = [path / "adapter", path / "adapters", path]
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate
    return None


def build_lora_config(config: dict[str, Any]):
    from peft import LoraConfig

    lora_cfg = config.get("lora") or {}
    return LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", lora_cfg.get("alpha", 32))),
        lora_dropout=float(lora_cfg.get("lora_dropout", lora_cfg.get("dropout", 0.05))),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )


def load_for_inference(config: dict[str, Any], model_path: str | Path | None = None):
    model_cfg = config.get("model") or {}
    paths_cfg = config.get("paths") or {}
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-3B-Instruct")
    tokenizer = load_tokenizer(model_name, cache_dir=paths_cfg.get("model_cache_dir"))
    adapter_path = find_adapter_path(model_path)
    if adapter_path:
        model = load_model_with_adapter(config, adapter_path)
    else:
        model = load_base_model(config)
    model.eval()
    return model, tokenizer
