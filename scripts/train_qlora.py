import argparse
import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import load_config
from src.model_loader import load_model_and_tokenizer
from src.metrics_io import save_metrics, log_event

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

def train(config_path: str, max_steps: int = None):
    # Load configs
    base_cfg, _ = load_config("configs/base.yaml", ["base"])
    train_cfg, config_hash = load_config(config_path, ["train", "lora", "qlora"])
    model_cfg, _ = load_config("configs/model.yaml", ["model"])
    
    output_dir = train_cfg["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Save resolved config
    resolved_config = {
        "train": train_cfg["train"],
        "lora": train_cfg["lora"],
        "qlora": train_cfg["qlora"],
        "model": model_cfg["model"]
    }
    with open(os.path.join(output_dir, 'training_config_resolved.json'), 'w') as f:
        json.dump(resolved_config, f, indent=2)
        
    print(f"Starting QLoRA fine-tuning using config: {config_path}")
    
    # Load dataset using our custom data layer
    from src.data import load_spider_train, load_spider_schemas
    from src.spider_schema import serialize_schema
    from src.prompts import render_prompt
    from datasets import Dataset
    
    try:
        raw_train = load_spider_train("data/processed/spider")
        schemas = load_spider_schemas("data/processed/spider")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load data: {e}. Run prepare_spider.py first.")
        
    formatted_data = []
    for ex in raw_train:
        schema_dict = schemas.get(ex["db_id"])
        schema_str = serialize_schema(schema_dict) if schema_dict else ""
        prompt = render_prompt(schema_str, ex["question"])
        formatted_data.append({"text": prompt + ex["query"]})
        
    dataset = Dataset.from_list(formatted_data)
    
    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_cfg, adapter_path=None)
    model.config.use_cache = False
    
    # LoRA Config
    lora_config = LoraConfig(
        r=train_cfg["lora"]["r"],
        lora_alpha=train_cfg["lora"]["alpha"],
        target_modules=train_cfg["lora"]["target_modules"],
        lora_dropout=train_cfg["lora"]["dropout"],
        bias=train_cfg["lora"]["bias"],
        task_type=train_cfg["lora"]["task_type"]
    )
    
    # Enable gradient checkpointing for VRAM savings
    if train_cfg["train"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training Arguments
    t_cfg = train_cfg["train"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=t_cfg.get("learning_rate", 2e-4),
        num_train_epochs=t_cfg.get("num_train_epochs", 1),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
        weight_decay=t_cfg.get("weight_decay", 0.0),
        logging_steps=t_cfg.get("logging_steps", 10),
        save_strategy=t_cfg.get("save_strategy", "epoch"),
        bf16=t_cfg.get("bf16", True),
        fp16=t_cfg.get("fp16", False),
        optim=t_cfg.get("optim", "paged_adamw_8bit"),
        max_steps=max_steps if max_steps else -1,
        report_to="none"
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=t_cfg.get("max_seq_length", 2048),
        tokenizer=tokenizer,
        args=training_args,
    )
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    # Save adapter
    adapter_dir = os.path.join(output_dir, "adapters")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    
    metrics = {
        "training_runtime_seconds": end_time - start_time,
        "num_train_epochs": t_cfg.get("num_train_epochs", 1)
    }
    save_metrics(output_dir, "train_metrics.json", metrics)
    log_event("train_qlora", metrics, config_hash)
    print("Training complete! Adapters saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    
    train(args.config, args.max_steps)
