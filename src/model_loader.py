import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_tokenizer(config: dict, adapter_path: str = None):
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-Coder-3B-Instruct")
    use_4bit = model_cfg.get("load_in_4bit", False)
    
    bnb_config = None
    if use_4bit:
        qlora_cfg = config.get("qlora", {})
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True)
        )
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Colab GPU or CPU fallback
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if use_4bit and torch.cuda.is_available() else None,
            device_map=device_map,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: Could not load model {model_name}. Reason: {e}")
        model = None
        
    if adapter_path and model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        
    # Save metadata
    os.makedirs('outputs/metrics', exist_ok=True)
    meta = {
        "model_name": model_name,
        "adapter_path": adapter_path,
        "use_4bit": use_4bit
    }
    with open('outputs/metrics/model_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
        
    return model, tokenizer
