import argparse
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import load_config
from src.model_loader import load_model_and_tokenizer
from src.generation import generate_text
from src.prompts import render_mmlu_prompt

def evaluate_mmlu(config_path: str, model_id: str, adapter_path: str, output_path: str, limit: int = None):
    config, _ = load_config(config_path, ["mmlu", "generation"])
    
    model_cfg, _ = load_config("configs/model.yaml", ["model"])
    model, tokenizer = load_model_and_tokenizer({"model": model_cfg["model"]}, adapter_path)
    
    suite_path = "outputs/mmlu/mmlu_suite.jsonl"
    if not os.path.exists(suite_path):
        print("MMLU suite not found. Run prepare_mmlu.py first.")
        return
        
    questions = []
    with open(suite_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
                
    if limit:
        questions = questions[:limit]
        
    correct = 0
    total = len(questions)
    
    results = []
    
    for q in questions:
        prompt = render_mmlu_prompt(q["subcategory"], q["question"], q["choices"])
        
        raw_output, latency = generate_text(model, tokenizer, prompt, config["generation"])
        
        parsed = "A" # mock parse
        if "A" in raw_output: parsed = "A"
        elif "B" in raw_output: parsed = "B"
        elif "C" in raw_output: parsed = "C"
        elif "D" in raw_output: parsed = "D"
        
        is_correct = parsed == q["answer"]
        if is_correct:
            correct += 1
            
        q["predicted_raw"] = raw_output
        q["predicted"] = parsed
        q["is_correct"] = is_correct
        results.append(q)
        
    acc = correct / total if total > 0 else 0.0
    
    metrics = {
        "model_id": model_id,
        "total": total,
        "correct": correct,
        "accuracy": acc
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    evaluate_mmlu(args.config, args.model_id, args.adapter_path, args.output_path, args.limit)
