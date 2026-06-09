import argparse
import sys
import os
import json
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import load_config
from src.reproducibility import set_global_seed

def prepare_mmlu(config_path: str, limit: int = None):
    config, _ = load_config(config_path, ["mmlu"])
    mmlu_cfg = config["mmlu"]
    
    set_global_seed(mmlu_cfg["seed"])
    
    output_dir = "outputs/mmlu"
    os.makedirs(output_dir, exist_ok=True)
    
    questions = []
    
    for category, cat_data in mmlu_cfg["categories"].items():
        subcategory = cat_data["subcategory"]
        count = cat_data["eval_count"]
        if limit:
            count = min(count, limit)
            
        for i in range(count):
            questions.append({
                "id": f"mmlu_{category}_{i}",
                "category": category,
                "subcategory": subcategory,
                "question": f"Question {i} about {subcategory}?",
                "choices": ["A choice", "B choice", "C choice", "D choice"],
                "answer": "A"
            })
            
    dataset_content = "\n".join(json.dumps(q) for q in questions)
    dataset_hash = hashlib.sha256(dataset_content.encode('utf-8')).hexdigest()
    
    suite_path = os.path.join(output_dir, "mmlu_suite.jsonl")
    with open(suite_path, 'w', encoding='utf-8') as f:
        f.write(dataset_content + "\n")
        
    print(f"Saved {len(questions)} MMLU questions to {suite_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    prepare_mmlu(args.config, args.limit)
