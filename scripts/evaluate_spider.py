import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import load_config
from src.model_loader import load_model_and_tokenizer
from src.generation import generate_text
from src.sql_utils import extract_sql
from src.metrics_io import save_prediction, save_metrics
from custom_metrics.execution_accuracy import ExecutionAccuracyMetric
from deepeval.test_case import LLMTestCase

def evaluate(config_path: str, adapter_path: str = None, output_dir: str = None, limit: int = None):
    # Load configs
    base_cfg, _ = load_config("configs/base.yaml", ["base"])
    eval_cfg, _ = load_config(config_path, ["evaluation", "generation"])
    model_cfg, _ = load_config("configs/model.yaml", ["model"])
    data_cfg, _ = load_config("configs/data.yaml", ["data"])
    
    if not output_dir:
        output_dir = eval_cfg["evaluation"]["output_dir"]
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    combined_cfg = {"model": model_cfg["model"]}
    model, tokenizer = load_model_and_tokenizer(combined_cfg, adapter_path)
    
    # Read eval data using new custom data layer
    from src.data import load_spider_dev, load_spider_schemas
    from src.spider_schema import serialize_schema
    from src.prompts import render_prompt
    
    split = eval_cfg["evaluation"]["split"]
    if split != "dev":
        print(f"Warning: split is {split}, fallback to dev.")
    
    try:
        raw_dev = load_spider_dev("data/processed/spider")
        schemas = load_spider_schemas("data/processed/spider")
    except Exception as e:
        print(f"Eval file not found: {e}")
        return
        
    examples = []
    for ex in raw_dev:
        schema_dict = schemas.get(ex["db_id"])
        schema_str = serialize_schema(schema_dict) if schema_dict else ""
        prompt = render_prompt(schema_str, ex["question"])
        examples.append({
            "db_id": ex["db_id"],
            "question": ex["question"],
            "prompt": prompt,
            "gold_sql": ex["query"],
            "db_path": f"data/raw/spider/database/{ex['db_id']}/{ex['db_id']}.sqlite"
        })
                
    if limit is None:
        limit = eval_cfg["evaluation"].get("limit")
    if limit:
        examples = examples[:limit]
        
    metric = ExecutionAccuracyMetric()
    correct = 0
    failed = 0
    invalid_sql = 0
    
    # Remove old predictions file if exists
    preds_file = os.path.join(output_dir, "spider_predictions.jsonl")
    if os.path.exists(preds_file):
        os.remove(preds_file)
    
    for ex in examples:
        prompt = ex["prompt"]
        raw_output, latency = generate_text(model, tokenizer, prompt, eval_cfg["generation"])
        
        extracted_sql = extract_sql(raw_output)
        if not extracted_sql or (extracted_sql == raw_output.strip() and "SELECT" not in extracted_sql.upper()):
            invalid_sql += 1
            
        test_case = LLMTestCase(
            input=ex["question"],
            actual_output=extracted_sql,
            expected_output=ex["gold_sql"],
            additional_metadata={"db_path": ex["db_path"]}
        )
        
        score = metric.measure(test_case)
        
        if metric.is_successful():
            correct += 1
            error_type = "NONE"
        else:
            failed += 1
            error_type = metric.reason
            
        record = {
            "db_id": ex["db_id"],
            "question": ex["question"],
            "gold_sql": ex["gold_sql"],
            "predicted_raw": raw_output,
            "predicted_sql": extracted_sql,
            "error_type": error_type,
            "latency_seconds": latency,
            "score": score
        }
        
        save_prediction(output_dir, "spider_predictions.jsonl", record)
        
    # Aggregate
    total = len(examples)
    acc = correct / total if total > 0 else 0.0
    
    metrics = {
        "total_examples": total,
        "correct_examples": correct,
        "failed_examples": failed,
        "execution_accuracy": acc,
        "invalid_sql_rate": invalid_sql / total if total > 0 else 0.0
    }
    
    save_metrics(output_dir, "spider_metrics.json", metrics)
    print(f"Evaluation complete. Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    evaluate(args.config, args.adapter_path, args.output_dir, args.limit)
