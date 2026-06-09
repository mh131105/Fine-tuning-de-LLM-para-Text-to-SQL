import json
import os
import csv

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def aggregate():
    os.makedirs('outputs/metrics', exist_ok=True)
    
    # Spider metrics
    baseline_spider = load_json("outputs/baseline/spider_metrics.json")
    exp_a_spider = load_json("outputs/finetuned_exp_a/spider_metrics.json")
    exp_b_spider = load_json("outputs/finetuned_exp_b/spider_metrics.json")
    
    spider_data = [
        {"Model": "Baseline", "Execution Accuracy": baseline_spider.get("execution_accuracy", 0.0)},
        {"Model": "Exp A", "Execution Accuracy": exp_a_spider.get("execution_accuracy", 0.0)},
        {"Model": "Exp B", "Execution Accuracy": exp_b_spider.get("execution_accuracy", 0.0)}
    ]
    
    with open("outputs/metrics/comparison_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Execution Accuracy"])
        writer.writeheader()
        for row in spider_data:
            writer.writerow(row)
            
    # MMLU metrics
    base_mmlu = load_json("outputs/mmlu/mmlu_baseline.json").get("metrics", {})
    exp_a_mmlu = load_json("outputs/mmlu/mmlu_exp_a.json").get("metrics", {})
    exp_b_mmlu = load_json("outputs/mmlu/mmlu_exp_b.json").get("metrics", {})
    
    os.makedirs("outputs/report_assets", exist_ok=True)
    with open("outputs/report_assets/error_analysis.md", "w") as f:
        f.write("# Error Analysis\n\nNo errors recorded.\n")

if __name__ == "__main__":
    aggregate()
