import json
import os
import time

def log_event(step: str, payload: dict, config_hash: str = None):
    os.makedirs('outputs/metrics', exist_ok=True)
    log_path = 'outputs/metrics/pipeline_events.jsonl'
    event = {
        "timestamp": time.time(),
        "step": step,
        "config_hash": config_hash,
        "payload": payload
    }
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(event) + '\n')
        f.flush()

def save_prediction(output_dir: str, filename: str, record: dict):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
        f.flush()

def save_metrics(output_dir: str, filename: str, metrics: dict):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

def save_prompt_sample(output_dir: str, prompt_str: str, template_hash: str):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'prompt_samples.md')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"# Prompt Sample\n\n**Hash:** {template_hash}\n\n```text\n{prompt_str}\n```\n")
