from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from custom_metrics.execution_accuracy import ExecutionAccuracy

from .config import ensure_dir, load_yaml
from .data import (
    load_mmlu_suite,
    load_spider_dev,
    load_spider_schemas,
    load_spider_train,
    read_jsonl,
    save_json,
    save_jsonl,
    serialize_schema,
)
from .inference import extract_mmlu_answer, generate_text
from .logging_utils import save_environment_snapshot, write_summary_markdown
from .model import load_for_inference
from .prompts import build_mmlu_prompt, build_spider_prompt, prompt_hash
from .reproducibility import set_global_seed


def _progress(items: list[dict[str, Any]], desc: str, unit: str = "example"):
    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc=desc, total=len(items), unit=unit, dynamic_ncols=True)
    except Exception:
        return items


def _load_model_for_eval(config: dict[str, Any], model_path: str | Path | None, mock: bool):
    if mock:
        return None, None
    return load_for_inference(config, model_path)


def _model_output_dir(config: dict[str, Any], model_path: str | Path | None, output_dir: str | Path | None) -> Path:
    if output_dir:
        return Path(output_dir)
    if model_path:
        return Path(model_path)
    return Path(config.get("paths", {}).get("baseline_dir", "outputs/base"))


def _few_shot_spider(train_examples: list[dict[str, Any]], schemas: dict[str, Any], count: int) -> list[dict[str, Any]]:
    shots = []
    for example in train_examples[:count]:
        schema = schemas.get(example["db_id"], {"db_id": example["db_id"], "tables": [], "foreign_keys": []})
        enriched = dict(example)
        enriched["schema_text"] = serialize_schema(schema)
        shots.append(enriched)
    return shots


def evaluate_spider(
    config_path: str | Path,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    output = ensure_dir(_model_output_dir(config, model_path, output_dir))
    paths = config.get("paths", {})
    spider_cfg = config.get("spider", {})
    execution_cfg = config.get("execution", {})
    generation_cfg = dict(config.get("generation", {}))
    generation_cfg.update({key: value for key, value in spider_cfg.items() if key in {"max_new_tokens"}})

    model, tokenizer = _load_model_for_eval(config, model_path, mock)
    save_environment_snapshot(output, seed=seed, model_name=config.get("model", {}).get("name"))

    spider_data_dir = Path(paths.get("spider_data_dir", "data/processed/spider"))
    spider_db_dir = Path(paths.get("spider_db_dir", "data/raw/spider/database"))
    dev_examples = load_spider_dev(spider_data_dir)
    train_examples = load_spider_train(spider_data_dir)
    schemas = load_spider_schemas(spider_data_dir)
    few_shot = _few_shot_spider(train_examples, schemas, int(spider_cfg.get("few_shot_count", 3)))
    if limit is not None:
        dev_examples = dev_examples[:limit]
    elif spider_cfg.get("limit"):
        dev_examples = dev_examples[: int(spider_cfg["limit"])]

    metric = ExecutionAccuracy(spider_db_dir=spider_db_dir, timeout_seconds=int(execution_cfg.get("sqlite_timeout_seconds", 5)))
    predictions: list[dict[str, Any]] = []
    correct = 0
    failed = 0
    error_breakdown: dict[str, int] = {}

    for example in _progress(dev_examples, "Spider eval"):
        schema = schemas.get(example["db_id"], {"db_id": example["db_id"], "tables": [], "foreign_keys": []})
        schema_text = serialize_schema(schema)
        prompt = build_spider_prompt(example, schema_text, few_shot)
        if mock:
            raw_output = example.get("mock_output") or example.get("gold_sql") or example.get("query")
            latency = 0.0
        else:
            raw_output, latency = generate_text(model, tokenizer, prompt, generation_cfg)
        test_case = SimpleNamespace(
            input=example["question"],
            actual_output=raw_output,
            expected_output=example.get("gold_sql") or example.get("query"),
            additional_metadata={"db_id": example["db_id"]},
        )
        score = metric.measure(test_case)
        if score >= 1.0:
            correct += 1
        else:
            failed += 1
            error_type = metric.error_type or "unknown_error"
            error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
        predictions.append(
            {
                "example_id": example["example_id"],
                "db_id": example["db_id"],
                "question": example["question"],
                "prompt_hash": prompt_hash(prompt),
                "gold_sql": example.get("gold_sql") or example.get("query"),
                "raw_output": raw_output,
                "predicted_sql": metric.predicted_sql,
                "score": score,
                "error_type": metric.error_type,
                "error_message": metric.error_message,
                "latency_seconds": latency,
            }
        )

    total = len(dev_examples)
    metrics = {
        "execution_accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "failed": failed,
        "error_breakdown": error_breakdown,
    }
    save_jsonl(output / "spider_predictions.jsonl", predictions)
    save_json(output / "spider_metrics.json", metrics)
    return metrics


def _mmlu_shots(question: dict[str, Any], few_shot_examples: dict[str, list[dict[str, Any]]], count: int) -> list[dict[str, Any]]:
    candidates = few_shot_examples.get(question.get("subcategory"), [])
    return candidates[:count]


def evaluate_mmlu(
    config_path: str | Path,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    output = ensure_dir(_model_output_dir(config, model_path, output_dir))
    paths = config.get("paths", {})
    mmlu_cfg = config.get("mmlu", {})
    generation_cfg = dict(config.get("generation", {}))
    generation_cfg.update({key: value for key, value in mmlu_cfg.items() if key in {"max_new_tokens"}})
    model, tokenizer = _load_model_for_eval(config, model_path, mock)
    save_environment_snapshot(output, seed=seed, model_name=config.get("model", {}).get("name"))

    questions, few_shot_examples = load_mmlu_suite(paths.get("mmlu_suite_path", "data/processed/mmlu_150_suite.json"))
    if limit is not None:
        questions = questions[:limit]
    elif mmlu_cfg.get("limit"):
        questions = questions[: int(mmlu_cfg["limit"])]

    predictions: list[dict[str, Any]] = []
    correct = 0
    by_category: dict[str, dict[str, Any]] = {}
    for question in _progress(questions, "MMLU eval", unit="question"):
        category = question["category"]
        by_category.setdefault(category, {"total": 0, "correct": 0, "accuracy": 0.0})
        by_category[category]["total"] += 1
        prompt = build_mmlu_prompt(question, _mmlu_shots(question, few_shot_examples, int(mmlu_cfg.get("few_shot_count", 5))))
        if mock:
            raw_output = question["answer"]
            latency = 0.0
        else:
            raw_output, latency = generate_text(model, tokenizer, prompt, generation_cfg)
        parsed = extract_mmlu_answer(raw_output)
        is_correct = parsed == question["answer"]
        if is_correct:
            correct += 1
            by_category[category]["correct"] += 1
        predictions.append(
            {
                "question_id": question["question_id"],
                "category": category,
                "subcategory": question["subcategory"],
                "raw_output": raw_output,
                "parsed_answer": parsed,
                "gold_answer": question["answer"],
                "is_correct": is_correct,
                "error_type": None if parsed else "mmlu_parse_error",
                "latency_seconds": latency,
                "prompt_hash": prompt_hash(prompt),
            }
        )
    total = len(questions)
    for values in by_category.values():
        values["accuracy"] = values["correct"] / values["total"] if values["total"] else 0.0
    metrics = {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct, "by_category": by_category}
    save_jsonl(output / "mmlu_predictions.jsonl", predictions)
    save_json(output / "mmlu_metrics.json", metrics)
    return metrics


def _read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def calculate_deltas(output_dir: str | Path, baseline_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    baseline = Path(baseline_dir)
    spider = _read_metrics(output / "spider_metrics.json") or {}
    mmlu = _read_metrics(output / "mmlu_metrics.json") or {}
    base_spider = _read_metrics(baseline / "spider_metrics.json") or {}
    base_mmlu = _read_metrics(baseline / "mmlu_metrics.json") or {}
    spider_delta = spider.get("execution_accuracy", 0.0) - base_spider.get("execution_accuracy", 0.0)
    mmlu_delta = mmlu.get("accuracy", 0.0) - base_mmlu.get("accuracy", 0.0)
    mmlu_base = base_mmlu.get("accuracy", 0.0)
    return {
        "spider_accuracy_delta": spider_delta,
        "mmlu_accuracy_delta": mmlu_delta,
        "mmlu_accuracy_percent_delta": (mmlu_delta / mmlu_base * 100.0) if mmlu_base else None,
        "baseline_dir": str(baseline),
    }


def run_benchmarks(
    config_path: str | Path,
    model_path: str | Path,
    limit: int | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    output = ensure_dir(model_path)
    spider_metrics = evaluate_spider(config_path, model_path=model_path, output_dir=output, limit=limit, mock=mock)
    mmlu_metrics = evaluate_mmlu(config_path, model_path=model_path, output_dir=output, limit=limit, mock=mock)
    deltas = calculate_deltas(output, config.get("paths", {}).get("baseline_dir", "outputs/base"))
    save_json(output / "benchmark_summary.json", {"spider": spider_metrics, "mmlu": mmlu_metrics, "delta_vs_base": deltas})
    write_summary_markdown(output, spider_metrics, mmlu_metrics, deltas)
    return {"spider": spider_metrics, "mmlu": mmlu_metrics, "delta_vs_base": deltas}


def validate_output_jsonl(path: str | Path) -> int:
    return len(read_jsonl(path))
