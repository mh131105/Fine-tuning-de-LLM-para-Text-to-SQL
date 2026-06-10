from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from .config import load_yaml
from .data_sources import SpiderSource, ensure_spider_raw_data
from .data import (
    convert_spider_table_schema,
    deterministic_sample,
    load_json,
    normalize_mmlu_record,
    save_json,
    save_jsonl,
    spider_rows_to_examples,
)
from .reproducibility import set_global_seed


def prepare_spider(
    data_dir: str | Path,
    output_dir: str | Path,
    source: SpiderSource = "auto",
    source_path: str | Path | None = None,
    hf_repo: str = "dreamerdeo/multispider",
    cache_dir: str | Path = ".cache/datasets/spider",
    force_download: bool = False,
) -> dict[str, Any]:
    source_result = ensure_spider_raw_data(
        data_dir,
        source=source,
        source_path=source_path,
        hf_repo=hf_repo,
        cache_dir=cache_dir,
        force=force_download,
    )
    raw_dir = Path(data_dir)
    output = Path(output_dir)
    train_path = raw_dir / "train_spider.json"
    dev_path = raw_dir / "dev.json"
    tables_path = raw_dir / "tables.json"
    for path in [train_path, dev_path, tables_path]:
        if not path.exists():
            raise FileNotFoundError(f"spider_file_not_found: expected {path}")

    train_rows = load_json(train_path)
    dev_rows = load_json(dev_path)
    table_rows = load_json(tables_path)
    train_examples = spider_rows_to_examples(train_rows, "train")
    dev_examples = spider_rows_to_examples(dev_rows, "dev")
    schemas = {schema["db_id"]: schema for schema in (convert_spider_table_schema(row) for row in table_rows)}

    missing_databases: list[str] = []
    unreadable_databases: list[str] = []
    for db_id in sorted(schemas):
        db_path = raw_dir / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            missing_databases.append(db_id)
            continue
        try:
            connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
            connection.close()
        except sqlite3.Error:
            unreadable_databases.append(db_id)

    output.mkdir(parents=True, exist_ok=True)
    save_jsonl(output / "train.jsonl", train_examples)
    save_jsonl(output / "dev.jsonl", dev_examples)
    save_json(output / "schemas.json", schemas)
    metadata = {
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "schemas": len(schemas),
        "missing_databases": missing_databases,
        "unreadable_databases": unreadable_databases,
        "data_source": {
            "source": source_result.source,
            "source_path": str(source_result.source_path) if source_result.source_path else None,
            "message": source_result.message,
        },
    }
    save_json(output / "metadata.json", metadata)
    return metadata


def _load_hf_mmlu_split(dataset_name: str, subcategory: str, split: str):
    from datasets import load_dataset

    try:
        return load_dataset(dataset_name, subcategory, split=split)
    except Exception:
        return load_dataset(dataset_name, name=subcategory, split=split)


def _mock_mmlu_rows(count: int, category: str, subcategory: str) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "question": f"Mock question {index} for {subcategory}?",
                "choices": ["Option A", "Option B", "Option C", "Option D"],
                "answer": index % 4,
            }
        )
    return rows


def prepare_mmlu(config_path: str | Path, mock: bool = False, limit_per_category: int | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    paths = config.get("paths", {})
    mmlu_cfg = config.get("mmlu", {})
    categories = mmlu_cfg.get("categories", {})
    if not categories:
        raise ValueError("invalid_config_schema: mmlu.categories is required")

    output_path = Path(paths.get("mmlu_suite_path", "data/processed/mmlu_150_suite.json"))
    dataset_name = mmlu_cfg.get("dataset_name", "cais/mmlu")
    eval_split = mmlu_cfg.get("eval_split", "test")
    few_shot_split = mmlu_cfg.get("few_shot_split", "dev")
    few_shot_count = int(mmlu_cfg.get("few_shot_count", 5))

    questions: list[dict[str, Any]] = []
    few_shot_examples: dict[str, list[dict[str, Any]]] = {}

    for category_index, (category, category_cfg) in enumerate(categories.items()):
        subcategory = category_cfg["subcategory"]
        count = int(category_cfg.get("count", category_cfg.get("eval_count", 50)))
        if limit_per_category is not None:
            count = min(count, limit_per_category)

        if mock:
            eval_rows = _mock_mmlu_rows(count + few_shot_count, category, subcategory)
            shot_rows = _mock_mmlu_rows(few_shot_count, category, subcategory)
        else:
            eval_rows = list(_load_hf_mmlu_split(dataset_name, subcategory, eval_split))
            shot_rows = list(_load_hf_mmlu_split(dataset_name, subcategory, few_shot_split))

        sampled = deterministic_sample(
            [
                normalize_mmlu_record(row, category, subcategory, index)
                for index, row in enumerate(eval_rows)
            ],
            count,
            seed + category_index,
        )
        for index, question in enumerate(sampled):
            question["question_id"] = f"mmlu-{category.lower()}-{index:06d}"
        questions.extend(sampled)

        normalized_shots = [
            normalize_mmlu_record(row, category, subcategory, index)
            for index, row in enumerate(shot_rows)
        ]
        few_shot_examples[subcategory] = deterministic_sample(
            normalized_shots,
            few_shot_count,
            seed + 1000 + category_index,
        )

    serialized_questions = "\n".join(
        hashlib.sha256(str(question).encode("utf-8")).hexdigest() for question in questions
    )
    suite = {
        "metadata": {
            "dataset_name": dataset_name,
            "seed": seed,
            "total_questions": len(questions),
            "question_hash": hashlib.sha256(serialized_questions.encode("utf-8")).hexdigest(),
            "mock": mock,
        },
        "questions": questions,
        "few_shot_examples": few_shot_examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, suite)
    return suite["metadata"]
