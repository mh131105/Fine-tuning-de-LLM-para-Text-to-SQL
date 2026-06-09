import json
import sqlite3

from src.tp2.evaluation import run_benchmarks
from src.tp2.preparation import prepare_spider


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_prepare_spider_and_mock_benchmark(tmp_path):
    raw = tmp_path / "raw" / "spider"
    db_dir = raw / "database" / "toy"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "toy.sqlite")
    conn.execute("CREATE TABLE singer (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO singer VALUES (1, 'Ana')")
    conn.commit()
    conn.close()

    _write_json(raw / "train_spider.json", [{"db_id": "toy", "question": "Who?", "query": "SELECT name FROM singer"}])
    _write_json(raw / "dev.json", [{"db_id": "toy", "question": "Who?", "query": "SELECT name FROM singer"}])
    _write_json(
        raw / "tables.json",
        [
            {
                "db_id": "toy",
                "table_names_original": ["singer"],
                "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
                "column_types": ["text", "number", "text"],
                "primary_keys": [1],
                "foreign_keys": [],
            }
        ],
    )

    processed = tmp_path / "processed" / "spider"
    metadata = prepare_spider(raw, processed)
    assert metadata["train_examples"] == 1

    mmlu_suite = {
        "metadata": {"mock": True},
        "questions": [
            {
                "question_id": "mmlu-stem-000000",
                "category": "STEM",
                "subcategory": "college_computer_science",
                "question": "2 + 2?",
                "choices": {"A": "3", "B": "4", "C": "5", "D": "6"},
                "answer": "B",
            }
        ],
        "few_shot_examples": {"college_computer_science": []},
    }
    suite_path = tmp_path / "mmlu_suite.json"
    _write_json(suite_path, mmlu_suite)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        f"""
seed: 42
paths:
  spider_data_dir: {processed}
  spider_db_dir: {raw / "database"}
  mmlu_suite_path: {suite_path}
  baseline_dir: {tmp_path / "outputs" / "base"}
model:
  name: Qwen/Qwen2.5-3B-Instruct
spider:
  few_shot_count: 1
mmlu:
  few_shot_count: 5
generation:
  temperature: 0.0
  do_sample: false
  max_new_tokens: 16
execution:
  sqlite_timeout_seconds: 5
""",
        encoding="utf-8",
    )
    summary = run_benchmarks(config_path, tmp_path / "outputs" / "base", mock=True)
    assert summary["spider"]["execution_accuracy"] == 1.0
    assert summary["mmlu"]["accuracy"] == 1.0
    assert (tmp_path / "outputs" / "base" / "summary.md").exists()
