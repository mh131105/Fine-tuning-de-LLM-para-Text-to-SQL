import json
import sqlite3

from src import evaluation


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluate_mmlu_uses_configured_eval_batch_size(tmp_path, monkeypatch):
    suite_path = tmp_path / "mmlu_suite.json"
    _write_json(
        suite_path,
        {
            "metadata": {"mock": True},
            "questions": [
                {
                    "question_id": f"mmlu-stem-{index:06d}",
                    "category": "STEM",
                    "subcategory": "college_computer_science",
                    "question": f"Question {index}?",
                    "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
                    "answer": "B",
                }
                for index in range(3)
            ],
            "few_shot_examples": {"college_computer_science": []},
        },
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        f"""
seed: 42
paths:
  mmlu_suite_path: {suite_path}
  baseline_dir: {tmp_path / "outputs" / "base"}
model:
  name: Qwen/Qwen2.5-3B-Instruct
mmlu:
  few_shot_count: 0
  eval_batch_size: 2
generation:
  temperature: 0.0
  do_sample: false
  max_new_tokens: 4
""",
        encoding="utf-8",
    )
    batch_sizes = []

    monkeypatch.setattr(evaluation, "_load_model_for_eval", lambda *_args, **_kwargs: (object(), object()))

    def fake_generate_text_batch(_model, _tokenizer, prompts, _generation_cfg):
        batch_sizes.append(len(prompts))
        return ["B"] * len(prompts), [0.0] * len(prompts)

    monkeypatch.setattr(evaluation, "generate_text_batch", fake_generate_text_batch)

    metrics = evaluation.evaluate_mmlu(config_path, model_path=tmp_path / "outputs" / "base")

    assert batch_sizes == [2, 1]
    assert metrics["accuracy"] == 1.0
    assert metrics["total"] == 3


def test_evaluate_spider_passes_stop_sequences_to_generation(tmp_path, monkeypatch):
    processed = tmp_path / "processed" / "spider"
    db_dir = tmp_path / "raw" / "spider" / "database" / "toy"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "toy.sqlite")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Ana')")
    conn.commit()
    conn.close()

    train = [{"example_id": "spider-train-000000", "db_id": "toy", "question": "Who?", "gold_sql": "SELECT name FROM users"}]
    dev = [{"example_id": "spider-dev-000000", "db_id": "toy", "question": "Who?", "gold_sql": "SELECT name FROM users"}]
    schemas = {
        "toy": {
            "db_id": "toy",
            "tables": [{"name": "users", "columns": [{"name": "id", "type": "number"}, {"name": "name", "type": "text"}]}],
            "foreign_keys": [],
        }
    }
    _write_jsonl(processed / "train.jsonl", train)
    _write_jsonl(processed / "dev.jsonl", dev)
    _write_json(processed / "schemas.json", schemas)

    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        f"""
seed: 42
paths:
  spider_data_dir: {processed}
  spider_db_dir: {tmp_path / "raw" / "spider" / "database"}
  baseline_dir: {tmp_path / "outputs" / "base"}
model:
  name: Qwen/Qwen2.5-3B-Instruct
spider:
  few_shot_count: 1
  eval_batch_size: 1
  max_new_tokens: 16
  stop_sequences:
    - "\\n\\nExample"
generation:
  temperature: 0.0
  do_sample: false
  max_new_tokens: 256
execution:
  sqlite_timeout_seconds: 5
""",
        encoding="utf-8",
    )
    seen_stop_sequences = []

    monkeypatch.setattr(evaluation, "_load_model_for_eval", lambda *_args, **_kwargs: (object(), object()))

    def fake_generate_text_batch(_model, _tokenizer, prompts, generation_cfg):
        seen_stop_sequences.append(generation_cfg.get("stop_sequences"))
        return ["SELECT name FROM users"], [0.0]

    monkeypatch.setattr(evaluation, "generate_text_batch", fake_generate_text_batch)

    metrics = evaluation.evaluate_spider(config_path, model_path=tmp_path / "outputs" / "base")

    assert seen_stop_sequences == [["\n\nExample"]]
    assert metrics["execution_accuracy"] == 1.0
