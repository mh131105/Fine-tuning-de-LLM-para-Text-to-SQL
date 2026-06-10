import json

from src.tp2 import evaluation


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


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
