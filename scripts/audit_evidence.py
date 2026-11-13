"""Check saved evidence without model inference or SQLite execution."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("exp_e", "exp_f", "exp_g", "exp_h")
SUITE_QUESTION_HASH = "8fc4621bde8bba56f49b1d9d4e26d50777653a583b46120e82979ce321e8cc6b"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def notebook_results(path: Path) -> dict:
    results = {}
    for index, cell in enumerate(read_json(path)["cells"]):
        source = "".join(cell.get("source", []))
        match = re.search(r"scripts.run_benchmarks .*--model_path outputs/(\w+)", source)
        if not match:
            continue
        output = "\n".join("".join(item.get("text", [])) for item in cell.get("outputs", []))
        for start in re.finditer(r"\{\n", output):
            try:
                value, _ = json.JSONDecoder().raw_decode(output[start.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "spider" in value and "mmlu" in value:
                results[match.group(1)] = {"cell_index": index, "summary": value}
                break
    return results


def aggregate(rows: list[dict], task: str) -> dict:
    if task == "spider":
        require(all(row["score"] in (0, 1) for row in rows), "Spider: non-binary score")
        correct = sum(row["score"] == 1 for row in rows)
        errors = Counter(row.get("error_type") or "unknown_error" for row in rows if row["score"] != 1)
        return {"execution_accuracy": correct / len(rows), "total": len(rows), "correct": correct,
                "failed": len(rows) - correct, "error_breakdown": dict(errors)}
    require(all(isinstance(row["is_correct"], bool) for row in rows), "MMLU: invalid correctness flag")
    categories = {}
    for row in rows:
        values = categories.setdefault(row["category"], {"total": 0, "correct": 0})
        values["total"] += 1
        values["correct"] += row["is_correct"]
        require(row["is_correct"] == (row["parsed_answer"] == row["gold_answer"]),
                "MMLU: correctness flag differs from parsed answer")
    for values in categories.values():
        values["accuracy"] = values["correct"] / values["total"]
    correct = sum(row["is_correct"] for row in rows)
    return {"accuracy": correct / len(rows), "total": len(rows), "correct": correct, "by_category": categories}


def audit(evidence_dir: Path) -> dict:
    manifest = read_json(evidence_dir / "source_manifest.json")
    checked_files = 0
    for source in manifest["sources"]:
        for item in source["files"]:
            payload = (evidence_dir / item["member"]).read_bytes()
            require(len(payload) == item["size"] and hashlib.sha256(payload).hexdigest() == item["sha256"],
                    f"Source manifest mismatch: {item['member']}")
            checked_files += 1
    suite_sha = hashlib.sha256((evidence_dir / "mmlu_150_suite.json").read_bytes()).hexdigest()
    require(suite_sha == manifest["mmlu_suite"]["sha256"], "MMLU suite file SHA-256 differs from source manifest")
    summary = read_json(evidence_dir / "notebook_summary.json")
    notebook_path = ROOT / summary["source_notebook"]
    notebook_sha = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    require(notebook_sha == summary["source_sha256"], "Notebook SHA-256 differs from summary")
    results = notebook_results(notebook_path)
    require(results == summary["results"], "Summary differs from saved notebook outputs")
    require(set(results) == {"base", *(f"exp_{letter}" for letter in "abcdefgh")},
            "Notebook must include baseline and experiments A-H")

    suite = read_json(evidence_dir / "mmlu_150_suite.json")
    questions = suite["questions"]
    require(len(questions) == 150 and suite["metadata"]["total_questions"] == 150, "MMLU suite: expected 150 questions")
    require(suite["metadata"]["mock"] is False, "MMLU suite is a mock")
    serialized = "\n".join(hashlib.sha256(str(question).encode("utf-8")).hexdigest() for question in questions)
    question_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    require(question_hash == suite["metadata"]["question_hash"] == SUITE_QUESTION_HASH,
            "MMLU suite question hash differs from the recorded final suite")
    require(Counter(q["subcategory"] for q in questions) == {
        "college_computer_science": 50, "philosophy": 50, "econometrics": 50}, "MMLU suite: unexpected subjects/counts")
    require(set(suite["few_shot_examples"]) == {q["subcategory"] for q in questions}, "MMLU suite: few-shot subjects differ")
    require(all(len(shots) == 5 for shots in suite["few_shot_examples"].values()), "MMLU suite: expected five shots per subject")
    by_question = {question["question_id"]: question for question in questions}
    require(len(by_question) == 150, "MMLU suite: duplicate question IDs")

    reference_rows = {}
    verified = {}
    for model in MODELS:
        verified[model] = {}
        for task, total, key in (("spider", 1034, "example_id"), ("mmlu", 150, "question_id")):
            rows = read_jsonl(evidence_dir / model / f"{task}_predictions.jsonl")
            metrics = read_json(evidence_dir / model / f"{task}_metrics.json")
            require(len(rows) == total, f"{model}/{task}: expected {total} predictions")
            keyed = {row[key]: row for row in rows}
            require(len(keyed) == total, f"{model}/{task}: duplicate IDs")
            require(all(re.fullmatch(r"sha256:[0-9a-f]{64}", row["prompt_hash"]) for row in rows),
                    f"{model}/{task}: invalid prompt hashes")
            require(aggregate(rows, task) == metrics, f"{model}/{task}: predictions differ from metrics")
            require(metrics == results[model]["summary"][task], f"{model}/{task}: metrics differ from final notebook")
            fields = ("db_id", "question", "gold_sql", "prompt_hash") if task == "spider" else (
                "category", "subcategory", "gold_answer", "prompt_hash")
            if task in reference_rows:
                previous = reference_rows[task]
                require(keyed.keys() == previous.keys(), f"{model}/{task}: IDs differ from other adapters")
                require(all(all(row[field] == previous[row_id][field] for field in fields)
                            for row_id, row in keyed.items()), f"{model}/{task}: inputs or prompt hashes differ between adapters")
            else:
                reference_rows[task] = keyed
            if task == "mmlu":
                require(keyed.keys() == by_question.keys(), f"{model}: MMLU IDs differ from suite")
                require(all(row["gold_answer"] == by_question[row_id]["answer"]
                            and row["category"] == by_question[row_id]["category"]
                            and row["subcategory"] == by_question[row_id]["subcategory"]
                            for row_id, row in keyed.items()), f"{model}: MMLU labels/categories differ from suite")
            verified[model][task] = {"correct": metrics["correct"], "total": total}
    return {"status": "ok", "source_files_verified": checked_files, "verified_predictions": verified, "mmlu_question_hash": question_hash,
            "notebook_sha256": notebook_sha,
            "scope": "Saved scores and labels only; no model inference, SQL re-execution, or final paired tests for A-D/baseline."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "reports/evaluation")
    args = parser.parse_args()
    try:
        result = audit(args.evidence_dir)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        parser.exit(1, f"Evidence audit failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
