from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


SplitName = Literal["train", "dev"]


@dataclass(frozen=True)
class SpiderExample:
    example_id: str
    db_id: str
    question: str
    gold_sql: str
    split: SplitName

    @property
    def query(self) -> str:
        return self.gold_sql


@dataclass(frozen=True)
class MMLUQuestion:
    question_id: str
    category: str
    subcategory: str
    question: str
    choices: dict[str, str]
    answer: str


def save_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def spider_rows_to_examples(rows: list[dict[str, Any]], split: SplitName) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        gold_sql = row.get("gold_sql") or row.get("query")
        if not gold_sql:
            raise ValueError(f"Spider row {index} has no query/gold_sql field")
        example = SpiderExample(
            example_id=row.get("example_id", f"spider-{split}-{index:06d}"),
            db_id=row["db_id"],
            question=row["question"],
            gold_sql=gold_sql,
            split=split,
        )
        serialized = asdict(example)
        serialized["query"] = example.gold_sql
        examples.append(serialized)
    return examples


def load_spider_split(data_dir: str | Path, split: SplitName) -> list[dict[str, Any]]:
    return read_jsonl(Path(data_dir) / f"{split}.jsonl")


def load_spider_train(data_dir: str | Path) -> list[dict[str, Any]]:
    return load_spider_split(data_dir, "train")


def load_spider_dev(data_dir: str | Path) -> list[dict[str, Any]]:
    return load_spider_split(data_dir, "dev")


def load_spider_schemas(data_dir: str | Path) -> dict[str, Any]:
    return load_json(Path(data_dir) / "schemas.json")


def convert_spider_table_schema(raw: dict[str, Any]) -> dict[str, Any]:
    db_id = raw["db_id"]
    table_names = raw.get("table_names_original") or raw.get("table_names") or []
    column_names = raw.get("column_names_original") or raw.get("column_names") or []
    column_types = raw.get("column_types") or []
    primary_key_indices = set(raw.get("primary_keys") or [])

    tables: list[dict[str, Any]] = []
    for table_index, table_name in enumerate(table_names):
        columns: list[dict[str, Any]] = []
        primary_keys: list[str] = []
        for column_index, column_entry in enumerate(column_names):
            source_table_index, column_name = column_entry
            if source_table_index != table_index:
                continue
            column_type = column_types[column_index] if column_index < len(column_types) else None
            is_primary_key = column_index in primary_key_indices
            if is_primary_key:
                primary_keys.append(column_name)
            columns.append(
                {
                    "name": column_name,
                    "type": column_type,
                    "primary_key": is_primary_key,
                }
            )
        tables.append({"name": table_name, "columns": columns, "primary_keys": primary_keys})

    foreign_keys: list[dict[str, str]] = []
    for source_index, target_index in raw.get("foreign_keys") or []:
        source_table_index, source_column = column_names[source_index]
        target_table_index, target_column = column_names[target_index]
        if source_table_index < 0 or target_table_index < 0:
            continue
        foreign_keys.append(
            {
                "from_table": table_names[source_table_index],
                "from_column": source_column,
                "to_table": table_names[target_table_index],
                "to_column": target_column,
            }
        )

    return {"db_id": db_id, "tables": tables, "foreign_keys": foreign_keys}


def load_spider_schema_from_sqlite(db_path: str | Path, db_id: str | None = None) -> dict[str, Any]:
    path = Path(db_path)
    db_identifier = db_id or path.stem
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        tables: list[dict[str, Any]] = []
        foreign_keys: list[dict[str, str]] = []
        for (table_name,) in table_rows:
            columns: list[dict[str, Any]] = []
            primary_keys: list[str] = []
            for _, name, column_type, notnull, default, pk in connection.execute(f"PRAGMA table_info({table_name!r})"):
                is_primary = bool(pk)
                if is_primary:
                    primary_keys.append(name)
                columns.append(
                    {
                        "name": name,
                        "type": column_type or None,
                        "not_null": bool(notnull),
                        "default": default,
                        "primary_key": is_primary,
                    }
                )
            for fk in connection.execute(f"PRAGMA foreign_key_list({table_name!r})"):
                _, _, to_table, from_column, to_column, *_ = fk
                foreign_keys.append(
                    {
                        "from_table": table_name,
                        "from_column": from_column,
                        "to_table": to_table,
                        "to_column": to_column,
                    }
                )
            tables.append({"name": table_name, "columns": columns, "primary_keys": primary_keys})
        return {"db_id": db_identifier, "tables": tables, "foreign_keys": foreign_keys}
    finally:
        connection.close()


def serialize_schema(schema: dict[str, Any]) -> str:
    lines = [f"Database: {schema.get('db_id', 'unknown')}"]
    for table in schema.get("tables", []):
        column_parts = []
        for column in table.get("columns", []):
            suffix = " primary key" if column.get("primary_key") or column.get("is_primary_key") else ""
            column_type = column.get("type") or "unknown"
            column_parts.append(f"{column['name']} {column_type}{suffix}".strip())
        lines.append(f"Table {table['name']} columns: " + ", ".join(column_parts))
    foreign_keys = schema.get("foreign_keys", [])
    if foreign_keys:
        lines.append("Foreign keys:")
        for key in foreign_keys:
            lines.append(
                f"- {key['from_table']}.{key['from_column']} -> {key['to_table']}.{key['to_column']}"
            )
    return "\n".join(lines)


def normalize_mmlu_record(raw: dict[str, Any], category: str, subcategory: str, index: int) -> dict[str, Any]:
    choices_raw = raw.get("choices") or raw.get("options")
    if isinstance(choices_raw, dict):
        choices = {letter: str(choices_raw[letter]) for letter in ["A", "B", "C", "D"]}
    elif isinstance(choices_raw, list):
        if len(choices_raw) < 4:
            raise ValueError("MMLU question has fewer than four choices")
        choices = {letter: str(choices_raw[pos]) for pos, letter in enumerate(["A", "B", "C", "D"])}
    else:
        raise ValueError("MMLU question has no choices/options field")

    answer = raw.get("answer")
    if isinstance(answer, int):
        answer = ["A", "B", "C", "D"][answer]
    answer = str(answer).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"invalid_mmlu_label: {answer}")

    question = str(raw.get("question") or raw.get("input") or "").strip()
    if not question:
        raise ValueError("MMLU question text is empty")

    return {
        "question_id": raw.get("question_id", f"mmlu-{category.lower()}-{index:06d}"),
        "category": category,
        "subcategory": subcategory,
        "question": question,
        "choices": choices,
        "answer": answer,
    }


def load_mmlu_suite(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    data = load_json(path)
    if isinstance(data, list):
        return data, {}
    questions = data.get("questions", [])
    few_shot_examples = data.get("few_shot_examples", {})
    if not isinstance(questions, list):
        raise ValueError("invalid_mmlu_suite: questions must be a list")
    if not isinstance(few_shot_examples, dict):
        raise ValueError("invalid_mmlu_suite: few_shot_examples must be a mapping")
    return questions, few_shot_examples


def deterministic_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows, but only {len(rows)} are available")
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[index] for index in indices[:count]]
