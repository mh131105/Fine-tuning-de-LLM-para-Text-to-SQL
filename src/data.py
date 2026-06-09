import json
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class SpiderExample:
    """Representa um exemplo único de conversão texto para SQL."""
    example_id: str
    db_id: str
    question: str
    query: str
    split: Literal["train", "dev"]

@dataclass(frozen=True)
class ColumnSchema:
    name: str
    type: str | None
    is_primary_key: bool

@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: list[ColumnSchema]
    primary_keys: list[str]

@dataclass(frozen=True)
class ForeignKey:
    source_table: str
    source_column: str
    target_table: str
    target_column: str

@dataclass(frozen=True)
class SpiderSchema:
    """Representa o banco de dados e todo o seu esquema de tabelas/chaves."""
    db_id: str
    tables: list[TableSchema]
    foreign_keys: list[ForeignKey]

def save_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Salva dados no formato JSONL (uma entidade por linha)."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows

def load_spider_train(data_dir: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(data_dir) / "train.jsonl")

def load_spider_dev(data_dir: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(data_dir) / "dev.jsonl")

def load_spider_schemas(data_dir: str | Path) -> dict[str, Any]:
    with Path(data_dir, "schemas.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def spider_rows_to_examples(rows: list[dict[str, Any]], split: Literal["train", "dev"]) -> list[dict[str, Any]]:
    """Converte os dicionários crus para o Dataclass limpo."""
    examples: list[dict[str, Any]] = []
    prefix = f"spider-{split}"
    for index, row in enumerate(rows):
        examples.append(
            asdict(
                SpiderExample(
                    example_id=row.get("example_id", f"{prefix}-{index:04d}"),
                    db_id=row["db_id"],
                    question=row["question"],
                    query=row["query"],
                    split=split,
                )
            )
        )
    return examples
