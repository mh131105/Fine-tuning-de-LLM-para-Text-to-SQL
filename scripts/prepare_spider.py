import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Adiciona o diretório raiz ao path para permitir imports do src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import save_json, save_jsonl, spider_rows_to_examples

def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _validate_databases(data_dir: Path, db_ids: Any) -> None:
    """Verifica se os arquivos .sqlite físicos existem no diretório raw."""
    missing = []
    for db_id in db_ids:
        db_path = data_dir / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            missing.append(db_id)
            continue
        # Tenta conectar no modo read-only para garantir que não está corrompido
        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        connection.close()
    if missing:
        raise FileNotFoundError(f"Databases SQLite ausentes para db_id(s): {', '.join(sorted(missing))}")

def _convert_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """
    O coração do processamento: Normaliza o JSON confuso do Spider em estruturas 
    relacionais limpas (TableSchema, ColumnSchema, ForeignKey).
    """
    db_id = raw["db_id"]
    table_names = raw.get("table_names_original") or raw.get("table_names")
    column_names = raw.get("column_names_original") or raw.get("column_names")
    column_types = raw.get("column_types") or []
    primary_key_indices = set(raw.get("primary_keys") or [])

    tables: list[dict[str, Any]] = []
    for table_index, table_name in enumerate(table_names):
        columns = []
        primary_keys = []
        for column_index, column_entry in enumerate(column_names):
            source_table_index, column_name = column_entry
            if source_table_index != table_index:
                continue
            
            is_primary = column_index in primary_key_indices
            if is_primary:
                primary_keys.append(column_name)
                
            columns.append({
                "name": column_name,
                "type": column_types[column_index] if column_index < len(column_types) else None,
                "is_primary_key": is_primary,
            })
        tables.append({"name": table_name, "columns": columns, "primary_keys": primary_keys})

    foreign_keys = []
    for source_index, target_index in raw.get("foreign_keys") or []:
        source_table_index, source_column = column_names[source_index]
        target_table_index, target_column = column_names[target_index]
        if source_table_index >= 0 and target_table_index >= 0:
            foreign_keys.append({
                "source_table": table_names[source_table_index],
                "source_column": source_column,
                "target_table": table_names[target_table_index],
                "target_column": target_column,
            })

    return {"db_id": db_id, "tables": tables, "foreign_keys": foreign_keys}

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Spider train/dev splits and schemas.")
    parser.add_argument("--data_dir", default="data/raw/spider")
    parser.add_argument("--output_dir", default="data/processed/spider")
    # Argumentos extras caso o notebook original passe flags como --config e --split
    parser.add_argument("--config", default="", help="Config (opcional)")
    parser.add_argument("--split", default="", help="Split (opcional)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 1. Carrega os dados brutos
    try:
        train_rows = _load_json(data_dir / "train_spider.json")
        dev_rows = _load_json(data_dir / "dev.json")
        table_rows = _load_json(data_dir / "tables.json")
    except FileNotFoundError as e:
        print(f"Aviso: {e}. O processamento foi abortado pois os dados não estão presentes.")
        return

    # 2. Converte as perguntas/queries para os moldes limpos
    train_examples = spider_rows_to_examples(train_rows, "train")
    dev_examples = spider_rows_to_examples(dev_rows, "dev")
    
    # 3. Converte as bizarrices do table.json para schemas de banco de dados reais
    schemas = {_schema["db_id"]: _schema for _schema in (_convert_schema(row) for row in table_rows)}
    
    # 4. Valida se os bancos de dados batem
    try:
        _validate_databases(data_dir, schemas.keys())
    except FileNotFoundError as e:
        print(f"Aviso: {e}")
        print("Continuando processamento sem bases de dados locais...")

    # 5. Salva na pasta final
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(output_dir / "train.jsonl", train_examples)
    save_jsonl(output_dir / "dev.jsonl", dev_examples)
    save_json(output_dir / "schemas.json", schemas)

    print(f"Preparados {len(train_examples)} exemplos de treino e {len(dev_examples)} de validação.")

if __name__ == "__main__":
    main()
