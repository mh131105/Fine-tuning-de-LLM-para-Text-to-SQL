import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.spider_schema import load_tables_json, serialize_schema

def test_serialize_schema(tmp_path):
    # Criar um arquivo tables.json fictício para o teste
    mock_tables = [
        {
            "db_id": "test_db",
            "table_names_original": ["users", "posts"],
            "column_names_original": [
                [-1, "*"], [0, "id"], [0, "name"], [1, "id"], [1, "user_id"], [1, "content"]
            ],
            "column_types": ["text", "number", "text", "number", "number", "text"],
            "primary_keys": [1, 3],
            "foreign_keys": [[4, 1]]
        }
    ]
    
    tables_file = tmp_path / "tables.json"
    import json
    with open(tables_file, "w") as f:
        json.dump(mock_tables, f)
        
    schema_map = load_tables_json(str(tables_file))
    serialized = serialize_schema(schema_map, "test_db")
    
    assert "CREATE TABLE users (" in serialized
    assert "id NUMBER (PRIMARY KEY)" in serialized
    assert "CREATE TABLE posts (" in serialized
    assert "FOREIGN KEY (user_id) REFERENCES users(id)" in serialized
    assert "content TEXT" in serialized
