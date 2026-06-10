import json
import shutil
import sqlite3

from src.data_sources import ensure_spider_raw_data, find_spider_root, has_spider_raw_files


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_spider_root(root):
    db_dir = root / "database" / "toy"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "toy.sqlite")
    conn.execute("CREATE TABLE singer (id INTEGER, name TEXT)")
    conn.commit()
    conn.close()
    _write_json(root / "train_spider.json", [{"db_id": "toy", "question": "Who?", "query": "SELECT name FROM singer"}])
    _write_json(root / "dev.json", [{"db_id": "toy", "question": "Who?", "query": "SELECT name FROM singer"}])
    _write_json(
        root / "tables.json",
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


def test_find_spider_root_inside_nested_download(tmp_path):
    nested = tmp_path / "download" / "dataset" / "spider"
    _make_spider_root(nested)

    assert find_spider_root(tmp_path / "download") == nested


def test_ensure_spider_raw_data_imports_nested_directory(tmp_path):
    nested = tmp_path / "source" / "dataset" / "spider"
    destination = tmp_path / "data" / "raw" / "spider"
    _make_spider_root(nested)

    result = ensure_spider_raw_data(destination, source="local", source_path=tmp_path / "source")

    assert result.source == "local"
    assert has_spider_raw_files(destination)
    assert (destination / "database" / "toy" / "toy.sqlite").exists()


def test_ensure_spider_raw_data_imports_zip_archive(tmp_path):
    nested = tmp_path / "source" / "dataset" / "spider"
    destination = tmp_path / "data" / "raw" / "spider"
    _make_spider_root(nested)
    archive = shutil.make_archive(str(tmp_path / "spider_bundle"), "zip", root_dir=tmp_path / "source")

    result = ensure_spider_raw_data(destination, source="zip", source_path=archive)

    assert result.source == "local"
    assert has_spider_raw_files(destination)
