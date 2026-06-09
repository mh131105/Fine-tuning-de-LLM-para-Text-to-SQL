import sqlite3

from src.tp2.sql_utils import compare_sql_results, execute_sql, has_order_by, is_safe_select_query


def _db(tmp_path):
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO singer (id, name) VALUES (?, ?)", [(1, "Ana"), (2, "Bruno")])
    conn.commit()
    conn.close()
    return path


def test_safe_select_filter():
    assert is_safe_select_query("SELECT * FROM singer")
    assert is_safe_select_query("WITH s AS (SELECT * FROM singer) SELECT * FROM s")
    assert not is_safe_select_query("DROP TABLE singer")
    assert not is_safe_select_query("UPDATE singer SET name='X'")
    assert not is_safe_select_query("SELECT * FROM singer; DROP TABLE singer")


def test_has_order_by():
    assert has_order_by("select * from singer order by name")
    assert not has_order_by("select * from singer")


def test_execute_and_compare_ignores_order_without_order_by(tmp_path):
    db_path = _db(tmp_path)
    predicted = execute_sql(db_path, "SELECT name FROM singer ORDER BY name DESC")
    gold = execute_sql(db_path, "SELECT name FROM singer")
    assert compare_sql_results(predicted, gold, preserve_order=False)


def test_compare_preserves_order_when_requested(tmp_path):
    db_path = _db(tmp_path)
    predicted = execute_sql(db_path, "SELECT name FROM singer ORDER BY name DESC")
    gold = execute_sql(db_path, "SELECT name FROM singer ORDER BY name ASC")
    assert not compare_sql_results(predicted, gold, preserve_order=True)
