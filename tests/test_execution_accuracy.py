import sqlite3
from types import SimpleNamespace

from custom_metrics.execution_accuracy import ExecutionAccuracy


def _db(tmp_path):
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO users VALUES (?, ?)", [(1, "Alice"), (2, "Bob")])
    conn.commit()
    conn.close()
    return path


def _case(db_path, actual, expected):
    return SimpleNamespace(
        input="question",
        actual_output=actual,
        expected_output=expected,
        additional_metadata={"db_path": str(db_path)},
    )


def test_execution_accuracy_correct_sql(tmp_path):
    metric = ExecutionAccuracy()
    score = metric.measure(_case(_db(tmp_path), "SELECT id, name FROM users", "SELECT * FROM users"))
    assert score == 1.0
    assert metric.is_successful()
    assert metric.reason == "SUCCESS"


def test_execution_accuracy_extracts_markdown(tmp_path):
    metric = ExecutionAccuracy()
    raw = "Here is the SQL:\n```sql\nSELECT name FROM users;\n```\nDone."
    score = metric.measure(_case(_db(tmp_path), raw, "SELECT name FROM users"))
    assert score == 1.0
    assert metric.predicted_sql == "SELECT name FROM users;"


def test_execution_accuracy_cuts_continuation_without_semicolon(tmp_path):
    metric = ExecutionAccuracy()
    raw = "SELECT name FROM users\n\nExample 4:\nSchema:\nDatabase: toy"
    score = metric.measure(_case(_db(tmp_path), raw, "SELECT name FROM users"))
    assert score == 1.0
    assert metric.predicted_sql == "SELECT name FROM users"


def test_execution_accuracy_order_mismatch(tmp_path):
    metric = ExecutionAccuracy()
    score = metric.measure(
        _case(
            _db(tmp_path),
            "SELECT name FROM users ORDER BY name DESC",
            "SELECT name FROM users ORDER BY name ASC",
        )
    )
    assert score == 0.0
    assert metric.reason == "RESULT_MISMATCH"


def test_execution_accuracy_bad_column(tmp_path):
    metric = ExecutionAccuracy()
    score = metric.measure(_case(_db(tmp_path), "SELECT missing FROM users", "SELECT name FROM users"))
    assert score == 0.0
    assert metric.error_type == "execution_error"
