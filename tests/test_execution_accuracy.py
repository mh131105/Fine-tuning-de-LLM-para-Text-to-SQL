import pytest
from deepeval.test_case import LLMTestCase
import os
import sqlite3
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_metrics.execution_accuracy import ExecutionAccuracyMetric

@pytest.fixture
def setup_db(tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER, name TEXT);")
    cursor.execute("INSERT INTO users VALUES (1, 'Alice');")
    cursor.execute("INSERT INTO users VALUES (2, 'Bob');")
    conn.commit()
    conn.close()
    return str(db_path)

def test_execution_accuracy_correct(setup_db):
    test_case = LLMTestCase(
        input="Get all users",
        actual_output="SELECT * FROM users",
        expected_output="SELECT id, name FROM users",
        additional_metadata={"db_path": setup_db}
    )
    metric = ExecutionAccuracyMetric()
    score = metric.measure(test_case)
    assert score == 1.0
    assert metric.is_successful()
    assert metric.reason == "SUCCESS"

def test_execution_accuracy_mismatch(setup_db):
    test_case = LLMTestCase(
        input="Get Alice",
        actual_output="SELECT * FROM users",
        expected_output="SELECT * FROM users WHERE name='Alice'",
        additional_metadata={"db_path": setup_db}
    )
    metric = ExecutionAccuracyMetric()
    score = metric.measure(test_case)
    assert score == 0.0
    assert not metric.is_successful()
    assert metric.reason == "RESULT_MISMATCH"

def test_execution_accuracy_syntax_error(setup_db):
    test_case = LLMTestCase(
        input="Get all users",
        actual_output="SELECT * FROM users WHERE",
        expected_output="SELECT * FROM users",
        additional_metadata={"db_path": setup_db}
    )
    metric = ExecutionAccuracyMetric()
    score = metric.measure(test_case)
    assert score == 0.0
    assert metric.reason == "SQL_SYNTAX_ERROR"
