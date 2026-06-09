import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sql_utils import extract_sql

def test_markdown_extraction():
    raw = "Here is the query:\n```sql\nSELECT * FROM users;\n```\nExplanation..."
    assert extract_sql(raw) == "SELECT * FROM users;"
    
def test_prefix_extraction():
    raw = "SQL: SELECT name FROM posts"
    assert extract_sql(raw) == "SELECT name FROM posts"
    
def test_select_extraction():
    raw = "I think the answer is SELECT count(*) FROM table"
    assert extract_sql(raw) == "SELECT count(*) FROM table"
