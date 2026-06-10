from src.tp2.inference import apply_stop_sequences, extract_sql


def test_extract_sql_cuts_prompt_continuation_without_semicolon():
    raw = """SELECT count(*) FROM continents

Example 4:
Schema:
Database: car_1
Question: How many continents are there?
SQL:"""

    assert extract_sql(raw) == "SELECT count(*) FROM continents"


def test_extract_sql_cuts_explanation_without_semicolon():
    raw = """SELECT DISTINCT Country FROM singer WHERE Age > 20
--------Explanation--------
The query filters singers older than 20."""

    assert extract_sql(raw) == "SELECT DISTINCT Country FROM singer WHERE Age > 20"


def test_extract_sql_prefers_first_statement_with_semicolon():
    raw = "SELECT name FROM users; Explanation: this query lists users."

    assert extract_sql(raw) == "SELECT name FROM users;"


def test_extract_sql_ignores_semicolon_inside_string_literal():
    raw = "SELECT name FROM users WHERE note = 'A; B'\nQuestion: generated continuation"

    assert extract_sql(raw) == "SELECT name FROM users WHERE note = 'A; B'"


def test_extract_sql_handles_markdown_fence_and_followup_text():
    raw = """Here is the SQL:
```sql
SELECT name FROM users
```
Explanation: follow-up text"""

    assert extract_sql(raw) == "SELECT name FROM users"


def test_extract_sql_returns_empty_when_no_select_or_with():
    assert extract_sql("Explanation only, no query.") == ""


def test_apply_stop_sequences_cuts_decoded_generation():
    text = "SELECT count(*) FROM continents\n\nExample 4:\nSchema:"

    assert apply_stop_sequences(text, ["\n\nExample"]) == "SELECT count(*) FROM continents"
