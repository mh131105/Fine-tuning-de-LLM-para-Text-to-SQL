from src.tp2.prompts import build_mmlu_prompt, build_spider_prompt


def test_spider_prompt_contains_schema_few_shot_and_sql_only_instruction():
    example = {"question": "List singer names.", "gold_sql": "SELECT name FROM singer", "db_id": "toy"}
    shot = {"question": "Count singers.", "gold_sql": "SELECT COUNT(*) FROM singer", "schema_text": "Table singer columns: id, name"}
    prompt = build_spider_prompt(example, "Table singer columns: id, name", [shot])
    assert "Return only SQL" in prompt
    assert "Count singers." in prompt
    assert "List singer names." in prompt
    assert "Table singer columns" in prompt


def test_mmlu_prompt_uses_five_shot_style_choices():
    question = {
        "subcategory": "philosophy",
        "question": "Which option is true?",
        "choices": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
        "answer": "A",
    }
    prompt = build_mmlu_prompt(question, [question])
    assert "Answer with exactly one letter" in prompt
    assert "A. Alpha" in prompt
    assert "Answer: A" in prompt
