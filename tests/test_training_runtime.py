import pytest

from src.training import _training_eos_token, _validate_effective_batch_size, _version_tuple, format_spider_for_sft


def test_version_tuple_parses_numeric_prefix():
    assert _version_tuple("0.17.0") == (0, 17, 0)
    assert _version_tuple("0.16.0+cpu") == (0, 16)


def test_validate_effective_batch_size_accepts_matching_value():
    _validate_effective_batch_size(
        {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
        }
    )


def test_validate_effective_batch_size_rejects_mismatch():
    with pytest.raises(ValueError, match="effective_batch_size"):
        _validate_effective_batch_size(
            {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 16,
            }
        )


def test_training_eos_token_prefers_config():
    tokenizer = type("Tokenizer", (), {"eos_token": "<tokenizer_eos>"})()

    assert _training_eos_token({"eos_token": "<configured_eos>"}, tokenizer) == "<configured_eos>"


def test_training_eos_token_uses_qwen_default_without_tokenizer():
    assert _training_eos_token({}) == "<|im_end|>"


def test_format_spider_for_sft_uses_prompt_completion_with_eos():
    examples = [
        {
            "db_id": "toy",
            "question": "How many singers are there?",
            "gold_sql": "SELECT count(*) FROM singer",
        }
    ]
    schemas = {
        "toy": {
            "db_id": "toy",
            "tables": [
                {
                    "name": "singer",
                    "columns": [
                        {"name": "id", "type": "number", "primary_key": True},
                        {"name": "name", "type": "text"},
                    ],
                }
            ],
            "foreign_keys": [],
        }
    }

    rows = format_spider_for_sft(examples, schemas, eos_token="<|im_end|>")

    assert len(rows) == 1
    assert set(rows[0]) == {"prompt", "completion"}
    assert rows[0]["prompt"].endswith("SQL: ")
    assert "Question: How many singers are there?" in rows[0]["prompt"]
    assert rows[0]["completion"] == "SELECT count(*) FROM singer<|im_end|>"
