import pytest

from src.config import load_yaml
from src.training import (
    _training_eos_token,
    _validate_effective_batch_size,
    _version_tuple,
    format_spider_for_sft,
    tokenize_sft_rows,
)


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


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/train_lora_exp_a.yaml",
        "configs/train_lora_exp_b.yaml",
        "configs/train_lora_exp_c.yaml",
        "configs/train_lora_exp_d.yaml",
        "configs/train_lora_exp_e.yaml",
        "configs/train_lora_exp_f.yaml",
    ],
)
def test_lora_training_configs_have_valid_effective_batch_size(config_path):
    config = load_yaml(config_path)

    _validate_effective_batch_size(config["training"])


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
    assert rows[0]["prompt"].endswith("SQL:")
    assert "Question: How many singers are there?" in rows[0]["prompt"]
    assert rows[0]["completion"] == " SELECT count(*) FROM singer<|im_end|>"


def test_tokenize_sft_rows_builds_explicit_completion_mask():
    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [ord(char) for char in text]}

    rows = [{"prompt": "SQL:", "completion": " SELECT 1<|im_end|>"}]

    tokenized = tokenize_sft_rows(rows, FakeTokenizer())

    assert tokenized == [
        {
            "input_ids": [ord(char) for char in "SQL: SELECT 1<|im_end|>"],
            "completion_mask": [0, 0, 0, 0] + [1] * len(" SELECT 1<|im_end|>"),
        }
    ]


def test_tokenize_sft_rows_truncates_input_and_mask_together():
    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(char) for char in text]}

    rows = [{"prompt": "SQL:", "completion": " SELECT 1<|im_end|>"}]

    tokenized = tokenize_sft_rows(rows, FakeTokenizer(), max_length=6)

    assert tokenized == [{"input_ids": [ord(char) for char in "SQL: S"], "completion_mask": [0, 0, 0, 0, 1, 1]}]
