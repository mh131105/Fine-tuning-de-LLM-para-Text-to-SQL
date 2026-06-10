import pytest

from src.training import _validate_effective_batch_size, _version_tuple


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
