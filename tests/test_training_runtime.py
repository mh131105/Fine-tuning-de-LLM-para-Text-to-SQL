from src.tp2.training import _version_tuple


def test_version_tuple_parses_numeric_prefix():
    assert _version_tuple("0.17.0") == (0, 17, 0)
    assert _version_tuple("0.16.0+cpu") == (0, 16)
