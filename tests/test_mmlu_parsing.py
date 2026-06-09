from src.tp2.inference import extract_mmlu_answer


def test_extract_mmlu_answer_exact():
    assert extract_mmlu_answer("A") == "A"


def test_extract_mmlu_answer_answer_prefix():
    assert extract_mmlu_answer("Answer: B") == "B"


def test_extract_mmlu_answer_sentence():
    assert extract_mmlu_answer("The correct answer is C.") == "C"


def test_extract_mmlu_answer_short_sentence():
    assert extract_mmlu_answer("I think it is D") == "D"


def test_extract_mmlu_answer_long_ambiguous_output():
    output = "A and B are both mentioned, but C also appears in a long explanation before D."
    assert extract_mmlu_answer(output) is None
