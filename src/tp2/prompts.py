from __future__ import annotations

import hashlib
from typing import Any


SPIDER_SYSTEM_INSTRUCTION = (
    "You are a Text-to-SQL assistant. Generate one SQLite query that answers the question. "
    "Use only the provided schema. Return only SQL, with no markdown and no explanation."
)


def prompt_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _example_gold_sql(example: dict[str, Any]) -> str:
    return str(example.get("gold_sql") or example.get("query") or "").strip()


def build_spider_prompt(
    example: dict[str, Any],
    schema_text: str,
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> str:
    sections = [SPIDER_SYSTEM_INSTRUCTION, ""]
    for index, shot in enumerate(few_shot_examples or [], start=1):
        shot_schema = shot.get("schema_text") or schema_text
        sections.extend(
            [
                f"Example {index}:",
                "Schema:",
                str(shot_schema).strip(),
                f"Question: {shot['question']}",
                f"SQL: {_example_gold_sql(shot)}",
                "",
            ]
        )
    sections.extend(
        [
            "Final task:",
            "Schema:",
            schema_text.strip(),
            f"Question: {example['question']}",
            "SQL:",
        ]
    )
    return "\n".join(sections)


def _choices_as_dict(choices: dict[str, str] | list[str]) -> dict[str, str]:
    if isinstance(choices, dict):
        return {letter: str(choices[letter]) for letter in ["A", "B", "C", "D"]}
    return {letter: str(choices[index]) for index, letter in enumerate(["A", "B", "C", "D"])}


def build_mmlu_prompt(
    question: dict[str, Any],
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> str:
    subcategory = question.get("subcategory", "the subject")
    sections = [
        f"The following are multiple choice questions about {subcategory}.",
        "Answer with exactly one letter: A, B, C, or D.",
        "",
    ]
    for shot in few_shot_examples or []:
        choices = _choices_as_dict(shot["choices"])
        sections.extend(
            [
                f"Question: {shot['question']}",
                f"A. {choices['A']}",
                f"B. {choices['B']}",
                f"C. {choices['C']}",
                f"D. {choices['D']}",
                f"Answer: {shot['answer']}",
                "",
            ]
        )
    choices = _choices_as_dict(question["choices"])
    sections.extend(
        [
            f"Question: {question['question']}",
            f"A. {choices['A']}",
            f"B. {choices['B']}",
            f"C. {choices['C']}",
            f"D. {choices['D']}",
            "Answer:",
        ]
    )
    return "\n".join(sections)
