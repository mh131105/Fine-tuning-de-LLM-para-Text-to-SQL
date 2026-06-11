from __future__ import annotations

import re
import time
from typing import Any


SQL_START_RE = re.compile(r"\b(select|with)\b", re.IGNORECASE)
SQL_CONTINUATION_MARKERS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\s+[-=#*\s]*explanation\b",
        r"\s+example\s*\d*\s*:",
        r"\s+schema\s*:",
        r"\s+question\s*:",
        r"\s+assistant\s*:",
        r"\s+teacher\s*:",
        r"\s+output\s*:\s*sql\s*:",
        r"\s+task\s+completed\.?\s*sql\s*:",
        r"\s+sql\s*:\s*(?=select|with)\b",
    ]
]


def apply_stop_sequences(text: str, stop_sequences: str | list[str] | tuple[str, ...] | None) -> str:
    if not text or not stop_sequences:
        return text
    if isinstance(stop_sequences, str):
        stop_sequences = [stop_sequences]
    cut_points = [text.find(sequence) for sequence in stop_sequences if sequence and text.find(sequence) != -1]
    if not cut_points:
        return text
    return text[: min(cut_points)].rstrip()


def _generation_kwargs(tokenizer: Any, generation_config: dict[str, Any]) -> dict[str, Any]:
    max_new_tokens = int(generation_config.get("max_new_tokens", 256))
    do_sample = bool(generation_config.get("do_sample", False))
    temperature = float(generation_config.get("temperature", 0.0))
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    return generate_kwargs


def generate_text_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    generation_config: dict[str, Any],
) -> tuple[list[str], list[float]]:
    start = time.perf_counter()
    if not prompts:
        return [], []
    if model is None or tokenizer is None:
        elapsed = time.perf_counter() - start
        return [""] * len(prompts), [elapsed / len(prompts)] * len(prompts)

    previous_padding_side = getattr(tokenizer, "padding_side", None)
    if previous_padding_side is not None:
        tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        if previous_padding_side is not None:
            tokenizer.padding_side = previous_padding_side

    if hasattr(model, "device"):
        inputs = inputs.to(model.device)

    outputs = model.generate(**inputs, **_generation_kwargs(tokenizer, generation_config))
    input_length = inputs["input_ids"].shape[1]
    stop_sequences = generation_config.get("stop_sequences")
    decoded = [
        apply_stop_sequences(
            tokenizer.decode(outputs[index][input_length:], skip_special_tokens=True),
            stop_sequences,
        )
        for index in range(len(prompts))
    ]
    elapsed = time.perf_counter() - start
    return decoded, [elapsed / len(prompts)] * len(prompts)


def generate_text(model: Any, tokenizer: Any, prompt: str, generation_config: dict[str, Any]) -> tuple[str, float]:
    outputs, latencies = generate_text_batch(model, tokenizer, [prompt], generation_config)
    return outputs[0], latencies[0]


def _strip_markdown_fence(text: str) -> str:
    fenced = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def extract_sql(raw_output: str) -> str:
    if not raw_output:
        return ""
    text = _strip_markdown_fence(raw_output)
    match = SQL_START_RE.search(text)
    if not match:
        return ""
    sql = text[match.start() :].strip()
    semicolon = _first_statement_semicolon(sql)
    if semicolon != -1:
        sql = sql[: semicolon + 1]
    else:
        continuation = _first_continuation_marker(sql)
        if continuation != -1:
            sql = sql[:continuation]
    return sql.strip()


def _first_statement_semicolon(sql: str) -> int:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(sql):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == ";":
            return index
    return -1


def _first_continuation_marker(sql: str) -> int:
    positions = []
    for pattern in SQL_CONTINUATION_MARKERS:
        for match in pattern.finditer(sql):
            if not _is_inside_sql_quote(sql, match.start()):
                positions.append(match.start())
                break
    return min(positions) if positions else -1


def _is_inside_sql_quote(sql: str, position: int) -> bool:
    quote: str | None = None
    escaped = False
    for char in sql[:position]:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
    return quote is not None


def extract_mmlu_answer(raw_output: str) -> str | None:
    if not raw_output:
        return None
    text = raw_output.strip()
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    compact = text.strip()

    exact = re.fullmatch(r"\(?\s*([ABCDabcd])\s*\)?\.?", compact)
    if exact:
        return exact.group(1).upper()

    leading = re.match(r"^\s*\(?([ABCDabcd])\)?(?:[ \t]*\n|[\.\):]|$)", compact)
    if leading:
        return leading.group(1).upper()

    answer_patterns = [
        r"(?i)\banswer\s*(?:is|:)?\s*\(?([ABCD])\)?\b",
        r"(?i)\bcorrect\s+answer\s*(?:is|:)?\s*\(?([ABCD])\)?\b",
        r"(?i)\bresposta\s*(?:correta)?\s*(?:e|é|:)?\s*\(?([ABCD])\)?\b",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, compact)
        if match:
            return match.group(1).upper()

    if len(compact) <= 40:
        candidates = re.findall(r"\b([ABCDabcd])\b", compact)
        if len(candidates) == 1:
            return candidates[0].upper()
    return None
