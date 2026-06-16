from __future__ import annotations

import re
from decimal import Decimal
from typing import Iterable

LETTERS = list("ABCDE")
NUM_PATTERN = r"[-+]?\d[\d,]*(?:\.\d+)?"


def normalize_numeric(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
    matches = re.findall(NUM_PATTERN, text)
    if matches:
        text = matches[-1].replace(",", "")
    try:
        number = Decimal(text)
    except Exception:
        return None
    out = format(number.normalize(), "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return "0" if out == "-0" else out


def normalize_choice(value: object, valid_letters: Iterable[str] | None = None) -> str | None:
    if value is None:
        return None
    letters = "".join(valid_letters or LETTERS)
    match = re.search(r"\b([" + letters + r"])\b", str(value).strip().upper())
    return match.group(1) if match else None


def extract_numeric_answer(text: object) -> str | None:
    if text is None:
        return None
    body = str(text)
    patterns = [
        r"Final answer\s*:\s*(" + NUM_PATTERN + r")",
        r"The final answer is\s*(" + NUM_PATTERN + r")",
        r"Answer\s*:\s*(" + NUM_PATTERN + r")",
        r"答案\s*[:：]\s*(" + NUM_PATTERN + r")",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if match:
            return normalize_numeric(match.group(1))
    matches = re.findall(NUM_PATTERN, body)
    return normalize_numeric(matches[-1]) if matches else None


def extract_choice_answer(text: object, valid_letters: Iterable[str] | None = None) -> str | None:
    if text is None:
        return None
    letters = "".join(valid_letters or LETTERS)
    body = str(text).strip()
    patterns = [
        r"Final answer\s*:\s*([" + letters + r"])",
        r"The final answer is\s*([" + letters + r"])",
        r"Answer\s*:\s*([" + letters + r"])",
        r"答案\s*[:：]\s*([" + letters + r"])",
        r"\boption\s*([" + letters + r"])\b",
        r"\(([" + letters + r"])\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    matches = re.findall(r"\b([" + letters + r"])\b", body.upper())
    return matches[-1] if matches else None


def canonical_answer(answer: object, answer_type: str, valid_letters: Iterable[str] | None = None) -> str | None:
    if answer_type == "numeric":
        return normalize_numeric(answer)
    return normalize_choice(answer, valid_letters)


def extract_answer(text: object, answer_type: str, valid_letters: Iterable[str] | None = None) -> str | None:
    if answer_type == "numeric":
        return extract_numeric_answer(text)
    return extract_choice_answer(text, valid_letters)


def is_correct(prediction: object, gold: object, answer_type: str, valid_letters: Iterable[str] | None = None) -> bool:
    return canonical_answer(prediction, answer_type, valid_letters) == canonical_answer(gold, answer_type, valid_letters)


def extract_gsm8k_gold(answer_text: object) -> str | None:
    body = str(answer_text)
    match = re.search(r"####\s*(" + NUM_PATTERN + r")", body)
    if match:
        return normalize_numeric(match.group(1))
    matches = re.findall(NUM_PATTERN, body)
    return normalize_numeric(matches[-1]) if matches else None
