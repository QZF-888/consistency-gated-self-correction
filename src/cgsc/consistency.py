from __future__ import annotations

from collections import Counter
from typing import Iterable

from .answer_extraction import canonical_answer


def majority_answer(sample_preds: Iterable[object], answer_type: str, valid_letters: Iterable[str] | None = None) -> tuple[str | None, int, int]:
    canonical = [canonical_answer(pred, answer_type, valid_letters) for pred in sample_preds]
    valid = [pred for pred in canonical if pred is not None]
    if not valid:
        return None, 0, 0
    top_answer, top_count = Counter(valid).most_common(1)[0]
    return top_answer, top_count, len(set(valid))


def initial_consistency(direct_pred: object, sample_preds: Iterable[object], answer_type: str, valid_letters: Iterable[str] | None = None) -> float:
    samples = list(sample_preds)
    if not samples:
        return 0.0
    direct = canonical_answer(direct_pred, answer_type, valid_letters)
    canonical_samples = [canonical_answer(pred, answer_type, valid_letters) for pred in samples]
    return sum(pred == direct for pred in canonical_samples) / len(samples)


def gated_answer(direct_pred: object, sample_preds: Iterable[object], answer_type: str, threshold: float, valid_letters: Iterable[str] | None = None) -> tuple[str | None, float, bool]:
    samples = list(sample_preds)
    consistency = initial_consistency(direct_pred, samples, answer_type, valid_letters)
    majority, _, _ = majority_answer(samples, answer_type, valid_letters)
    direct = canonical_answer(direct_pred, answer_type, valid_letters)
    use_majority = consistency < threshold
    return (majority if use_majority else direct), consistency, use_majority


def transition(initial_correct: bool, final_correct: bool) -> str:
    if initial_correct and final_correct:
        return "C->C"
    if initial_correct and not final_correct:
        return "C->W"
    if (not initial_correct) and final_correct:
        return "W->C"
    return "W->W"
