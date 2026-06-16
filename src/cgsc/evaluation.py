from __future__ import annotations

from collections import Counter
from typing import Iterable

from .answer_extraction import canonical_answer, is_correct
from .consistency import initial_consistency, majority_answer, transition


def evaluate_direct(item: dict, pred: object, output: str = "") -> dict:
    correct = is_correct(pred, item["gold_answer"], item["answer_type"], item["valid_letters"])
    return {"id": int(item["id"]), "gold_answer": item["gold_answer"], "answer_type": item["answer_type"], "direct_output": output, "direct_pred": pred, "direct_correct": bool(correct)}


def evaluate_standard(item: dict, direct_pred: object, direct_correct: bool, pred: object, output: str = "") -> dict:
    correct = is_correct(pred, item["gold_answer"], item["answer_type"], item["valid_letters"])
    return {"id": int(item["id"]), "gold_answer": item["gold_answer"], "direct_pred": direct_pred, "direct_correct": bool(direct_correct), "standard_output": output, "standard_pred": pred, "standard_correct": bool(correct), "transition": transition(bool(direct_correct), bool(correct))}


def evaluate_consistency(item: dict, direct_pred: object, direct_correct: bool, sample_preds: Iterable[object], sample_outputs: Iterable[str]) -> dict:
    sample_preds, sample_outputs = list(sample_preds), list(sample_outputs)
    answer_type, valid_letters = item["answer_type"], item["valid_letters"]
    top_answer, top_count, unique_count = majority_answer(sample_preds, answer_type, valid_letters)
    direct_canon = canonical_answer(direct_pred, answer_type, valid_letters)
    canonical_samples = [canonical_answer(pred, answer_type, valid_letters) for pred in sample_preds]
    counts = Counter(pred for pred in canonical_samples if pred is not None)
    sample_corrects = [is_correct(pred, item["gold_answer"], answer_type, valid_letters) for pred in sample_preds]
    return {"id": int(item["id"]), "gold_answer": item["gold_answer"], "direct_pred": direct_pred, "direct_correct": bool(direct_correct), "sample_preds": " ||| ".join(str(pred) for pred in sample_preds), "sample_corrects": " ||| ".join(str(v) for v in sample_corrects), "sample_outputs": "\n\n===== SAMPLE SPLIT =====\n\n".join(sample_outputs), "top_sample_answer": top_answer, "top_sample_count": top_count, "top_sample_correct": is_correct(top_answer, item["gold_answer"], answer_type, valid_letters), "initial_answer_count": counts.get(direct_canon, 0), "initial_consistency_confidence": initial_consistency(direct_pred, sample_preds, answer_type, valid_letters), "top_consistency_confidence": top_count / len(sample_preds) if sample_preds else 0.0, "unique_answer_count": unique_count, "sample_accuracy": sum(sample_corrects) / len(sample_corrects) if sample_corrects else 0.0}
