from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset

from .answer_extraction import LETTERS, extract_gsm8k_gold


def convert_gsm8k_item(row: dict[str, Any], idx: int) -> dict[str, Any]:
    return {"id": idx, "dataset": "gsm8k", "question": str(row["question"]), "gold_answer": extract_gsm8k_gold(row["answer"]), "answer_type": "numeric", "options": None, "valid_letters": None}


def convert_arc_item(row: dict[str, Any], idx: int) -> dict[str, Any]:
    choice_texts = row["choices"]["text"]
    choice_labels = row["choices"]["label"]
    answer_key = str(row["answerKey"]).strip()
    options, gold = [], None
    for i, text in enumerate(choice_texts):
        label = LETTERS[i]
        options.append((label, str(text)))
        if str(choice_labels[i]).strip() == answer_key:
            gold = label
    if gold is None and answer_key in choice_labels:
        gold = LETTERS[choice_labels.index(answer_key)]
    return {"id": idx, "dataset": "arc_challenge", "question": str(row["question"]), "gold_answer": gold, "answer_type": "choice", "options": options, "valid_letters": LETTERS[:len(options)]}


def _find_col(row: dict[str, Any], candidates: list[str]) -> str | None:
    lower = {k.lower().strip(): k for k in row.keys()}
    for candidate in candidates:
        if candidate.lower().strip() in lower:
            return lower[candidate.lower().strip()]
    for key in row.keys():
        if any(candidate.lower() in key.lower() for candidate in candidates):
            return key
    return None


def convert_gpqa_item(row: dict[str, Any], idx: int, seed: int = 42) -> dict[str, Any]:
    q_col = _find_col(row, ["Question", "question", "problem"])
    correct_col = _find_col(row, ["Correct Answer", "correct_answer", "answer"])
    wrong_cols = [k for k in row.keys() if "incorrect" in k.lower()]
    if q_col is None or correct_col is None or len(wrong_cols) < 3:
        raise ValueError(f"Cannot parse GPQA columns: {list(row.keys())}")
    choices = [{"text": str(row[correct_col]), "is_correct": True}] + [{"text": str(row[c]), "is_correct": False} for c in wrong_cols[:3]]
    random.Random(seed + idx).shuffle(choices)
    options, gold = [], None
    for i, choice in enumerate(choices):
        label = LETTERS[i]
        options.append((label, choice["text"]))
        if choice["is_correct"]:
            gold = label
    return {"id": idx, "dataset": "gpqa_diamond", "question": str(row[q_col]), "gold_answer": gold, "answer_type": "choice", "options": options, "valid_letters": LETTERS[:len(options)]}


def load_items(dataset_key: str, dataset_cfg: dict[str, Any], token: str | None = None) -> list[dict[str, Any]]:
    kwargs = {"path": dataset_cfg["hf_path"], "name": dataset_cfg.get("hf_name"), "split": dataset_cfg["split"]}
    if token:
        kwargs["token"] = token
    dataset = load_dataset(**kwargs)
    converters = {"gsm8k": convert_gsm8k_item, "arc_challenge": convert_arc_item, "gpqa_diamond": convert_gpqa_item}
    items = [converters[dataset_key](row, idx) for idx, row in enumerate(dataset)]
    return items[:int(dataset_cfg["num_samples"])]
