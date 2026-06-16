from __future__ import annotations


def format_options(options: list[tuple[str, str]]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in options)


def build_direct_prompt(item: dict) -> str:
    if item["answer_type"] == "numeric":
        return f"""Answer the following math problem.\n\nProblem:\n{item['question']}\n\nPlease reason briefly, and end your response with exactly one line:\nFinal answer: <number>\n"""
    option_text = format_options(item["options"])
    valid = "/".join(item["valid_letters"])
    task_name = "multiple-choice science question" if item.get("dataset") == "gpqa_diamond" else "multiple-choice question"
    return f"""Answer the following {task_name}.\n\nQuestion:\n{item['question']}\n\nOptions:\n{option_text}\n\nChoose the single best answer. Please reason briefly, and end your response with exactly one line:\nFinal answer: <{valid}>\n"""


def build_standard_self_correction_prompt(item: dict, previous_solution: str) -> str:
    if item["answer_type"] == "numeric":
        return f"""You previously answered the following math problem.\n\nProblem:\n{item['question']}\n\nYour previous solution was:\n{previous_solution}\n\nPlease carefully review your previous reasoning and final answer. If there is a mistake, correct it. Then provide the final answer.\n\nEnd your response with exactly one line:\nFinal answer: <number>\n"""
    option_text = format_options(item["options"])
    valid = "/".join(item["valid_letters"])
    return f"""You previously answered the following multiple-choice question.\n\nQuestion:\n{item['question']}\n\nOptions:\n{option_text}\n\nYour previous solution was:\n{previous_solution}\n\nPlease carefully review your previous reasoning and final answer. If there is a mistake, correct it. Then provide the final answer.\n\nEnd your response with exactly one line:\nFinal answer: <{valid}>\n"""
