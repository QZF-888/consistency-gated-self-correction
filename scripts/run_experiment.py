#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from cgsc.answer_extraction import extract_answer
from cgsc.datasets import load_items
from cgsc.evaluation import evaluate_consistency, evaluate_direct, evaluate_standard
from cgsc.generation import GenerationConfig, HFGenerator
from cgsc.prompts import build_direct_prompt, build_standard_self_correction_prompt


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = load_yaml(args.models_config)["models"]
    datasets = load_yaml(args.datasets_config)["datasets"]
    experiment = load_yaml(args.experiments_config)["experiment"]
    model_cfg = models[args.model]
    dataset_cfg = datasets[args.dataset]

    run_name = f"{args.dataset}_{args.model}_n{experiment['n_generations']}_{dataset_cfg['num_samples']}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.dataset, dataset_cfg, token=args.hf_token)
    generator = HFGenerator(model_cfg["hf_id"], GenerationConfig(max_new_tokens=int(dataset_cfg.get("max_new_tokens", model_cfg["max_new_tokens_default"])), temperature=float(experiment["temperature"]), top_p=float(experiment["top_p"]), load_in_4bit=not args.no_4bit), token=args.hf_token, trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))

    direct_rows = []
    for item in tqdm(items, desc="Direct"):
        output = generator.generate(build_direct_prompt(item), do_sample=False, n_return=1)
        pred = extract_answer(output, item["answer_type"], item["valid_letters"])
        direct_rows.append(evaluate_direct(item, pred, output))
        pd.DataFrame(direct_rows).to_csv(output_dir / "direct.csv", index=False)

    direct_df = pd.DataFrame(direct_rows).set_index("id")
    standard_rows = []
    for item in tqdm(items, desc="Standard SC"):
        drow = direct_df.loc[int(item["id"])]
        prompt = build_standard_self_correction_prompt(item, str(drow["direct_output"]))
        output = generator.generate(prompt, do_sample=False, n_return=1)
        pred = extract_answer(output, item["answer_type"], item["valid_letters"])
        standard_rows.append(evaluate_standard(item, drow["direct_pred"], bool(drow["direct_correct"]), pred, output))
        pd.DataFrame(standard_rows).to_csv(output_dir / "standard_self_correction.csv", index=False)

    consistency_rows = []
    for item in tqdm(items, desc="Self-consistency"):
        drow = direct_df.loc[int(item["id"])]
        outputs = generator.generate(build_direct_prompt(item), do_sample=True, n_return=int(experiment["n_generations"]))
        preds = [extract_answer(output, item["answer_type"], item["valid_letters"]) for output in outputs]
        consistency_rows.append(evaluate_consistency(item, drow["direct_pred"], bool(drow["direct_correct"]), preds, outputs))
        pd.DataFrame(consistency_rows).to_csv(output_dir / "self_consistency.csv", index=False)

    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
