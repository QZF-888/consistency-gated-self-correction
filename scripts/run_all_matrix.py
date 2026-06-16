#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually launch jobs sequentially.")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    args = parser.parse_args()
    with open(args.experiments_config, "r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)["matrix"]
    commands = [["python", "scripts/run_experiment.py", "--model", model, "--dataset", dataset] for model in matrix["models"] for dataset in matrix["datasets"]]
    for command in commands:
        print(" ".join(command))
        if args.execute:
            subprocess.check_call(command)


if __name__ == "__main__":
    main()
