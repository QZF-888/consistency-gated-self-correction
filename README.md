# Consistency-Gated Self-Correction

**Consistency-Gated Self-Correction** is a lightweight inference-time method for improving reasoning reliability in large language models. The model first gives a direct answer, samples additional answers, measures whether the samples support the direct answer, and revises only low-consistency cases.

中文介绍见 [README.zh-CN.md](README.zh-CN.md).

## Core Idea

Given a direct answer `a0` and `K` sampled answers `a1...aK`, define the consistency of the direct answer as:

```text
c = count(ai = a0) / K
```

The final answer is:

```text
if c < tau: use majority(a1...aK)
else:       keep a0
```

The main setting uses `K = 5` and fixed `tau = 0.4`. The `K = 3` analysis is computed post-hoc from the first three of the five sampled answers.

## Main Result

Across five instruction-tuned models and three reasoning benchmarks, fixed-threshold Gated K=5 improves the average accuracy from **62.8%** to **65.2%**.

| Dataset | Models | Direct | Standard SC | Gated K=5 | Gain | Trigger rate |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 5 | 74.8 | 70.2 | 77.8 | +3.0 | 14.5 |
| ARC-Challenge | 5 | 86.0 | 80.4 | 87.5 | +1.6 | 3.7 |
| GPQA-Diamond | 5 | 27.6 | 27.0 | 30.3 | +2.7 | 30.4 |
| Overall | 15 | 62.8 | 59.2 | 65.2 | +2.4 | 16.2 |

## Models and Datasets

Models:

- Qwen2.5-7B-Instruct
- InternLM3-8B-Instruct
- Llama3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3
- Gemma2-9B-IT

Datasets:

- GSM8K, 500 examples
- ARC-Challenge, 500 examples
- GPQA-Diamond, 198 examples

## Repository Layout

```text
configs/        Model, dataset, and experiment settings
src/cgsc/       Core implementation
scripts/        Experiment and analysis entrypoints
kaggle/         Foreground Kaggle usage
results/        Released CSV summaries
paper/          Figure SVGs and LaTeX references
```

## Quick Start

```bash
pip install -e .
pip install -r requirements.txt
```

Run one experiment:

```bash
python scripts/run_experiment.py --model internlm3_8b --dataset gsm8k --output-dir runs
```

Print the full model-by-dataset matrix:

```bash
python scripts/run_all_matrix.py
```

Build compact release tables from committed CSV files:

```bash
python scripts/build_released_tables.py
```

## Released CSVs

- `results/summary/main_tau04.csv`: fixed-threshold main table
- `results/summary/best_k5.csv`: best-threshold K=5 ablation
- `results/summary/k3_posthoc.csv`: post-hoc K=3 analysis
- `results/p0/p0_method_ci.csv`: confidence intervals and paired statistics
- `results/p1/p1_consistency_bins_aggregate.csv`: consistency-bin analysis

## Notes

- GPQA answer choices are shuffled with seed 42.
- The gate uses strict comparison: revise when `consistency < tau`, not `<=`.
- Some models require Hugging Face access and `HF_TOKEN`.
