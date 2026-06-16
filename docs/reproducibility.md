# Reproducibility

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

For 7B-9B instruction models, the runner enables 4-bit quantization by default.

## Model Access

Some Hugging Face models require access approval. Set `HF_TOKEN` before running:

```bash
export HF_TOKEN=...
```

## Dataset Splits

| Dataset | Hugging Face path | Config | Split | N |
|---|---|---|---|---:|
| GSM8K | `openai/gsm8k` | `main` | `test` | 500 |
| ARC-Challenge | `allenai/ai2_arc` | `ARC-Challenge` | `train` | 500 |
| GPQA-Diamond | `Idavidrein/gpqa` | `gpqa_diamond` | `train` | 198 |

## Thresholds

The main paper uses fixed `tau = 0.4`. Best-threshold results are ablations, not the main claim.

## K=3

K=3 is computed post-hoc from the first three samples of each K=5 run, avoiding a second model run.
