# Kaggle Usage

This workflow runs in the foreground and shows progress bars.

1. Clone or upload this repository into `/kaggle/working/consistency-gated-self-correction`.
2. Enable GPU and internet.
3. Add `HF_TOKEN` in Kaggle Secrets if required.
4. Edit `MODEL` and `DATASET` in `foreground_cell.py`.

Supported model keys: `qwen2_5_7b`, `internlm3_8b`, `llama3_1_8b`, `mistral_7b_v03`, `gemma2_9b`.

Supported dataset keys: `gsm8k`, `arc_challenge`, `gpqa_diamond`.
