from pathlib import Path
import os
import subprocess
import sys

REPO_DIR = Path('/kaggle/working/consistency-gated-self-correction')
MODEL = 'internlm3_8b'
DATASET = 'gsm8k'
OUTPUT_DIR = Path('/kaggle/working/runs')

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(REPO_DIR / 'requirements.txt')])
cmd = [sys.executable, str(REPO_DIR / 'scripts' / 'run_experiment.py'), '--model', MODEL, '--dataset', DATASET, '--output-dir', str(OUTPUT_DIR)]
print('Running:', ' '.join(cmd))
subprocess.check_call(cmd, cwd=str(REPO_DIR), env=os.environ.copy())
