#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / 'results' / 'summary'
P1 = ROOT / 'results' / 'p1'
OUT = ROOT / 'results' / 'derived'
OUT.mkdir(parents=True, exist_ok=True)


def percent(series: pd.Series) -> pd.Series:
    return (100 * series).round(1)


def main() -> None:
    tau04 = pd.read_csv(SUMMARY / 'main_tau04.csv')
    dataset_avg = tau04.groupby('Dataset', as_index=False).agg(Models=('Model','count'), Direct=('Direct','mean'), Standard_SC=('Standard_SC','mean'), Gated_K5_tau04=('Gated_K5_tau0.4','mean'), Gain=('Gain_tau0.4_vs_Direct','mean'), Trigger_Rate=('Use_Rate_tau0.4','mean'))
    overall = pd.DataFrame([{'Dataset':'Overall','Models':len(tau04),'Direct':tau04['Direct'].mean(),'Standard_SC':tau04['Standard_SC'].mean(),'Gated_K5_tau04':tau04['Gated_K5_tau0.4'].mean(),'Gain':tau04['Gain_tau0.4_vs_Direct'].mean(),'Trigger_Rate':tau04['Use_Rate_tau0.4'].mean()}])
    dataset_avg = pd.concat([dataset_avg, overall], ignore_index=True)
    for col in ['Direct','Standard_SC','Gated_K5_tau04','Gain','Trigger_Rate']:
        dataset_avg[col] = percent(dataset_avg[col])
    dataset_avg.to_csv(OUT / 'dataset_average_tau04_percent.csv', index=False)
    bins = pd.read_csv(P1 / 'p1_consistency_bins_aggregate.csv')
    for col in ['direct_acc','majority_k3_acc','majority_k5_acc','gated_k5_tau04_acc','use_rate_tau04']:
        bins[col] = percent(bins[col])
    bins.to_csv(OUT / 'consistency_bins_percent.csv', index=False)
    print('Wrote release tables to', OUT)


if __name__ == '__main__':
    main()
