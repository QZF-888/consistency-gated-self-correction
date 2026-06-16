# 基于一致性门控的自我修正

**Consistency-Gated Self-Correction** 是一个轻量级推理阶段方法，用来提升大语言模型在推理任务上的可靠性。模型先给出直接答案，再额外采样若干答案；只有当采样答案不支持直接答案时，才用采样多数答案替换它。

英文首页见 [README.md](README.md)。

## 方法

给定直接答案 `a0` 和 `K` 个采样答案 `a1...aK`，定义直接答案一致性：

```text
c = count(ai = a0) / K
```

最终答案：

```text
如果 c < tau：使用 majority(a1...aK)
否则：保留 a0
```

主实验使用 `K = 5` 和固定阈值 `tau = 0.4`。`K = 3` 是从五次采样的前三次 post-hoc 得到的，不需要重新运行模型。

## 主要结果

在五个指令模型和三个推理数据集上，固定阈值 Gated K=5 将总体平均准确率从 **62.8%** 提升到 **65.2%**。

| 数据集 | 模型数 | Direct | Standard SC | Gated K=5 | 提升 | 触发率 |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 5 | 74.8 | 70.2 | 77.8 | +3.0 | 14.5 |
| ARC-Challenge | 5 | 86.0 | 80.4 | 87.5 | +1.6 | 3.7 |
| GPQA-Diamond | 5 | 27.6 | 27.0 | 30.3 | +2.7 | 30.4 |
| Overall | 15 | 62.8 | 59.2 | 65.2 | +2.4 | 16.2 |

## 模型和数据集

模型：Qwen2.5-7B、InternLM3-8B、Llama3.1-8B、Mistral-7B-v0.3、Gemma2-9B。

数据集：GSM8K 500 条、ARC-Challenge 500 条、GPQA-Diamond 198 条。

## 快速开始

```bash
pip install -e .
pip install -r requirements.txt
python scripts/run_experiment.py --model internlm3_8b --dataset gsm8k --output-dir runs
```

## 结果文件

- `results/summary/main_tau04.csv`：固定阈值主表
- `results/summary/best_k5.csv`：K=5 最佳阈值消融
- `results/summary/k3_posthoc.csv`：K=3 post-hoc 分析
- `results/p0/p0_method_ci.csv`：置信区间和配对统计
- `results/p1/p1_consistency_bins_aggregate.csv`：一致性分桶分析
