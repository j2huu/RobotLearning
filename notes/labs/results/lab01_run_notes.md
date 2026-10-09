# Lab 01 准备验证记录

日期：2026-10-09。由助手运行，用于确认实验可用；不代表学习者已经完成实验验收。

```bash
python notes/labs/01_probabilistic_bc.py --steps 800 --seed 7 --output notes/labs/results/lab01_seed7.json
```

运行环境：Python 3.13.11、PyTorch 2.10.0+cu128；本实验使用 CPU 和一个计算线程。

| 模型 | Held-out mean-prediction MSE | Held-out NLL | Sampled mode-compatibility |
|---|---:|---:|---:|
| deterministic MSE | 1.0038 | 未定义 | 0.000 |
| single Gaussian | 1.0057 | 1.4219 | 0.146 |
| two-component mixture | 1.0073 | -0.8823 | 0.993 |

MSE 一列衡量各模型的 mean prediction，而 mode-compatibility 衡量模型产生的动作样本距合成专家的任一分支是否小于 0.15。两者评价的对象不同。

观测为 0 时：MSE 模型 mean 约为 -0.004；single Gaussian mean 约为 -0.025、standard deviation 约为 0.992；mixture 的两个 mean 约为 -0.992 和 0.988，standard deviation 约为 0.052 和 0.053。

这一次运行展示了 conditional mean、宽单峰分布与双峰分布的差别。连续密度的 NLL 可以为负，不意味着概率大于 1。Gaussian 与 mixture 的 NLL 使用同一动作单位，才在这里进行比较。

实验没有机器人动力学、contact、controller 或 rollout。0.993 不是机器人成功率，也不是对某篇论文的复现结果。检查更多 seed 是学习任务的一部分，本记录只报告 seed 7。

机器可读结果：[lab01_seed7.json](lab01_seed7.json)。
