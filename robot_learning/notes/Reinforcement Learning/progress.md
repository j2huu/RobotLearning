# RL 学习与实验记录

[目录](README.md) · 建立日期：2026-10-25

讲义已准备不等于本人已掌握。以下保留未勾选，实际完成后填写。每课 A/B/C 小单元见专题目录。

| 课 | 已读 | 独立手算/推导 | 口述通过 | 尚未解决的问题 |
|---|---|---|---|---|
| R1 MDP/return | [ ] | [ ] | [ ] | |
| R2 Bellman | [ ] | [ ] | [ ] | |
| R3 MC/TD | [ ] | [ ] | [ ] | |
| R4 DQN | [ ] | [ ] | [ ] | |
| R5 PG | [ ] | [ ] | [ ] | |
| R6 GAE | [ ] | [ ] | [ ] | |
| R7 PPO | [ ] | [ ] | [ ] | |
| R8 SAC | [ ] | [ ] | [ ] | |
| R9 探索/模型 | [ ] | [ ] | [ ] | |
| R10 Offline/IL | [ ] | [ ] | [ ] | |
| R11 机器人/诊断 | [ ] | [ ] | [ ] | |
| R7D Walker项目 W1–W4 | [ ] | [ ] | [ ] | 先解释24维observation、6维motor与Gaussian分布 |

## Walker项目课与RL02（2026-11-09）

- [R7D讲义](07d_dm_walker_project.md) 已依据本人`hw4/walker.py`核对，源码SHA256与既有项目复盘一致；未修改外部课程源码。
- W1：待独立说明14+1+9字段及六个actuators；W2：待讲mu/std/V与joint log-prob；W3：待算timeout/terminal；W4：待追一轮buffer/update。
- [RL02](notebooks/RL02_walker_ppo_tensor_trace.ipynb) 是合成tensor/gradient检查；`runtime_execution: not performed`，不是Walker训练结果。
- 本人实际Colab日期、输出与解释：待记录。历史900+表述不由本次教材生成升级成独立评估均值。

## RL01 实验状态

- 文件：[RL01_tabular_foundations.ipynb](notebooks/RL01_tabular_foundations.ipynb)
- `runtime_execution: not performed`
- 当前证据：代码已编写；静态检查结果见 [VALIDATION](VALIDATION.md)。
- Colab 运行、输出、耗时和学习者解释：待本人记录。

```text
日期 / runtime：
从干净 runtime 全部执行：
seed / 改过的变量：
退出与准备路径的 return：
Bellman 迭代的 start value：
TD / Q-learning 的观察：
一个失败或反例：
这些结果支持什么：
这些结果没有验证什么：
```

## 每次复习的最小记录

```text
单元 / 日期：
没看答案时我的解释：
手算中间步骤：
错在什么假设：
用新数字重新做一次：
30–60 秒面试回答：
下一次从哪里接：
```
