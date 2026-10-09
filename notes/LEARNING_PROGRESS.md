# Robot Learning 学习记录

计划版本：2026-10-09。每周 8–10 小时，课程安排见 [执行计划](ROBOT_LEARNING_COURSE_PLAN.md)。

“已讲解”表示已有讨论/讲义；“通过”需要独立推导、代码解释和口述。当前不根据阅读数量推断掌握程度。

## 之前的课程

| 课程 | 状态 | 下一次复盘重点 |
|---|---|---|
| 1：机器人策略与 VLA | 已讲解，待独立验收 | MSE 多峰平均、distribution shift、语义与物理泛化 |
| 2：chunk 与生成策略 | 已讲解，待独立验收 | episode/prediction/execution horizon；CVAE μ/σ；flow vs token |
| 3：slow-fast 接触策略 | 已讲解，待独立验收 | causal leakage、force correction vs regulation、缓存与调度 |
| 3b：FP2 adaptor 与 twist–wrench | 已讲解，待独立验收 | 双对偶变换、控制 mask、desired/measured wrench |

## 后续课程

| 周/课 | 状态 | 代码/推导证据 | 英文口述 | 未解决问题 |
|---|---|---|---|---|
| 1 / 4：概率 BC | 讲义与 Lab 01 已准备 | 待本人运行与解释 | 待完成 | |
| 1 / 5：训练、数据与观测 | 讲义与 Lab 02 数据窗口已准备 | 待本人运行与解释 | 待完成 | |
| 2 / 6：多模态表示 | 已安排 | | | |
| 2 / 7：Transformer 与因果性 | 已安排 | | | |
| 3 / 8：CVAE 与 DP | 已安排 | | | |
| 3 / 9：Flow Matching | 已安排 | | | |
| 4 / 10：VLM → VLA | 已安排 | | | |
| 4 / 11：π0 | 已安排 | | | |
| 5 / 12：FAST 与 π0.5 | 已安排 | | | |
| 5 / 13：动作与控制接口 | 已安排 | | | |
| 6 / 14：力觉融合与 adaptation | 已安排 | | | |
| 6 / 15：泛化与评估 | 已安排 | | | |
| 7 / 16：RL 与 Actor-Critic | 已安排 | | | |
| 7 / 17：Offline RL / DAgger | 已安排 | | | |
| 8 / 18：Dynamics / MPC | 已安排 | | | |
| 8 / 19：代码与面试整合 | 已安排 | | | |

## 每次课后的记录模板

```text
日期 / 课号：
本次能独立解释的三个概念：
一条自己推导的公式（含假设）：
代码入口 / seed / 配置 / 指标：
实验支持什么、不能支持什么：
英文口述中的一个薄弱点：
尚未解决的问题：
下一次具体动作：
```

## 阶段验收

- [ ] 第 2 周：从 conditional BC loss 解释到 causal attention 和模态消融。
- [ ] 第 4 周：写出 flow loss，并解释 π0 的条件、动作表示与推理。
- [ ] 第 6 周：独立设计一种 force-aware VLA 与公平的消融方案。
- [ ] 第 8 周：完成数据流报告、可重复小实验和 FP2 英文面试。
