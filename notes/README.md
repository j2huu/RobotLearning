# VLA / Physical Intelligence 学习工作区

这个目录用于系统学习 Vision-Language-Action（VLA）、Physical Intelligence 的 π 系列，以及接触丰富操作中的力觉与顺应控制。

当前执行：[第 4–19 课：8 周 Robot Learning 课程计划](ROBOT_LEARNING_COURSE_PLAN.md)，按每周 8–10 小时推进。保留 [第 4 课：概率视角的 BC](04_probabilistic_bc_and_training.md)，从 [第 5 课：训练、数据与观测](05_training_data_and_observation.md) 接着准备。配套 [代码实验](labs/README.md)、[学习记录](LEARNING_PROGRESS.md) 和 [资源地图](RESOURCE_MAP.md) 均位于本目录。

## 从这里开始

1. 阅读 [学习路线](VLA_PI_STUDY_ROADMAP.md)，先看整体依赖和阶段目标。
2. 阅读 [论文索引](PAPER_INDEX.md)，了解当前 22 篇本地论文分别解决什么问题。
3. 开始 [第 1 课：从机器人策略到 VLA](01_vla_policy_foundations.md)。
4. 继续 [第 2 课：Action Chunk 与生成式动作策略](02_action_chunk_and_generative_policy.md)。
5. 进入 [第 3 课：Slow-Fast 接触策略](03_slow_fast_contact_policy.md)，对比 RDP、ImplicitRDP 与 Force Policy。
6. 深入阅读 [FP2 adaptor 与 twist–wrench 机制](03b_fp2_adaptor_twist_wrench.md)。
7. 使用 [第 3 课面试题卡](03_interview_drill.md) 做口述训练。
8. 阅读 [第 4 课：概率 BC](04_probabilistic_bc_and_training.md)，完成 Lab 01。
9. 继续 [第 5 课：训练、数据与观测](05_training_data_and_observation.md)，完成 Lab 02 的数据窗口阶段。
10. 按 [后续计划](ROBOT_LEARNING_COURSE_PLAN.md) 继续第 6、7、8……课。
11. 每读一篇新论文，复制 [论文笔记模板](reading_template.md)，并在论文索引中增加一行。

## 学习进度

| 课程 | 状态 | 核心验收 |
|---|---|---|
| 第 1 课：从机器人策略到 VLA | 已讲解，待独立验收 | 能解释 BC、多峰性、分布偏移与 VLA 接口 |
| 第 2 课：Action Chunk 与生成式动作策略 | 已讲解，待独立验收 | 能比较 ACT、Diffusion Policy、π0、π0-FAST |
| 第 3 课：Slow-Fast 接触策略 | 已讲解，待独立验收 | 能比较 RDP、ImplicitRDP、Force Policy 及其与 VLA 的关系 |
| 第 4 课：概率 BC | 讲义与代码已准备 | 推导 MSE/NLL，解释条件均值与多峰性 |
| 第 5 课：训练、数据与观测 | 讲义与数据窗口代码已准备 | episode split、因果对齐、normalization、mask 与训练诊断 |
| 第 6–19 课 | 已安排 | 表示、生成、VLA、力控、RL、模型学习与实验闭环 |

## 两条学习线

```text
VLA 主干：BC → ACT / Diffusion Policy → RT / Open X → Octo / OpenVLA
                                            ↓
                                      π0 → FAST → π0.5 → openpi

研究专题：视觉-触觉表示 → 力觉模仿学习 → 顺应/混合力位控制
                                            ↓
                         force-aware VLA → fast/slow closed-loop VLA
```

主干回答“通用机器人策略怎样从多模态输入生成动作”；专题回答“遇到接触、遮挡、卡滞和高频扰动时，策略怎样稳定地感知并控制”。两条线在 π 系列的 action expert、action chunk 和闭环执行问题处汇合。
