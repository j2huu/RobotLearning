# Robot Learning 学习项目

这个目录用于系统学习 Vision-Language-Action（VLA）、Physical Intelligence 的 π 系列，以及接触丰富操作中的力觉与顺应控制。

当前执行：[第 4–19 课：8 周 Robot Learning 课程计划](planning/course_plan.md)，按每周 8–10 小时推进。保留 [第 4 课：概率视角的 BC](notes/04_probabilistic_bc_and_training.md)，从 [第 5 课：训练、数据与观测](notes/05_training_data_and_observation.md) 接着准备。配套 [代码实验](labs/README.md)、[学习记录](planning/progress.md) 和 [资源地图](resources/resource_map.md) 均位于本目录。

## 目录职责

| 目录 | 放什么 | 入口 |
|---|---|---|
| `notes/` | 连续编号的课程讲义、专题笔记与面试题 | [讲义目录](notes/README.md) |
| `labs/` | 可运行代码、实验规格与小型结果 | [实验目录](labs/README.md) |
| `planning/` | 课序、时间安排、验收与个人学习记录 | [课程计划](planning/course_plan.md)、[进度](planning/progress.md) |
| `resources/` | 本地论文索引、VLA 路线与外部资源映射 | [论文索引](resources/paper_index.md)、[资源地图](resources/resource_map.md) |

每课沿用“读讲义 → 推导/答题 → 做 Lab → 解释结果 → 记录进度”的顺序。课号与 Lab 编号独立；一个 Lab 可以跨多课扩展。

## 当前课程与实验对应

| 课 | 讲义 | 配套实验 | 当前材料 |
|---|---|---|---|
| 4 | [概率 BC](notes/04_probabilistic_bc_and_training.md) | [Lab 01](labs/01_probabilistic_bc.py) | [运行记录](labs/results/lab01_run_notes.md)、[结果](labs/results/lab01_seed7.json) |
| 5 | [训练、数据与观测](notes/05_training_data_and_observation.md) | [Lab 02 数据窗口](labs/02_episode_windows.py) | [结果](labs/results/lab02_episode_windows.json)；frame/action conversion 后续扩展 |
| 6–19 | 按 [课序](planning/course_plan.md) 继续准备 | 按 [实验规格](labs/README.md) 实现 | 尚未完成的讲义/代码不生成占位文件 |

## 从这里开始

1. 阅读 [学习路线](resources/vla_pi_roadmap.md)，先看整体依赖和阶段目标。
2. 阅读 [论文索引](resources/paper_index.md)，了解当前 22 篇本地论文分别解决什么问题。
3. 开始 [第 1 课：从机器人策略到 VLA](notes/01_vla_policy_foundations.md)。
4. 继续 [第 2 课：Action Chunk 与生成式动作策略](notes/02_action_chunk_and_generative_policy.md)。
5. 进入 [第 3 课：Slow-Fast 接触策略](notes/03_slow_fast_contact_policy.md)，对比 RDP、ImplicitRDP 与 Force Policy。
6. 深入阅读 [FP2 adaptor 与 twist–wrench 机制](notes/03b_fp2_adaptor_twist_wrench.md)。
7. 使用 [第 3 课面试题卡](notes/03_interview_drill.md) 做口述训练。
8. 阅读 [第 4 课：概率 BC](notes/04_probabilistic_bc_and_training.md)，完成 Lab 01。
9. 继续 [第 5 课：训练、数据与观测](notes/05_training_data_and_observation.md)，完成 Lab 02 的数据窗口阶段。
10. 按 [后续计划](planning/course_plan.md) 继续第 6、7、8……课。
11. 每读一篇新论文，复制 [论文笔记模板](notes/reading_template.md)，并在论文索引中增加一行。

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
