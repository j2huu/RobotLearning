# 第 15 课：VLA 泛化、实验设计与失败诊断

[上一课](14_force_aware_adaptation.md) · [主线计划：16–19](../planning/course_plan.md) · [学习记录](../planning/progress.md)

目标：评价前面构建的策略系统，区分模型、数据、控制和测试协议带来的改进，不重复通用分类指标八股。

按 [循序路线](../planning/vla_learning_path.md) 分三次：**15A读§1–2定义测试问题与指标；15B读§3–5提出区分候选原因的实验；15C完成§6报告并用面试练习册复盘**。没有运行的实验保留为计划。最后使用 [第15课面试卡与25分钟模拟](vla_interview_workbook.md)，把结果、推断与尚未排除的解释分别陈述。

## 1. 先定义想证明哪种能力

| 测试变化 | 需要隔离的对象 | 不应扩大的结论 |
|---|---|---|
| 新颜色/背景/纹理 | 训练视觉分布 | 不等于新技能或物理参数泛化 |
| 新对象/几何 | object identity、尺寸/形状 | 不等于任意新 embodiment |
| 新指令/组合 | task semantics 与组合结构 | 不等于接触稳定性 |
| 新摩擦/刚度/质量 | interaction dynamics | 不等于视觉语言理解改善 |
| 新机器人 | state/action/controller 接口 | 不能只按相同 action width 判断已对齐 |

明确 train、validation、test 在 episode/object/scene/task/robot 哪一层隔离。增强后的同源 episode 不应跨 split。

## 2. 三层评价

1. Offline：masked action error、NLL/CE、codec error、分任务/阶段误差。
2. Closed-loop：task success、progress、time-to-complete、恢复能力、失败类型。
3. Interaction/system：wrench error/peak、contact loss、latency、sensor age、stale command、输出连续性。

Offline loss 低不保证 rollout；插值后轨迹平滑也不证明接触闭环有效。测量 policy inference 和输出采样率时要使用第13课的不同口径。

## 3. 对照需要控制什么

写明 backbone、初始化、训练数据与采样权重、optimizer updates、训练/推理 compute、controller、测试初态与随机扰动。匹配参数量不自动匹配 FLOPs；匹配样本数不自动匹配独立 episode 数或接触覆盖。

采用 paired initial configurations 比较方法，可减少场景差异。多个 seed 与多个 test scenes 控制不同变异来源，报告试验单位、数量和置信区间。

## 4. 失败分类先于“调大模型”

```text
指令/目标错误 → 检查 task conditioning 与高层规划
几何/位置错位 → 检查视觉、标定、state 与 action frame
接触失稳/力不足 → 检查 force interface、sensor 与 regulation
轨迹跳变/过时 → 检查 latency、chunk alignment、interpolation
只在训练场景好 → 检查 split、shortcut、数据覆盖与适配
```

同一现象可能有多个原因。用最小干预区分：固定 reference 测 controller、固定 controller 替换感知、改变一项 force delay、固定 scene 改 instruction。

## 5. Sim-to-real 与 benchmark 的边界

仿真可以检查接口与受控动力学，结果受 contact model、传感器、视觉和动作执行差异影响。一个视觉匹配 benchmark 不自动验证瞬时 force response。

Domain randomization 应对应拟评估的变化，不是增加任意随机数就证明 robustness。Toy spring 实验只能讨论指定模型的闭环规律，不能直接报告成真实机器人成功率。

## 6. 本阶段交付物

完成一个可以审查的 VLA/FP2 study report：

- 图像/语言/state→tokens→head→native action→controller 的数据流。
- 一张 shape/units/frame/time/mask 表。
- 一份训练/推理可见信息与梯度图。
- 一个 Colab notebook 的本人运行记录和失败反例。
- 至少三项公平消融与两类无法解决的失败。
- 五分钟英文口述，分清个人贡献、团队方法和论文结果。

## 7. 下一阶段

第16课进入 reward/Bellman/actor-critic；17课进入 DAgger/offline RL；18课 dynamics/MPC；19课综合项目。当前这四课仍按计划逐步补充，不因已有第6–15课讲义就标成已完成。

如果目的仍是 VLA 主线巩固，先完成本课交付物，再决定需要哪种 RL 或 model-based 能力；不因为“现在都做 RL”就跳过数据与控制接口。

参考：[第5课数据协议](05_training_data_and_observation.md)、[ML评价专题](ml_interview/06_evaluation_and_leakage.md)、[Pi项目与证据](pi_series/06_vla_foundations_and_capstone.md)。
