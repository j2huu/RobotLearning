# R11：把 RL 放回机器人系统与面试

[目录](README.md) · [上一课](10_offline_rl_and_imitation.md) · [33 道题卡](interview_drill.md)

## R11A：policy 输出之后还有系统

```text
camera / proprioception / wrench
 → 对齐、归一化、history / state estimator
 → policy action
 → 反归一化、frame/单位变换、限幅
 → controller / actuator
 → 新观测与环境奖励
```

策略 action 可以是 torque、joint target、末端位姿增量、速度、residual 或高层参数。选择改变了 transition、控制带宽和学习难度。必须区分 policy frequency、controller frequency、simulator integration step；reward 每步与真实秒数的对应见 R1。

Residual RL 的教学形式为 $a=a_{base}+\Delta a_\theta$，但相加前须满足相同单位、frame 和语义。Residual 也可以由监督学习训练；“residual”只说明结构，不能证明使用了 RL。

Asymmetric actor-critic 可以让训练期 critic 读仿真特权状态，而 actor 只读部署可用观测。注意这些信息不能经 encoder/缓存泄漏给部署 actor，训练时的特权估值也不能替代真实评估。

回接 FP2/VLA 时，逐模块说明训练数据、target、loss 和部署作用。网络输出 force、frame 或 selection mask，不代表采用 PPO/SAC；没有训练证据时不要把监督 adaptor 说成 RL。已有接口讲义见 [第 13 课](../13_action_and_control_interfaces.md)。

## R11B：仿真好、真机差，先定位在哪一层

| 假设 | 可检查的证据 | 一个对照 |
|---|---|---|
| 时延/丢帧 | sensor/action 时间戳、控制周期分布 | 在仿真注入实测延迟 |
| 单位/frame/action scale 错 | policy/native action 对照、坐标定义 | 固定小动作检查方向和幅值 |
| 动力学差异 | 摩擦、质量、执行器响应与接触日志 | 依据测量改变参数范围 |
| 观测/归一化偏移 | 特征分布、图像处理与 running stats | 固定统计量并对比输入 |
| reward 与目标不一致 | 各奖励项、成功率、接触和时间 | 去掉可钻空子的项做消融 |
| 初始化/重置差异 | 初始状态与失败恢复分布 | 对齐初始条件后独立评估 |

Domain randomization 随机化动力学/观测条件以训练鲁棒行为，system identification 用数据估计参数；可结合使用。范围过宽可能损害可学性，范围缺失也可能漏掉真实偏差。不能把所有真机失败都归咎 sim-to-real，更不能指望随机化修复接口错误。

接触任务中，成功率、峰值力、力冲量、完成时间、限幅触发和失败模式应按任务选择记录。这里是实验设计建议，不是仓库已完成的真机验证。

## R11C：评价与训练故障

先区分训练曲线与独立评估：固定 checkpoint，在预先定义的初始分布上运行足够多 episode；报告 episode 数、独立训练 seed 数、均值/离散程度或区间、成功定义和交互预算。不同 episode 不能代替多次独立训练 seed。

```text
没学会
├─ 没有有效奖励经验 → 检查探索、重置、数据覆盖
├─ target 错 → 检查 reward scale、terminal、final observation、GAE
├─ 梯度错 → 检查 actor loss 符号、detach、log-prob 和 shape
├─ 更新失控 → 检查 LR、KL、target 更新、update/data 比例
└─ 学到了错误目标 → 检查 reward hacking 与评估协议
```

降低 TD/value loss 不保证策略改善；训练 reward 增长也不保证任务成功。先复现一条 transition 和一次 update，确认数据/目标正确，再做有针对性的超参数调整。

### 一道系统题的作答顺序

“只有少量插入示范，能在仿真训练，最终上真机，你如何设计？”

1. 明确观测、动作 contract、频率、任务终止和评价指标。
2. 建 BC 或已有控制器 baseline，检查数据与执行链。
3. 根据模拟可信度、交互成本和探索难度，提出 PPO/SAC/residual 等候选并解释取舍。
4. 设计训练分布、reward、重置和接口一致性检查；部署评估按实际硬件约束进行。
5. 报告 baseline、消融、seed、成功/失败日志和泛化范围，不承诺未验证的真机效果。

### 项目表达模板

> 问题是 __；观测/动作为 __；反馈和数据来自 __。我负责 __。选择 __ 的原因是 __。在 __ 分布、__ 次试验/seed、__ 预算下，相比 __，指标 __。失败集中在 __；结果目前支持 __，尚未验证 __。

验收：画系统数据流；定位一个 sim-to-real 反例；用模板讲自己的项目；明确哪些结果来自实际运行、哪些只是设计或教学代码。

参考：[Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/)、[SAC Algorithms and Applications](https://arxiv.org/abs/1812.05905)。本课诊断表是学习用检查思路，不是特定论文实现或真机实验记录。
