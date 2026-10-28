# R1：先看清 RL 在优化什么

[专题目录](README.md) · [下一课：价值与 Bellman](02_value_and_bellman.md)

本课分三次：R1A 交互与状态，R1B 长期回报，R1C RL 与 BC。第一遍先完成 A/B，不要求立刻背 MDP 的全部符号。

## R1A：动作会改变下一次看到的数据

BC 从数据中学习“在这个 observation 下，专家做了什么”。RL 用 reward 评价行为产生的后果，并学习怎样提高未来累计收益。奖励一般不会告诉你正确动作的标签。

机器人插入任务中，立刻向前推可能让距离误差下降，却造成卡死；先退一点重新对齐可能有更好的长期结果。这里的关键是 **动作改变状态，也改变后续的数据和可选动作**。延迟奖励进一步带来 credit assignment：最后成功，前面哪些动作值得增加概率？

```text
当前状态 s_t → 策略 π → 动作 a_t
     ↑                   ↓
下一状态 s_(t+1) ← 环境转移，同时产生 r_t
```

MDP 可写为 $(\mathcal S,\mathcal A,P,r,\rho_0,\gamma)$，也要说明终止规则或 horizon：

| 量 | 意义 | 插入任务中的教学例子 |
|---|---|---|
| $s_t$ | 足以描述下一步分布的状态 | 相对位姿、速度、接触模式等；具体充分变量依模型而定 |
| $o_t$ | agent 实际能读到的观测 | 图像、关节位置/速度、测得 wrench |
| $a_t$ | 策略发出的动作 | 有坐标系和单位约定的末端增量 |
| $P(s'\mid s,a)$ | 状态转移规律 | 接触动力学与不确定扰动 |
| $r_t$ | 动作后的即时反馈 | 成功奖励、时间代价、超力惩罚 |
| $\rho_0$ | 初始状态分布 | 初始偏移、姿态和物体配置 |
| $\gamma$ | 折扣因子 | 当前与未来奖励的相对权重 |

Markov 性不是“时间上独立”，而是给定当前 state/action 后，预测下一步不再需要更早历史：

$$
p(s_{t+1},r_t\mid s_0,a_0,\ldots,s_t,a_t)
=p(s_{t+1},r_t\mid s_t,a_t).
$$

机器人位置相同、速度不同，下一步结果可以不同，所以仅位置未必充分。只有单帧图像或力传感器也未必足够；可采用 history、belief 或 recurrent policy，但需要检查信息是否足够。若任务有内在截止时间，剩余时间也可能属于状态。

**停下来做：** 对“机械臂抓杯子”，分别列 state、observation、action、reward。不要把 image tensor 直接等同于完整 state，也不要把成功/失败混成 action label。

## R1B：reward 与 return 为什么不同

一条长度为 $T$ 的 episode 包含 $T$ 次动作。采用本专题奖励下标：

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}.
$$

目标通常是最大化初始状态和策略随机性下的期望：

$$
J(\theta)=\mathbb E_{s_0\sim\rho_0,\,\tau\sim\pi_\theta}[G_0].
$$

$G_t$ 是一次轨迹的回报，$J$ 是分布上的平均目标。一次成功不能证明策略的期望表现更好。

### 手算一：当前吃亏可能更好

教学环境只有两条确定路径，都从 `start` 开始：

```text
start --退出 / +2--------------------> terminal
start --准备 / -1--> ready --完成 / +5--> terminal
```

取 $\gamma=0.9$：退出的 $G_0=2$；准备再完成的 $G_0=-1+0.9\times5=3.5$。因此即使第一步奖励更低，第二条路径仍更好。

若改成 $\gamma=0.5$，第二条路径为 $1.5$，最优选择变成退出。两条路径相等时 $-1+5\gamma=2$，即 $\gamma=0.6$。折扣改变了任务偏好，不只是优化器的一个调参选项。

### 手算二：倒着算 return

给定奖励 $[1,2,3]$、$\gamma=0.9$：

$$
G_2=3,\qquad G_1=2+0.9\times3=4.7,\qquad
G_0=1+0.9\times4.7=5.23.
$$

倒推公式 $G_t=r_t+\gamma G_{t+1}$ 是后面 Bellman 和 TD 的直觉起点；Bellman 用期望价值替代未来随机回报。

### 折扣与时间单位

对有界奖励、无限时域，$\gamma<1$ 便于保证折扣和有限；有限长度任务也可用 $\gamma=1$。粗略的折扣步数尺度为 $1/(1-\gamma)$，它不是 episode 长度，也不是硬性截止。

若希望按物理时间保持相同指数衰减，可用 $\gamma=\exp(-\Delta t/\tau)$。改变控制频率时，同时检查每步 reward 的离散化；只沿用旧的 $\gamma$ 和每步惩罚会改变任务。平均奖励 RL 是另一种目标，本专题先学折扣形式。

## R1C：RL、BC 与“有无 reward”

| 方法 | 用什么反馈优化 | 训练时能否新增交互 | 核心问题 |
|---|---|---|---|
| BC | 专家动作的监督 loss | 基础 BC 使用固定示范 | 分布偏移、示范覆盖与动作多峰 |
| Online RL | reward 与长期回报估计 | 可以 | 探索、样本成本和更新稳定性 |
| Offline RL | 固定数据中的 reward/transition 等 | 训练阶段不新增环境交互 | 数据支持之外的估值/决策 |

“RL 一定没有数据集”“只要在线采数据就是 RL”都不准确。DAgger 会交互收集状态并请专家标动作，但主要监督仍是专家动作。奖励也可以由人或模型提供；RL 不要求必须手写 reward，关键是优化形式与反馈来源。

## 本课验收

先独立回答，再核对上文：

1. 为什么“总选即时 reward 最大的动作”可能失败？用两条路径解释。
2. 对奖励 $[-1,0,10]$、$\gamma=0.9$，$G_0$ 是多少？答案：$7.1$。
3. “只提供机器人位置”何时可能违反 Markov 假设？举速度或接触隐藏变量的例子。
4. 若最后因采样时长用完被停止，是否必然代表任务真实终止？不是；下一课再处理它对 target 的影响。

30 秒英文练习：Reinforcement learning optimizes expected cumulative reward through sequential decisions. An action changes both the immediate reward and the future states the agent visits. Behavior cloning instead learns from demonstrated actions, while offline RL learns from a fixed set of transitions without collecting new interactions during training.

实验：[RL01](notebooks/RL01_tabular_foundations.ipynb) 第 1–2 节。下一次只带走一句话：**reward 评价这一步，return 累计从这一步开始的未来。**

参考：[Spinning Up：基本概念](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)；[资料表中的 Sutton & Barto 第 3 章](resources.md)。本课两路径环境和数字是自建教学例子。
