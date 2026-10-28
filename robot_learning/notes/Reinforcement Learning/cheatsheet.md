# RL 公式与算法对照

[目录](README.md) · [题卡](interview_drill.md)

用于完成手算后的回忆，不代替推导。约定 $r_t$ 在 $a_t$ 后产生，$d_t$ 为真实终止；$b_t=1-d_t$。所有 target 在对应回归 loss 中固定。

## 核心公式

| 量 | 公式 | 要说清的条件 |
|---|---|---|
| Return | $G_t=\sum_{k\geq0}\gamma^k r_{t+k}$ | 有限回合到终止；无限时域检查收敛 |
| V/Q | $V^\pi=\mathbb E_\pi[G_t\mid s_t]$；$Q^\pi=\mathbb E_\pi[G_t\mid s_t,a_t]$ | 未来依赖策略 |
| Advantage | $A^\pi=Q^\pi-V^\pi$ | 对 $a\sim\pi$ 的均值为 0 |
| Bellman evaluation | $V^\pi(s)=\mathbb E[r+\gamma V^\pi(s')]$ | 当前 action 按 $\pi$ 平均；终止 continuation=0 |
| TD residual | $\delta=r+\gamma bV(s')-V(s)$ | 使用 final observation |
| SARSA target | $r+\gamma bQ(s',a')$ | 下一动作来自当前行为策略 |
| Q-learning target | $r+\gamma b\max_{a'}Q(s',a')$ | 目标为贪心；数据可探索采样 |
| GAE | $\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}$ | $c_t$ 切断跨 episode/当前 buffer 边界 |
| Policy gradient | $\mathbb E[\sum_t\gamma^t\nabla\log\pi(a_t\mid s_t)(G_t-b(s_t))]$ | 精确折扣目标、轨迹采样约定见 R5 |
| PPO surrogate | $\mathbb E[\min(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A)]$ | 最大化；old log-prob、A 固定 |
| SAC target | $r+\gamma b(\min_i Q_i'(s',a')-\alpha\log\pi(a'\mid s'))$ | $a'$ 从当前 actor 采；target 无梯度 |
| SAC actor loss | $\mathbb E[\alpha\log\pi(a\mid s)-\min_iQ_i(s,a)]$ | 最小化；Q 对 actor 动作保留梯度 |

## 算法地图

| 算法 | 学什么 | 数据/策略关系 | 常见动作空间 | 关键机制 |
|---|---|---|---|---|
| MC evaluation | V/Q | 基础形式 on-policy | 表格例子 | 完整采样 return，无 bootstrap |
| SARSA | Q | on-policy | 离散 | 实际下一动作作 target |
| Q-learning / DQN | Q | off-policy | 离散 | max backup；DQN 加 replay/target |
| REINFORCE | stochastic policy | on-policy | 离散/连续 | log-prob × return/baseline |
| PPO | actor + 常见 V critic | on-policy、近期 batch 有限复用 | 离散/连续 | ratio/clipped surrogate |
| DDPG | deterministic actor + Q | off-policy | 连续 | actor 沿 Q 的动作梯度更新 |
| TD3 | deterministic actor + 双 Q | off-policy | 连续 | min-Q、delayed update、target smoothing |
| SAC | stochastic actor + 双 Q | off-policy | 本专题讲连续版 | entropy、reparameterization、replay |
| CQL / IQL | 离线价值与策略 | 固定数据 offline | 依具体版本 | 保守估值 / expectile 与加权 BC |
| DAgger | imitation policy | 交互采样状态、专家标注 | 离散/连续 | dataset aggregation |

On/off-policy 与 online/offline 是两个轴。Model-based/model-free 是另一个轴；使用 neural network 不等于 model-based，使用已知模型的 MPC 也不必是 RL。

## 面试中最常见的概念坑

- Reward、return、value、advantage 含义不同；高即时 reward 未必高 return。
- Bootstrap 用估计构造 target；它和是否用神经网络无关。
- Policy gradient 不需环境可导；SAC/DDPG 的 actor 更新需要 critic 对动作可导。
- PPO clip 不直接裁剪动作，也不是全局概率比/KL 的硬约束。
- 多维 Gaussian log-prob 沿动作维求和；tanh action 需密度变换。
- 外部截断可能 bootstrap；GAE 仍须切断 reset 后新 episode 的递推。
- 小 TD loss、高训练 reward、一次成功，都不足以证明策略可靠。

公式依据及原文见 [资料表](resources.md) 和各课末尾；数值例子均为教学构造。
