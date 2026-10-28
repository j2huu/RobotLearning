# R3：从轨迹学习——MC、TD、SARSA 与 Q-learning

[目录](README.md) · [上一课](02_value_and_bellman.md) · [下一课](04_dqn_and_stability.md)

核心问题：没有完整环境模型，手上只有 $(s,a,r,s')$，怎么学价值？

## R3A：MC 与 TD 的 target 从哪里来

用 $V(s)\leftarrow V(s)+\alpha[y-V(s)]$ 更新：

| 方法 | target $y$ | 是否 bootstrap | 主要取舍 |
|---|---|---|---|
| Monte Carlo | 真实采样的完整 $G_t$ | 否 | 通常等回合结束；回报方差可能高 |
| TD(0) | $r_t+\gamma V(s_{t+1})$ | 是 | 一步可更新；依赖当前估计质量 |

Bootstrap 是“用估计构造另一个估计的目标”。MC 也会估计 value，但它的 target 不使用后继 value 估计。MC 的无偏性需针对同一固定策略、完整且正确采样的回报等条件；不能把“MC 无偏/TD 有偏”当成不带条件的性能排名。

定义 TD error：

$$
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
$$

$d_t=1$ 表示 transition **真实终止**。手算：$r=1,V(s)=2,V(s')=4,\gamma=0.9,\alpha=0.1$，得到 target 4.6，$\delta=2.6$，新值 2.26。如果是真实终止，target=1，新值 1.9。

## R3B：SARSA 与 Q-learning 差在“下一步按谁走”

SARSA 的五元组是 $(s_t,a_t,r_t,s_{t+1},a_{t+1})$：

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha
\left[r_t+\gamma(1-d_t)Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\right].
$$

标准 SARSA 使用当前行为策略采出的下一动作，学习该探索策略的价值；Q-learning 则用贪心下一动作作 target：

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha
\left[r_t+\gamma(1-d_t)\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right].
$$

行为策略 behavior policy 负责采样，目标策略 target policy 是正在评价/改进的策略。Q-learning 可用 epsilon-greedy 采样却学习贪心目标，因此 off-policy；**off-policy 不等于 offline**。前者比较策略分布，后者描述能否再与环境交互。

手算：$Q(s',L)=4,Q(s',R)=1$，探索采到 R，$r=0,\gamma=0.9$。SARSA target=0.9，Q-learning target=3.6。Expected SARSA 对 $\pi(a'\mid s')$ 平均；若 L/R 概率 0.8/0.2，target=3.06。

表格 Q-learning 的经典收敛依赖充分访问 state/action、合适步长等条件；这些保证不直接覆盖神经网络、有限 buffer 或固定离线数据。

## R3C：n-step 和 episode 边界

尚未终止且能收集 n 步时：

$$
y_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^nV(s_{t+n}).
$$

若在 n 步内真实终止，只累加到终止且不 bootstrap。增大 n 通常减少对短期 value 估计的依赖，但可能增加采样回报的方差；off-policy 的多步 target 还要考虑策略不匹配。

### 终止与截断必须分开

- `terminated`：MDP 定义中的任务结束，例如成功、跌倒，或任务本身的有限时域到期；终止 continuation=0。
- `truncated`：环境外部停止采样，例如持续任务的外部时间限制。若任务仍可继续，使用最后有效状态 bootstrap。
- `reset`：开始新的独立 episode。绝不能把 reset 后的初始状态当成上一个 transition 的 $s'$。

```python
# 概念片段：next_value 必须来自 transition 的最终有效 observation。
target = reward + gamma * (1.0 - terminated) * next_value
```

这不表示所有超时都一定 bootstrap：先判断时间限制是任务定义的一部分还是采样器加的边界。Vector/autoreset 环境需按实际 API 取最终 observation。

外部截断时可以 bootstrap，但多步 return/GAE 不能直接接上下一回合奖励。R6 会使用两个不同 mask。

## 验收

1. 手写 MC、TD、SARSA、Q-learning target，标明 target 来自样本还是估计。
2. $r=2,V(s')=10,\gamma=0.9$：真实终止 target=2；外部截断且保留有效后继观测 target=11。
3. 为什么 Q-learning 的采样动作可以不是 argmax？目标策略和行为策略分开解释。
4. 写一个会跨 reset 错连数据的反例，说明它怎样污染价值。

实验：[RL01](notebooks/RL01_tabular_foundations.ipynb) 第 4–6 节。参考：[Sutton & Barto 第 5–7 章](resources.md)；边界处理依据 [Gymnasium：Handling Time Limits](https://gymnasium.farama.org/v1.3.0/tutorials/gymnasium_basics/handling_time_limits/)。
