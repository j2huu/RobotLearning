# R7：PPO——知道 ratio、clip 与训练循环

[目录](README.md) · [上一课](06_actor_critic_and_gae.md) · [下一课](08_continuous_control_and_sac.md)

## R7A：同一批动作在新旧策略下有多大可能

使用旧策略 $\pi_{old}$ 收集 rollout，固定 old log-prob、advantage 和 value targets：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t)).
$$

PPO-Clip 最大化：

$$
L^{clip}=\mathbb E\left[\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

Ratio 是对**已经采到的动作**重新计算密度的比值。它不是 clip 动作幅度、clip 梯度，也不是整个轨迹分布已被精确 importance-corrected 的保证。

### 正负 advantage 都要手算

取 $\epsilon=0.2$：

| $\hat A$ | $\rho$ | 原项 $\rho A$ | 裁剪项 | min | 含义 |
|---:|---:|---:|---:|---:|---|
| 2 | 1.4 | 2.8 | 2.4 | 2.4 | 好动作概率已增多，额外激励被截平 |
| 2 | 0.6 | 1.2 | 1.6 | 1.2 | 好动作概率被压低，仍有改进激励 |
| -2 | 0.6 | -1.2 | -1.6 | -1.6 | 坏动作概率已减少，额外激励被截平 |
| -2 | 1.4 | -2.8 | -2.4 | -2.8 | 坏动作概率增多，仍受惩罚 |

“超出区间梯度一定为 0”是错的，必须结合 advantage 符号和 min 选中的分支判断。

## R7B：完整训练循环

1. 用当前策略与环境交互，记录 obs/action/reward/terminal、old log-prob 和 old value。
2. 按 R6 处理边界，计算并固定 GAE/returns；按需要标准化 advantage。
3. 在本轮 rollout 上做多个 minibatch epoch，重算新 log-prob 和 value。
4. 优化 actor surrogate、value loss 和可选 entropy bonus；监控 KL，必要时提前停止本轮 actor 更新。
5. 丢弃这批作为后续标准 PPO actor 训练数据的 rollout，再用新策略收集下一批。

常见最小化写法：$L=-L^{clip}+c_V L_V-c_H\mathbb E[\mathcal H(\pi)]$。系数、value clipping、advantage normalization 依实现选择，不是算法所有版本固定一致。

PPO 通常归类 on-policy：数据由最近的策略收集，只在有限次更新内复用。它不是“绝对不能重复用任何数据”，也不适合无条件从很老的 replay 中采样训练。

## R7C：clip 不能保证什么

PPO-Clip 没有严格约束所有状态动作的概率比或 KL；参数共享会影响未采到的动作，多个更新也可导致策略漂移。KL 是分布层面的变化，不能只看一个 ratio 判断。TRPO 使用近似的 KL 约束优化思路，PPO-Clip 则用更简单的 surrogate 抑制过大改动。

| 观察 | 优先检查 |
|---|---|
| ratio 初始就不接近 1 | old log-prob、动作变换、dropout/normalization 模式、采样参数 |
| KL/clip fraction 很快升高 | 学习率、epoch 数、advantage 尺度、log-prob 实现 |
| value loss 大 | reward 尺度、return mask、critic capacity、target 计算 |
| reward 上升但成功率不升 | reward hacking、训练/评估初始分布、统计方式 |

高 clip fraction 本身不能证明“所有样本都被截平”；其常见统计只计 ratio 超界，仍需看 advantage 符号。训练 reward 也不能替代独立评估。

验收：不用笔记算上表；用 60 秒讲一轮 PPO；指出 clip 与硬约束的区别；说明 old log-prob 为什么应固定且不带梯度。

参考：[PPO 原文](https://arxiv.org/abs/1707.06347)、[Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。

## 用你的Walker项目对照

你的具体实现是actor loss+.5 value MSE，一个Adam优化两套独立网络，10 epochs、minibatch256、ratio clip .2、联合grad norm clip .5。未启用entropy bonus/KL early stop。4096是整回合buffer触发门槛；5000条时每epoch20个minibatches，共200次optimizer steps。

自问：动作clip、ratio clip、梯度clip各限制哪个对象？用同一个raw action算old/new logp，为什么不重新采动作？

详见 [R7D：DM Walker项目课](07d_dm_walker_project.md)。
