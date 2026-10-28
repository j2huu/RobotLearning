# R5：Policy Gradient 从哪里来

[目录](README.md) · [上一课](04_dqn_and_stability.md) · [下一课](06_actor_critic_and_gae.md)

## R5A：先把策略放进轨迹概率

先用有限时域、$\gamma=1$ 推导，环境转移和初始分布不依赖参数 $\theta$：

$$
p_\theta(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

令 $R(\tau)=\sum_t r_t$。用 $\nabla p=p\nabla\log p$：

$$
\nabla_\theta J
=\mathbb E_{\tau\sim\pi_\theta}
\left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right]
=\mathbb E\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)R(\tau)\right].
$$

环境概率项对 $\theta$ 的 log derivative 为 0，所以这种 score-function estimator 不要求环境可导，也不需要已知 $P$。但状态分布仍由策略影响，不能说“忽略了动作对未来状态的影响”。

## R5B：REINFORCE 与 reward-to-go

动作不影响过去奖励，可把 $R(\tau)$ 换成该时刻之后的 $G_t$（此处仍取 $\gamma=1$）：

$$
\nabla_\theta J=\mathbb E\left[\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t\right].
$$

用梯度下降实现梯度上升：

$$
L_{actor}=-\frac1N\sum_t\log\pi_\theta(a_t\mid s_t)
\operatorname{stopgrad}(G_t).
$$

离散 categorical policy 输出 logits `[B,K]`，动作 `[B]`，log-prob `[B]`。多维独立 Gaussian 先得到每维 log density `[B,d_a]`，再沿动作维求和为 `[B]`；连续 log-prob 是密度的对数，不是“精确动作点的概率”。

**手算：** Bernoulli 策略 $p=\sigma(\theta)$，采到 $a=1$，则 $\partial_\theta\log\pi(1)=1-p$。当 $p=0.25$、权重为 2，梯度为 1.5，梯度上升会增大动作 1 概率；权重为 -2 则反向。

### 折扣形式别漏约定

若严格优化 $J=\mathbb E[\sum_t\gamma^t r_t]$，且 $G_t=\sum_k\gamma^k r_{t+k}$，按轨迹展开的精确式含外层 $\gamma^t$：

$$
\nabla_\theta J=\mathbb E\left[\sum_t\gamma^t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t\right].
$$

也可把权重吸收到 discounted occupancy 的采样定义。很多实践用均匀采样的 timestep surrogate；面试推导时说明目标与采样约定，不要把缺少 $\gamma^t$ 的公式无条件称为上述 $J$ 的精确梯度。

## R5C：baseline 为什么能减方差

任意不依赖当下动作的 $b(s)$ 满足：

$$
\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)b(s)]
=b(s)\nabla\sum_a\pi(a\mid s)=0.
$$

所以可用 $G_t-b(s_t)$；常用 $b=V^\pi$，对应 advantage 的采样估计。Baseline 不改变该期望梯度，但不代表任何 baseline 都降低方差，也不代表近似 critic/bootstrapped target 完全没有偏差。

Actor 更新时把估计 advantage 当固定权重。若 actor/critic 共享 encoder，要明确两种 loss 如何各自更新共享参数；不能简单将整个 critic 分支“全部 detach”而导致 critic 不学习。

验收：从轨迹概率推导梯度；解释为何环境不可导仍可学；用公式证明 state-only baseline 的期望项为 0；检查 actor loss 符号。

参考：[Spinning Up：Policy Optimization 推导](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)、[Sutton & Barto 第 13 章](resources.md)。
