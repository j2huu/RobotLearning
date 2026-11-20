# R6：Actor-Critic 与 GAE

[目录](README.md) · [上一课](05_policy_gradient.md) · [下一课](07_ppo.md)

## R6A：actor 改行为，critic 估计未来

常见 on-policy actor-critic 中，actor 为 $\pi_\theta$，critic 为 $V_\phi$。用近似 critic 构造 TD residual / advantage：

$$
\delta_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t).
$$

若 $V_\phi=V^\pi$，在给定 $s_t,a_t$ 后，$\delta_t$ 的条件期望为 $A^\pi(s_t,a_t)$；单个 TD residual 仍有随机噪声。Critic 不准时，这个 advantage 估计也会受影响。

常见 surrogate：

$$
L_\pi=-\mathbb E[\log\pi_\theta(a_t\mid s_t)\operatorname{stopgrad}(\hat A_t)],
\qquad
L_V=\mathbb E[(V_\phi(s_t)-\operatorname{stopgrad}(\hat R_t))^2].
$$

这里展示均匀 timestep 训练的实践形式；严格折扣目标的时间权重约定见 R5。Actor-Critic 是一类结构，critic 也可以估计 Q，例如 SAC，不限定必须是 V。

## R6B：GAE 把多个 TD residual 累起来

一段连续且未终止的轨迹上：

$$
\hat A_t^{GAE}=\delta_t+\gamma\lambda\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}+\cdots.
$$

$\lambda=0$ 为一步 TD；$\lambda$ 接近 1 使用更长的回报信息。通常在 bootstrap bias 与采样 variance 间取舍，不能理解成“越大越准”。当真实终止且 $\lambda=1$，递推 telescoping 到 $G_t-V(s_t)$；在截断边界则保留最后 value bootstrap。

手算：$\delta=[1,2,3],\gamma=0.9,\lambda=0.8$，于是 $\gamma\lambda=0.72$：

$$
\hat A_2=3,\quad \hat A_1=2+0.72\times3=4.16,
\quad \hat A_0=1+0.72\times4.16=3.9952.
$$

常见 value regression target 为 $\hat R_t=\hat A_t+V_{old}(s_t)$，在本轮更新前计算并固定。Advantage normalization 可改善数值尺度，但会改变有限 batch 的估计，不应当作为理论无偏证明的一部分。

## R6C：两个 mask 与 tensor

Rollout 常存 `[T,N,...]`：T 是采样步数，N 是并行环境。每个环境的 GAE 沿 T 反推，不能先随意打乱或 flatten 后跨环境递推。

| 量 | shape | 用途 |
|---|---|---|
| observation | `[T,N,d_o]` | 图像时可带额外轴 |
| action | `[T,N,d_a]` 或 `[T,N]` | continuous / discrete |
| reward/value/log_prob/advantage | `[T,N]` | 每个 transition 一个标量 |
| policy entropy | `[T,N]` | 动作维已合并 |

定义 bootstrap mask $b_t=1-terminated_t$。定义递推连接 mask $c_t$：只有 buffer 的下一个槽仍是同一条 episode 的后续 transition 时才为 1；终止、外部截断/reset、当前 rollout 末端都切断递推。

$$
\delta_t=r_t+\gamma b_t V(s_{t+1}^{final})-V(s_t),
\qquad \hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}.
$$

外部截断可有 $b_t=1,c_t=0$：从最终状态估值，但不把 reset 后下一回合的 residual 接进来。Rollout 长度用完但任务仍继续时也是 bootstrap 后切断当前 buffer 的 residual 累加。

```python
# 概念片段；next_value 已正确处理 autoreset 的最终 observation。
gae = 0.0
for t in reversed(range(T)):
    delta = reward[t] + gamma * bootstrap_mask[t] * next_value[t] - value[t]
    gae = delta + gamma * lam * continuation_mask[t] * gae
    advantage[t] = gae
returns = advantage + value  # 更新前固定的 value
```

验收：算对三步 GAE；解释 lambda；说清为什么截断时两个 mask 可以不同；检查 reward `[B]` 与 value `[B,1]` 的广播。

参考：[GAE 原文](https://arxiv.org/abs/1506.02438)、[Gymnasium 的终止/截断处理](https://gymnasium.farama.org/v1.3.0/tutorials/gymnasium_basics/handling_time_limits/)。双 mask 伪代码是按本课 buffer 约定写的教学形式。

## 用你的Walker项目对照

先按项目课的三步trajectory手算，再检查你自己的calculate_advantage。gamma=.99、lambda=.95，GAE按每条trajectory反推，之后才拼入buffer；old values和targets在PPO更新前固定。last统一bootstrap是原实现的边界选择，不要把更完整的双mask参考代码说成旧项目已实现。

自问：末端V=.8改为真正terminal的0，最早一步的advantage为何也会变化？

详见 [R7D：DM Walker项目课](07d_dm_walker_project.md)。
