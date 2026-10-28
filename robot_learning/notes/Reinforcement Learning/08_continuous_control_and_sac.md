# R8：连续动作——DDPG、TD3 与 SAC

[目录](README.md) · [上一课](07_ppo.md) · [下一课](09_exploration_reward_and_models.md)

## R8A：连续动作如何找高 Q 的动作

连续高维动作无法像 DQN 那样枚举全部动作。DDPG 学 deterministic actor $\mu_\theta(s)$，用 critic 的动作梯度更新 actor：

$$
L_\pi=-\mathbb E_s[Q_\phi(s,\mu_\theta(s))],
\qquad
y=r+\gamma(1-d)Q_{\bar\phi}(s',\mu_{\bar\theta}(s')).
$$

Replay 允许 off-policy 数据复用；行为探索常在 actor 输出上加噪声。这里需要 critic 对动作可导，不要求物理环境可导。若 critic 在某些动作上产生虚假高值，actor 会利用这些误差。

## R8B：TD3 的三个修改

1. 两个 critic，target 取较小值以抑制过估计。
2. Actor 和 target 更新频率低于 critic（delayed policy updates）。
3. Target policy smoothing：在目标动作上加裁剪噪声，并限制到动作范围，缓解狭窄虚假 Q 峰的利用。

Target smoothing 的噪声用于 target 计算；行为采样的探索噪声是另一处。两个 critic 并非一个专门估好、一个专门估坏；取 min 也可能引入低估。

## R8C：SAC 用随机 actor 与熵正则

最大熵目标在 reward 之外鼓励策略保持一定随机性。$\alpha$ 为 temperature，决定熵项相对 reward 的尺度。这里采用常见的双 Q、无单独 V 网络版本：

$$
a'\sim\pi_\theta(\cdot\mid s'),\qquad
y=r+\gamma(1-d)\left[\min_{i=1,2}Q_{\bar\phi_i}(s',a')
-\alpha\log\pi_\theta(a'\mid s')\right].
$$

Critic 对固定 target 做回归。通过 reparameterization 采样当前动作，actor 最小化：

$$
L_\pi=\mathbb E_{s\sim\mathcal D,\,\epsilon}
\left[\alpha\log\pi_\theta(a_\theta(s,\epsilon)\mid s)
-\min_i Q_{\phi_i}(s,a_\theta(s,\epsilon))\right].
$$

更新 critic 时 target 不带梯度；更新 actor 时 critic 参数可冻结，但 **Q 对输入动作的梯度要保留**。整个 $Q(s,a_\theta)$ 都 detach 会切断 actor 从 Q 学习的路径。

### tanh 动作与 log-prob

常用 $u=\mu+\sigma\odot\epsilon$、$a=\tanh u$ 将动作限制到 $[-1,1]$。密度必须做变量变换：

$$
\log\pi(a\mid s)=\log\mathcal N(u;\mu,\sigma)
-\sum_j\log(1-\tanh^2u_j).
$$

实际需要数值稳定实现。若再做 $a_{native}=b+c\odot a$，在 native 单位下的 log density 还需减 $\sum_j\log|c_j|$；先声明熵和 log-prob 使用哪个动作坐标。多维 log-prob 要沿动作维相加。

### 手算 target

取 $r=1,\gamma=0.9,d=0,Q_1'=4,Q_2'=5,\alpha=0.2,\log\pi(a'\mid s')=-1$，得到 $y=1+0.9(4+0.2)=4.78$。若真实终止则 $y=1$。这个例子选用负 log density；连续密度的 log 也可能为正，不要将其符号固定化。

| 维度 | PPO | SAC（本课版本） |
|---|---|---|
| 数据 | 近期 rollout，有限次复用 | replay，多次复用 |
| 策略 | stochastic；可离散/连续 | stochastic；此处为连续动作版本 |
| 价值 | 常见 V critic | 双 Q critic 与 target Q |
| 主要机制 | clipped policy surrogate | maximum entropy + off-policy actor-critic |
| 成本判断 | 可利用大量并行仿真 | 数据复用适合交互昂贵场景，但需调更新/数据比例 |

这不是跨任务性能排名。SAC 也有离散版本，原始与后续版本的网络结构不同；先明确自己讲哪个版本。

验收：写 SAC 两种 loss；解释熵、双 Q、replay 分别做什么；说明 actor 的梯度通过哪里；按具体交互成本比较 PPO/SAC。

参考：[TD3 原文](https://arxiv.org/abs/1802.09477)、[SAC 原始论文](https://arxiv.org/abs/1801.01290)、[SAC 后续算法版本](https://arxiv.org/abs/1812.05905)、[Spinning Up SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)。
