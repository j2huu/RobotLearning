# R2：V、Q、Advantage 与 Bellman

[目录](README.md) · [上一课](01_mdp_and_return.md) · [下一课](03_mc_td_and_q_learning.md)

前置：能算 $G_t$，知道 reward 下标。核心问题：还没真的走完未来，怎样评价当前选择？

## R2A：三个量回答三个问题

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s],
\qquad Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$

$V$ 评价“从这里开始一直按策略行动”；$Q$ 评价“第一步先指定动作，以后按策略”。它们都依赖未来采用的策略，不是环境自带的固定分数。

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a),
\qquad A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

Advantage 问“这个动作比该策略在当前状态的平均选择好多少”。因此 $\mathbb E_{a\sim\pi}[A^\pi(s,a)]=0$；$A<0$ 不表示 reward 一定为负。

沿用 R1：`start` 可退出得 2，或准备得 -1 后到 `ready`；`ready` 完成得 5。$\gamma=0.9$，策略在 start 两动作各选一半，在 ready 必定完成。

| 状态/动作 | 值 |
|---|---:|
| $V^\pi(\mathrm{ready})$ | 5 |
| $Q^\pi(\mathrm{start},\mathrm{退出})$ | 2 |
| $Q^\pi(\mathrm{start},\mathrm{准备})$ | $-1+0.9\times5=3.5$ |
| $V^\pi(\mathrm{start})$ | $0.5\times2+0.5\times3.5=2.75$ |
| 两动作 advantage | $-0.75,+0.75$ |

**手算变化：** 准备概率改为 0.8，则 $V(start)=3.2$，退出/准备的 advantage 为 $-1.2,0.3$。检查 $0.2(-1.2)+0.8(0.3)=0$。

## R2B：Bellman 是把未来分成“一步 + 剩余”

从 $G_t=r_t+\gamma G_{t+1}$ 取条件期望：

$$
V^\pi(s)=\sum_a\pi(a\mid s)
\mathbb E_{r,s'\mid s,a}[r+\gamma V^\pi(s')].
$$

同样：

$$
Q^\pi(s,a)=\mathbb E_{r,s'\mid s,a}
\left[r+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\right].
$$

这里既平均环境随机性，也平均策略随机性；真实终止后的 continuation value 定义为 0。注意 $V$ 放在**下一个**状态，不能把即时 reward 重复算两次。

最优方程则是：

$$
V^*(s)=\max_a\mathbb E_{r,s'\mid s,a}[r+\gamma V^*(s')],
\qquad
Q^*(s,a)=\mathbb E_{r,s'\mid s,a}[r+\gamma\max_{a'}Q^*(s',a')].
$$

**evaluation 固定 $\pi$，control 寻找更好的 $\pi$。** 把期望方程里策略平均直接换成 max，问题已经变了。对随机转移，通常 $\max_a\mathbb E[\cdot]$ 不能随意换成 $\mathbb E[\max_a\cdot]$。

## R2C：知道模型时怎样求值

Dynamic Programming 假设能用转移/奖励模型做期望 backup：

- Policy evaluation：固定策略，重复应用 Bellman expectation backup。
- Policy iteration：先评价当前策略，再对其 Q 贪心改进，循环。
- Value iteration：重复 $V_{k+1}(s)=\max_a\mathbb E[r+\gamma V_k(s')]$。

有限状态、$\gamma<1$ 等标准条件下，最优 Bellman operator 是 sup-norm 下的 $\gamma$ contraction；这是表格 value iteration 收敛的依据，不能直接推广为任意深度 RL 都收敛。

### 两轮同步 backup

初始化两个非终止状态值为 0，每轮都读上一轮值：

| 轮数 | ready | start 的退出候选 | start 的准备候选 | 新 start |
|---|---:|---:|---:|---:|
| 1 | 5 | 2 | -1 | 2 |
| 2 | 5 | 2 | $-1+0.9\times5=3.5$ | 3.5 |

价值从奖励出现的地方向前传播。同步更新和原地更新的中间轨迹可能不同，做手算前说明使用哪一种。

## 验收

1. 不看公式解释 V、Q、A；说清 Q 的第一步和未来策略。
2. 固定策略 Bellman 方程里为什么是平均，最优方程为什么是 max？
3. 随机转移：奖励 1，下一状态价值以 0.25/0.75 概率为 4/8，$\gamma=0.9$。一步期望 target：$1+0.9(0.25\times4+0.75\times8)=7.3$。
4. 若没有 $P$ 但能采到 transition，如何近似期望？下一课用 MC/TD。

实验：[RL01](notebooks/RL01_tabular_foundations.ipynb) 第 3 节。参考：[Spinning Up：价值函数](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)、[Sutton & Barto 第 3/4 章](resources.md)。

## 用你的Walker项目对照

你的critic是独立的24→256→256→1网络。这个1估计从当前观测继续按策略行动的折扣回报，不是即时reward，也不是给六个motor各打一个分；当前项目没有训练SAC式Q网络。用V(next)构造target时，先说明next是否真实terminal。

自问：相同姿态但速度不同，为什么V可能不同？为什么它不是[0,1]的成功概率？

详见 [R7D：DM Walker项目课](07d_dm_walker_project.md)。
