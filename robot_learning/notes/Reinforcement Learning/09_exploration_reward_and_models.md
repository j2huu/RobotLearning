# R9：探索、奖励与模型——为什么“会优化”还不够

[目录](README.md) · [上一课](08_continuous_control_and_sac.md) · [下一课](10_offline_rl_and_imitation.md)

## R9A：探索是为了获得有用经验

Exploitation 使用当前已知的高价值行为；exploration 尝试尚不确定的行为，以改善后续决策。动作随机不保证有效探索：在一个窄通道任务中，大噪声可能只导致更早失败。

| 机制 | 做什么 | 局限 |
|---|---|---|
| epsilon-greedy | 一定概率随机选动作 | 稀疏奖励下难以形成长序列探索 |
| Gaussian/action noise | 在连续动作上扰动 | 需要与动作单位、范围和时序匹配 |
| entropy regularization | 鼓励策略分布保持随机性 | 不能保证访问所有有价值状态 |
| intrinsic reward，例如 RND | 用预测误差等产生探索 bonus | bonus 不等于任务成功，也不是可靠的通用不确定性 |
| curriculum / demonstrations | 从可达或较容易的任务开始 | 需评价能否迁移到目标分布 |

多臂 bandit 是理解探索的起点：最简单的版本没有动作影响未来状态的转移结构。完整 MDP 还要处理长程状态访问与 credit assignment。探索困难和奖励传播困难可以同时存在，不能只靠增大网络解决。

## R9B：reward 设计会改变最优行为

教学奖励可以写成成功项、距离项、动作代价、接触惩罚的组合，但系数取决于单位和采样频率。持续领取“靠近目标”的奖励可能让策略停在目标附近，不愿完成任务；过大的动作惩罚可能鼓励完全不动。

先列出真正的评估目标，再检查每项 reward 是否能被钻空子。训练 return、成功率、峰值接触力、完成时间应分开报告。目标权重变化属于任务变化，不能只称为“优化更稳定”。

Potential-based shaping 采用：

$$
r'(s,a,s')=r(s,a,s')+\gamma\Phi(s')-\Phi(s).
$$

折扣和相加后，中间 potential telescoping。无限时域中有界 $\Phi$ 且 $\gamma<1$ 时，剩余项只与起点有关；episodic 任务需正确处理终止 potential，例如设为 0。满足相应条件时可保持最优策略；随便增加距离奖励没有这个保证。

手算两步、终止 potential 为 0，$\Phi(s_0)=3,\Phi(s_1)=2,\gamma=0.9$：两步 shaping 的折扣和为 $(0.9\times2-3)+0.9(0-2)=-3$，只改变起点常数。

HER 在 goal-conditioned 任务中，把某条实际达到的结果重新作为目标，并按新目标重算 reward。它不是凭空把失败标成原目标成功，也依赖目标重标记对动力学/数据解释成立等条件。

## R9C：model-free、model-based 与 MPC

Model-free 方法直接学 value/policy，不显式用转移模型展开规划；这不代表没有神经网络或表示。Model-based 方法使用已知或学到的动力学模型，做规划、生成训练数据或辅助策略学习。

$$
\hat s_{k+1}=f_\psi(\hat s_k,a_k),\qquad
\max_{a_{0:H-1}}\sum_{k=0}^{H-1}\gamma^k\hat r(\hat s_k,a_k)
+\gamma^H\hat V(\hat s_H).
$$

MPC 每次基于新观测重新规划，只执行前一小段。若使用已知动力学和固定 cost，不必发生 RL；若模型从交互数据学习并用于提升决策，则可属于 model-based RL 系统的一部分。

长 horizon 可能更好地考虑未来，也更容易累积模型误差、增加优化成本。短 horizon 依赖 terminal value 或局部目标。学习模型上的低一步预测误差不保证闭环规划好，因为 planner 可能主动寻找模型误差大的区域。

验收：对稀疏成功奖励提出两种有区别的改进；构造 reward hacking 反例；解释 shaping 的条件；说明 MPC 和 RL 的关系。

参考：[Potential-based shaping 原文](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)、[RND](https://arxiv.org/abs/1810.12894)、[HER](https://arxiv.org/abs/1707.01495)、[Berkeley CS285 的探索与 model-based 讲义入口](https://rail.eecs.berkeley.edu/deeprlcourse/)。机器人数字例子为教学设计。
