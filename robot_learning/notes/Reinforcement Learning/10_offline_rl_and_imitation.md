# R10：Offline RL、BC 与 DAgger

[目录](README.md) · [上一课](09_exploration_reward_and_models.md) · [下一课](11_robotics_and_diagnostics.md)

## R10A：先分清监督来源与交互权限

| 方法 | 主要反馈 | 训练阶段新增交互 | 需要什么额外资源 |
|---|---|---|---|
| BC | 专家动作 label | 基础版本不需要 | 示范数据 |
| DAgger | 学习者访问状态上的专家动作 | 需要 | 能给这些状态重新标注的专家 |
| Online RL | reward/transition | 需要 | 交互环境或机器人 |
| Offline RL | 固定 transition 数据及奖励等 | 不新增 | 覆盖相关状态/动作的数据 |

DAgger 迭代收集学习者访问的状态，请专家标动作并聚合数据，减轻只在专家状态分布训练的偏移。它不是“把 reward 换成人类动作的 PPO”，也不要求必须在线实时遥操作；专家标注协议可以不同，但需能为所访问状态提供目标。

Offline RL 数据可以来自专家、次优策略、多种行为的混合。固定数据集不会自动使任意 off-policy 算法适合离线使用。

## R10B：数据外动作的高 Q 从哪里来

某个状态只有动作 A 的数据，观测 reward=1 后终止；B 从未出现。网络在 A 上拟合为 1，在 B 上外推为 100。训练 loss 在 A 上很低，贪心策略却选 B。若 B 实际失败，离线训练没有新的 B 反馈来纠正它。

更长链条中，Bellman max/actor 查询数据外动作，把过高估计传播到更早状态。关键是 **缺少支持的数据不能被低训练 loss 补足**。动作在数值范围内也不表示位于行为分布支持中。

必须追问：数据覆盖了哪些 state/action？成功与失败各多少？奖励能否从日志正确重建？terminal、时间戳和 observation 是否可靠？Evaluation 是否依赖额外在线试验？OPE 也依赖覆盖和建模假设，不能凭离线 TD loss 排名策略。

## R10C：两种代表性思路

### CQL：对某些数据外高值保持保守

离散动作下，一个教学简式是：

$$
L_Q=L_{TD}+\eta\,\mathbb E_{s\sim\mathcal D}
\left[\log\sum_a\exp Q(s,a)
-\mathbb E_{a\sim\mathcal D(\cdot\mid s)}Q(s,a)\right].
$$

最小化该项压低总体高 Q 相对于数据动作 Q 的差距。这是便于理解的离散形式；连续动作要近似积分/采样，论文有不同变体。保守性过强也可能损失改进空间；不能将理论下界条件泛化成任意训练都可靠的保证。

### IQL：避免在 value backup 中直接查询数据外动作

IQL 用数据动作的 Q 拟合上 expectile 的 V，再做 Q 的 Bellman regression，最后用 advantage-weighted BC 提取策略。Expectile 是非对称平方损失的解，不是 quantile：

$$
L_V=\mathbb E_{(s,a)\sim\mathcal D}
\left[|\tau-\mathbf 1_{u<0}|u^2\right],
\qquad u=\operatorname{stopgrad}(Q(s,a))-V(s).
$$

$\tau>0.5$ 时对低估较高 Q 的误差给更大权重。Q target 使用 $r+\gamma(1-d)V(s')$，actor 增大数据中高 advantage 动作的 likelihood。实际通常限制指数权重以改善数值稳定性。

IQL 不需要在这些 value 更新中显式评价 actor 生成的未见动作，但策略泛化仍可能产生数据外行为；数据覆盖问题并未消失。具体网络、target 和更新细节需回原文/实现。

## 机器人场景练习

有一批固定插入示范，没有可靠 reward，但可从成功标记和传感器重建部分指标。先确定训练目标和日志质量，建立 BC baseline，再判断是否有足够覆盖和可信奖励支持 offline RL。若还能请专家在偏离状态标注，DAgger 是另一种资源配置。不要因为“有数据集”就直接跳 SAC。

验收：解释 off-policy/offline；画出 B=100 的反例；比较 CQL 与 IQL；说明 BC、DAgger、RL 各需要何种反馈。BC 不以显式回报为优化目标，但不能据此断言它在任何情况下都无法超过某条示范轨迹。

参考：[DAgger](https://proceedings.mlr.press/v15/ross11a.html)、[CQL](https://arxiv.org/abs/2006.04779)、[IQL](https://arxiv.org/abs/2110.06169)。

## 用你的Walker项目对照

你的Walker数据来自当前策略与环境交互，按最近buffer做有限轮PPO更新后清空；不是专家action数据集，也不是offline RL。将所有历史transition长期混用，会改变原on-policy训练假设，不能只把buffer更名为replay就认为算法不变。

自问：若只剩旧轨迹、不能再交互，你还缺什么信息，原PPO更新的哪些前提会失效？

详见 [R7D：DM Walker项目课](07d_dm_walker_project.md)。
