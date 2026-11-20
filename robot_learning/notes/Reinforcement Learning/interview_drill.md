# RL 面试题卡：33 题

[目录](README.md) · [公式表](cheatsheet.md)

先遮住参考要点，口述 30–60 秒；推导题用 3–5 分钟。每题 0–2 分：0=概念混淆，1=定义基本正确但缺条件/例子，2=能解释机制、写关键式并指出边界。分数是自测，不代表公司评价。先做每课三题，再按错因返回对应讲义。

## R1：任务与回报

### Q01：RL 与 supervised learning / BC 有什么区别？

参考要点：RL 优化序列决策的期望累计 reward；动作影响后续状态分布。BC 用专家动作监督。不能简单用“有没有数据集”区分；offline RL 使用固定数据。追问：DAgger 在线交互，为什么仍是 imitation learning？

### Q02：state 与 observation、Markov 性分别是什么？

参考要点：state 对预测下一步分布充分，observation 是实际传感器信息。单帧图像/位置未必包含速度和隐藏接触变量。历史或 RNN 可以帮助，但不自动保证充分状态。追问：有限时域任务为何可能需要剩余时间？

### Q03：gamma 的意义是什么，怎么手算 return？

题目：奖励 $[-1,0,10]$、$\gamma=0.9$。答案：$G_0=7.1$。解释折扣对偏好的影响与无限折扣和的条件。追问：控制频率加倍还沿用相同每步 gamma/惩罚，物理时间目标是否改变？

## R2：价值与 Bellman

### Q04：V、Q 和 advantage 各回答什么问题？

参考要点：V 从当前状态按策略行动；Q 第一动作指定后继续按策略；A=Q−V，比较该动作与策略平均。追问：A 为负是否意味着 reward 为负？为何策略下 A 的动作期望为 0？

### Q05：推导 Bellman expectation equation，再写 optimality equation。

参考要点：从 $G_t=r_t+\gamma G_{t+1}$ 取条件期望。固定策略平均动作，求最优时对可选动作 max；终止 continuation=0。追问：为何 max 与随机转移期望不能随便交换？

### Q06：policy iteration 与 value iteration 有何不同？

参考要点：前者交替固定策略评价/贪心改进；后者反复最优 backup。经典 DP 能访问模型；采样 RL 不要求完整模型。追问：表格收敛结论为何不能直接套到 DQN？

## R3：采样学习

### Q07：MC 与 TD 有何区别，什么是 bootstrap？

参考要点：MC target 用完整采样 return；TD 用一步 reward 加后继估值。Bootstrap 用估计构造 target。讨论方差、对估值误差的依赖和适用条件。追问：MC 不 bootstrap，是否意味着它不估计 value？

### Q08：SARSA、Q-learning、on/off-policy 如何区分？

参考要点：SARSA 用行为策略采出的下一动作；Q-learning 用下一状态 max。Behavior policy 收数据，target policy 是学习目标；off-policy 不等于 offline。手算 R3 中 next-Q `[4,1]`、采到第二动作时的两种 target。

### Q09：time limit 到了，value target 是否设为 reward？

参考要点：先判断任务真实终止还是外部截断。真实终止不 bootstrap；仍可继续的截断应从 final observation bootstrap。不能使用 autoreset 后的新初始 observation。追问：为什么 GAE 还需要另一个递推连接 mask？

## R4：DQN

### Q10：DQN 的 replay buffer 与 target network 各解决什么？

参考要点：replay 减弱连续样本相关并复用数据；target 延缓 target 变化。二者不是理论收敛保证。追问：target 为什么 no-grad，replay 为什么不等于完全独立采样？

### Q11：Double DQN 如何减少过估计？

参考要点：online 网络选择 argmax，target 网络评价所选动作；分离带噪估值的选择与评价。追问：next online=`[4,3]`，target=`[2,5]`，$r=1,\gamma=.9$，DQN=5.5，Double=2.8。

### Q12：DQN 发散或 Q 越来越大，先查什么？

参考要点：terminal/截断与 reward scale、target/梯度路径、replay 数据覆盖、target 更新与学习率、`[B]`/`[B,1]` 广播。Q 的合理量级要结合回报范围，不能仅凭大数断言错误。追问：dueling 与 Double 改的是同一件事吗？

## R5：Policy Gradient

### Q13：从轨迹概率推导 policy gradient，为什么不需要环境可导？

参考要点：$\nabla p=p\nabla\log p$；轨迹 log-prob 对策略求导，环境转移项不直接依赖参数。状态分布的影响由轨迹采样包含。追问：严格折扣目标中外层 $\gamma^t$ 或 occupancy 约定是什么？

### Q14：为什么 state-only baseline 不改变期望梯度？

参考要点：$b(s)\sum_a\pi(a\mid s)\nabla\log\pi(a\mid s)=b(s)\nabla1=0$。动作无关条件关键；critic 的近似和 bootstrap 另可能引入偏差。追问：任意 action-dependent baseline 都可直接相减吗？

### Q15：实现连续 policy loss 时，log-prob 和 detach 如何处理？

参考要点：独立动作维 log densities 求和成每样本一个标量；REINFORCE/常见 on-policy surrogate 的 advantage 固定，loss 前有负号。Tanh 变换需 Jacobian。追问：log density 能为正吗？可以，不能将它当离散概率的 log。

## R6：Actor-Critic / GAE

### Q16：critic 做什么，TD residual 什么时候可估计 advantage？

参考要点：critic 估计 V/Q，帮助 actor 评价行为；若 V 准确，给定 state/action 后 TD residual 的条件期望是 A。单个残差仍有采样噪声。追问：Actor-Critic 的 critic 必须是 V 吗？

### Q17：解释 GAE 的 lambda，并手算三步。

题目：$\delta=[1,2,3],\gamma=.9,\lambda=.8$。答案从后向前为 $[3.9952,4.16,3]$。解释较短/较长估计的取舍；lambda=1 在真实终止与截断下的末端项不同。

### Q18：为什么 bootstrap mask 和 GAE continuation mask 不一定相同？

参考要点：外部截断时可用 final value bootstrap，但下一 buffer 槽可能已 reset，不能跨 episode 累积 residual。Rollout 末端也用最后 value 并结束本段递推。追问：`[T,N]` 先 flatten 再反推会怎样？

## R7：PPO

### Q19：PPO ratio 和 clip 到底在裁剪什么？

参考要点：ratio 是相同采样动作的新/旧策略密度比；clipped surrogate 抑制有利方向的过度更新激励。不是裁剪动作/梯度。追问：$A=-2,\rho=1.4,\epsilon=.2$，min 项为 -2.8，仍不是平坦分支。

### Q20：PPO 为什么通常是 on-policy，又为什么能训练多个 epoch？

参考要点：从近期策略收集 batch，用固定 old log-prob 和 advantage 做有限次 surrogate 优化，然后重新采样。不是每样本只能用一次，也不是适合任意旧 replay。追问：old log-prob 更新中重算并替换成当前值会怎样？

### Q21：clip 能保证策略 KL 很小或性能单调提升吗？

参考要点：不能保证；它不是全分布的硬约束，参数共享、多轮更新与未采样动作都会影响变化。监测 KL/clip fraction，并检查学习率/epoch。追问：为何 clip fraction 高也不表示所有样本 actor 梯度都为 0？

## R8：连续动作与 SAC

### Q22：DDPG、TD3 和 SAC 怎样处理连续动作？

参考要点：DDPG deterministic actor 用 Q 的动作梯度；TD3 加双 Q 最小值、延迟更新和 target smoothing；SAC stochastic actor 加熵正则与 reparameterization。均可用 replay。追问：target smoothing 与采样探索噪声的用途相同吗？

### Q23：写出 SAC 的 critic target 与 actor loss，解释 entropy。

参考要点：target 中 $\min Q'-\alpha\log\pi$；actor 最小化 $\alpha\log\pi-\min Q$。熵权重相对 reward scale；不保证全局探索。追问：更新 actor 时为什么不能把整个 Q 输出 detach？

### Q24：PPO 与 SAC 怎么选？

参考要点：结合并行仿真规模、交互成本、动作空间、replay 适用性与训练稳定性实测；PPO 用近期 rollout，SAC 复用 transition。不能只说一个永远更好。追问：比较时是否统一交互预算、评估分布和 seed？

## R9：探索、reward 与模型

### Q25：稀疏奖励为什么难？增大 entropy 一定有效吗？

参考要点：缺少成功经验，长程 credit assignment 难；随机行为不等于有效探索。可考虑 curriculum、示范、HER 或探索 bonus，并明确适用假设。追问：HER 为什么不是把原目标的失败直接改为成功？

### Q26：举 reward hacking 例子，shaping 是否保持最优策略？

参考要点：接近奖励导致悬停、过大控制惩罚导致不动。任意 shaping 会改变任务；potential form $\gamma\Phi(s')-\Phi(s)$ 在相应有界/终止条件下有 invariance 性质。追问：terminal potential 不处理会出现什么末端项？

### Q27：model-based RL 与 MPC 有什么关系？

参考要点：model-based 利用动力学模型做规划/学习；MPC 反复重规划并只执行一小段。已知模型+固定 cost 的 MPC 不必涉及 RL。追问：为什么模型一步 MSE 很低，规划仍可能失败？

## R10：Offline RL / imitation

### Q28：为什么不能把 SAC 直接放固定数据上就认为解决 offline RL？

参考要点：actor/max 可能利用数据外虚假高 Q，没有在线反馈纠正，误差经 bootstrap 传播。数据支持比动作数值范围更关键。用只见 A、不见 B 而 Q(B)=100 的例子解释。

### Q29：CQL 与 IQL 分别怎样缓解离线问题？

参考要点：CQL 对高 Q 相对数据动作加保守约束；IQL 用数据动作拟合 expectile V、做 Q backup，再 advantage-weighted BC。Expectile 不等于 quantile；二者都不保证弥补任意数据缺口。

### Q30：BC、DAgger、offline RL 需要什么反馈？

参考要点：BC 需要示范动作；DAgger 需要学习者访问状态的专家标签；offline RL 需要固定 transition 和适当 reward 等。先按现有资源选方法。追问：只有成功示范、无失败/覆盖信息时，怎样建立可信 baseline？

## R11：机器人和系统

### Q31：policy 学好了，为什么上真机仍会失败？

参考要点：action scale/frame、延迟、观测处理、控制频率、接触/执行器差异、初始分布。先定位数据流与接口再做随机化。追问：如何区分模型能力不足与时间对齐错误？

### Q32：如何公平评价一个 RL 算法或机器人策略？

参考要点：定义任务/成功、独立评估分布、环境步数/墙钟预算、多个独立训练 seed 和每策略 rollout 数；报告分散程度、失败类型与关键物理指标。训练 reward、单个最好 seed 都不充分。

### Q33：如何讲自己的 RL 项目，避免被追问卡住？

参考要点：任务→观测/action→feedback→算法选择→本人贡献→baseline/消融→样本/seed/指标→失败与边界。若只完成仿真/教学机制，直接说清；不能把 FP2 的监督适配或未运行代码改称个人 RL 实验。

## 25 分钟模拟

### Walker项目加试：学完R7D后使用

先读 [项目课](07d_dm_walker_project.md)，每次挑两题替换通用题，原Q01–Q33编号保留。

| 追问 | 应回答的机制与边界 |
|---|---|
| 24维分别是什么？ | body orientations14 + torso height1 + generalized velocity9；不是24关节角 |
| 6维输出是位置还是力矩？ | 六个motor controls；不能直接把[-1,1]说成相同N·m范围，需计入gear |
| 只有6个输出如何定义随机策略？ | mu为6维；另有可学习state-independent log_std6，执行动作仍6维 |
| Critic为什么不输出6个值？ | 当前策略从观测继续的标量折扣回报，不是逐actuator评分 |
| log-prob为什么sum？ | 对角Gaussian联合密度取乘积，因此log求和；每transition一个ratio |
| buffer=4096时为什么可能有200个optimizer steps？ | 完整1000步回合聚合到5000；ceil(5000/256)×10=200 |
| 900+证明什么？ | 既有材料的训练episode return；不是多seed独立评估均值 |
| 哪些功能没有做？ | 最终代码无entropy bonus/KL early stop/checkpoint-resume；不把模板可选项当本人实现 |

### 通用题模拟

5 分钟：Q01/Q04；5 分钟：Q08 手算；7 分钟：Q19/Q23；5 分钟：Q28 或 Q31；3 分钟：Q33。每次只挑一套，记下追问中最先断开的因果环，再回对应单元补习。

每个主题的原始来源见 [资料表](resources.md)。这些是按知识覆盖设计的练习题，不是任何公司的真实题库。
