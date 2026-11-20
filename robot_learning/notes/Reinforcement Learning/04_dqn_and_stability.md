# R4：DQN 为什么需要 replay 和 target network

[目录](README.md) · [上一课](03_mc_td_and_q_learning.md) · [下一课](05_policy_gradient.md)

## R4A：把 Q 表换成网络

离散动作数为 $K$，网络输入一批状态/观测，输出所有动作的 Q：`obs [B,d_o] → Q [B,K]`。从中 gather 实际执行动作，得到 `q_taken [B]`。图像输入可用 encoder，但输出维度仍由离散动作集合决定。

$$
y=r+\gamma(1-d)\max_{a'}Q_{\bar\phi}(s',a'),\qquad
L(\phi)=\mathbb E[(Q_\phi(s,a)-\operatorname{stopgrad}(y))^2].
$$

$\phi$ 为 online network，$\bar\phi$ 为延迟更新的 target network。实际实现也常采用 Huber loss。训练走向 TD target，而不是要求 Q 数值本身尽量大。

```python
# PyTorch 概念片段：所有标量 batch 统一为 [B]。
q_taken = online(obs).gather(1, action[:, None]).squeeze(1)
with torch.no_grad():
    next_q = target(next_obs).max(dim=1).values
    y = reward + gamma * (1.0 - terminated.float()) * next_q
loss = torch.nn.functional.smooth_l1_loss(q_taken, y)
```

若一边 `[B,1]`、另一边 `[B]`，广播可变成 `[B,B]` 且仍正常 backward。先检查 shape，再解释 loss 曲线。

## R4B：两种机制分别解决什么

Replay buffer 存 transition，随机小批量采样，减少连续样本强相关的影响，并复用旧数据；它不保证样本真正 i.i.d.，也不会自动修复稀有成功样本不足。

Target network 让目标在若干更新内变化较慢，可定期复制参数或软更新：

$$
\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi.
$$

它缓解“估计在追逐自己快速变化的目标”，不消除所有偏差。Function approximation、bootstrapping、off-policy 组合常称 deadly triad；这是理解发散风险的线索，不表示三者组合一定失败。

## R4C：三个扩展要分清

Double DQN 用 online network 选动作、target network 评价：

$$
y=r+\gamma(1-d)Q_{\bar\phi}
\left(s',\arg\max_{a'}Q_\phi(s',a')\right).
$$

同一组带噪估计同时选最大值并评价，会产生过高估计的倾向；分离选择和评价可减轻该问题，不保证估计无偏。

手算：下一状态 online Q 为 `[4,3]`，target Q 为 `[2,5]`，$r=1,\gamma=0.9,d=0$。DQN target=5.5，Double DQN target=2.8。两个网络不是分别专门估计好/坏动作。

- Dueling：结构上拆 $V(s)$ 和 action-dependent stream，以 $Q=V+A-\mathrm{mean}_a A$ 等形式组合。网络中叫 A 的 stream 不自动等于某个策略的真实 advantage。
- Prioritized replay：按 priority 偏置抽样，常用 TD error，配合 importance weights 减轻抽样偏差；异常大误差不一定最有学习价值。
- 连续高维动作不能直接枚举 max；要用优化器或 actor 等机制，R8 继续。

验收：解释 replay/target/Double 分别改变数据、目标稳定性、max 估计中的哪一环；画出哪些张量需要梯度。

参考：[DQN：2015 Nature 原文](https://www.nature.com/articles/nature14236)、[Double DQN](https://arxiv.org/abs/1509.06461)。Dueling/PER 属于扩展识别，原文入口见 [资料表](resources.md)。

## 用你的Walker项目对照

Walker的6维action代表六个连续motor输入，不是六个可选动作。因此不能把DQN的6个Q输出直接解释成这份actor。若想用离散化方案，需要定义每维分箱和联合动作组合，且这是额外设计；你实际实现的是Gaussian PPO。

自问：每个motor分成5档，共有多少联合动作？5**6=15625，而不是30。

详见 [R7D：DM Walker项目课](07d_dm_walker_project.md)。
