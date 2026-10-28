# P2：Flow Matching——从概率路径到可执行的采样循环

前课：[P1](01_pi0_architecture.md)。主读：[π0 v1](../../../paper/2410.24164v1.pdf) §IV、附录 B/D；[π0.5 v1](../../../paper/2504.16054v1.pdf) §IV-B。代码约定见固定版 [pi0.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)。

目标：能从端点与时间方向推导 target、解释训练和推理的信息差异，并定位正负号错误。

## 1. 为什么需要一个随机动作模型

给定条件 $c$，专家可能左绕，也可能右绕。固定方差 Gaussian BC 的 mean prediction 最小化 MSE，可能落在两条轨迹中间。Flow matching 学习如何将简单噪声分布运输成条件动作分布，而不是直接预测一个均值。

“运输”指样本在 **action chunk 空间**中移动。它不是在真实空间里把机器人沿着随机噪声移动；只有采样结束并完成动作转换后，chunk 才交给执行器。

## 2. 先采用噪声 → 数据的时间方向

采样一对端点：

$$
A\sim p_{data}(A\mid c),\quad \epsilon\sim\mathcal N(0,I),\quad
x_\tau=(1-\tau)\epsilon+\tau A.
$$

则 $x_0=\epsilon$、$x_1=A$。对生成时间求导：

$$
u_\tau=\frac{dx_\tau}{d\tau}=A-\epsilon.
$$

网络读入 $x_\tau,\tau,c$，最小化：

$$
\mathcal L_{FM}
=\mathbb E_{c,A,\epsilon,\tau}
\left[\|v_\theta(x_\tau,\tau,c)-(A-\epsilon)\|^2\right].
$$

各项形状：$A,\epsilon,x_\tau,v_\theta,u_\tau$ 都是 `[B,N_a,d_a]`；$\tau$ 是 `[B]`，通过广播成为 `[B,1,1]`。它控制整段 chunk 的噪声程度，不是每个真实动作的时间戳。

教学推导可以用 $\tau\sim U[0,1]$；π0 实际使用偏向 noisy 区域的非均匀时间采样，不能把 uniform 写成完整论文配方。

## 3. 为什么 target 知道 A，推理网络却不用知道 A

训练时 $A$ 是示范标签，用来构造 noisy input 和 loss target；推理时没有专家动作，只提供噪声和观测。

平方损失下，理想网络学到：

$$
v^*(x,\tau,c)=\mathbb E[A-\epsilon\mid x_\tau=x,\tau,c].
$$

单个训练样本的 $A-\epsilon$ 是监督用的条件向量场；网络学到的是给定当前变量的平均场。不要把“拿到测试专家 A 后直接积分常速度”当成训练出的生成模型。

也不要因这里出现条件期望，就断言它退化成单步 MSE 的动作均值。期望条件中有随时间变化的 $x_\tau$，而采样从不同噪声出发，整个动力系统可以生成不同动作模态。

## 4. 训练只采一个噪声等级，推理才反复迭代

概念伪代码：

```text
training:
  get c and expert A
  sample epsilon and tau
  x = (1-tau)*epsilon + tau*A
  target = A-epsilon
  predict v(x,tau,c)
  loss → backward → optimizer step

sampling:
  x = fresh Gaussian noise
  encode condition c
  repeat K times:
    predict v(x,tau,c)
    x = x + delta*v
    tau = tau + delta
  return x as normalized action chunk
```

训练不是通过完整 $K$ 步 rollout 后再和 $A$ 做一次 MSE。增加 inference steps 也不等于增加训练样本数量。

## 5. Euler 法：先手算，再对代码

采样 ODE：

$$
\frac{dx_\tau}{d\tau}=v_\theta(x_\tau,\tau,c),\qquad x_0\sim\mathcal N(0,I).
$$

显式 Euler 更新：

$$
x_{\tau+\Delta\tau}=x_\tau+\Delta\tau\,v_\theta(x_\tau,\tau,c).
$$

**仅用于检验符号的端点已知例子**：$\epsilon=1,A=3$，路径速度恒为 2。取 $K=4,\Delta\tau=0.25$，得到 $1\to1.5\to2\to2.5\to3$。

这里常速度已知，因此 Euler 一步也能精确到端点。真实学到的场通常依赖 $x,\tau,c$；增加步数可能降低离散化误差，但不能修复错误条件、错误动作单位或模型误差，也不保证成功率单调提升。

## 6. openpi 使用数据 → 噪声的时间坐标

固定 revision 的 `compute_loss` / `sample_actions` 采用：

$$
x_s=(1-s)A+s\epsilon,\qquad u_s=\epsilon-A.
$$

$s=0$ 是数据，$s=1$ 是噪声。采样从 $s=1$ 积分到 0，因此 $\Delta s<0$：

$$
x_{s+\Delta s}=x_s+\Delta s\,v_\theta(x_s,s,c),\qquad\Delta s=-1/K.
$$

令 $s=1-\tau$，就有：

$$
u_s=-u_\tau,\qquad \Delta s=-\Delta\tau.
$$

符号和时间方向同时变，结果一致。上述标量例子仍是 $1\to1.5\to2\to2.5\to3$，因为每次增加 $(-0.25)(-2)=0.5$。

该版代码的 time 采样为 $s=0.999b+0.001$，$b\sim Beta(1.5,1)$，偏向较大 $s$，即较 noisy 的输入。采样分布改变哪些噪声区域更常参与训练，不改变这条线性路径的导数。

注意两个代码细节：loss 先在动作维度求均值，保留 batch/horizon 维；padding 和最后如何 reduction 要继续查调用链。不能看到一个 `.mean()` 就认定所有 mask 都已处理。

## 7. 本地 v1 原文中的公式不一致

π0 v1 PDF 第 5 页先写 $x_\tau=\tau A+(1-\tau)\epsilon$，随后却写目标为 $\epsilon-A$，又描述从 0 到 1 正向积分。这三项按字面不能同时成立。π0.5 v1 §IV-B 也沿用了类似的路径/target 符号。

此外，按该显式路径，给定 $A$ 的噪声协方差应为 $(1-\tau)^2I$；π0 v1 的相邻 Gaussian 表达式写成了 $(1-\tau)I$。本专题以显式路径求导和端点检验为准。这是对数学一致性的核对，不把 PDF 抽取结果直接当成排版错误：已经检查了原 PDF 页面。

遇到新实现的固定检查顺序是：**噪声在哪端 → 路径怎么写 → 求导得到 target → dt 的符号 → 用一维端点验证**。不要仅凭变量名叫 `time` 或 loss 叫 diffusion 就推断方向。

## 8. FM loss 和 BC NLL 的关系

π0 仍是在示范数据上做 imitation learning；它没有因使用 flow 而变成 RL。训练监督来自专家动作。

但不能把 flow loss 的原始数值直接叫作 action NLL。它是向量场回归目标。若想从连续流计算密度，需要额外处理变量变换和场的 divergence；本课程不需要为控制采样计算 likelihood。

因此比较 Gaussian BC、token policy、flow policy 时，分别看各自拟合指标，再用相同 rollout protocol 比较成功率、时延、动作平滑性和失败模式。

## 9. Diffusion、flow、AR 到底比较什么

| 问题 | 常见 diffusion policy | 本课线性 flow | token AR |
|---|---|---|---|
| 训练输入 | 带噪动作、noise level、条件 | 路径中间点、flow time、条件 | prefix 与前面的目标 tokens |
| 监督 | noise/score/velocity 等，依参数化而定 | 路径导数 | next-token ID |
| 推理 | 按选定 scheduler 逐步去噪 | 数值积分向量场 | 逐 token 解码 |
| 时序位置 | 所有 chunk 位置可共同处理 | 所有 chunk 位置可共同处理 | token 序列顺序不一定对应真实时间 |

Diffusion 与 flow 的数学联系很深，不能绝对说“一个随机一个确定”。本课比较的是具体训练路径、预测量和采样器。采样步数与 wall-clock 也不能直接互换：每步成本、cache、batch 和硬件都影响总时延。

## 10. 机制实验与答案

在 Colab 运行 [Pi_mechanisms.ipynb](../../labs/notebooks/Pi_mechanisms.ipynb)（[原脚本](../../labs/pi_series_mechanics.py) 保留）。其中 `flow` 部分用固定端点做有限差分和正反时间积分，并故意只翻转 target 符号构造反例。另有 [ML06](../../labs/notebooks/ML06_diffusion_flow.ipynb) 实际学习一个小型 conditional velocity network；两种实验的证明范围不同。

回答：

1. 为什么两种自洽约定结果相同？
2. 只翻转 velocity、不翻 dt，会到哪里？对 $\epsilon=1,A=3$ 是 $-1$。
3. 为什么端点积分正确不能证明模型会生成机器人动作？因为这里没有学习网络，且已知端点。
4. 为什么 noise 和 target 必须与 action shape 一致？
5. 若 $K$ 从 10 改为 5，哪些计算减少，哪些不变？prefix 编码原则上不变，expert 调用次数减少，采样误差需重新评估。

英文口述：

> Derive the flow-matching target from the interpolation path. Then explain how the target and integration direction change under the openpi time convention.

通过标准：合上笔记推导两种约定，手算一个 Euler step，解释为什么这个教学实验不是 VLA 复现。下一课：[P3：FAST](03_fast_action_representation.md)。
