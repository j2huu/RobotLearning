# 第 4 课：概率视角的 BC 与训练目标

本课把第 1–3 课已经接触的 MSE、CVAE、diffusion、flow 和 action policy 放回同一个 ML 框架。重点是学会从“模型学什么分布”推出“为什么使用这个 loss”。

配套实验：[Lab 01](../labs/01_probabilistic_bc.py)。课程计划：[第 4–19 课](../planning/course_plan.md)。概率背景参考 [Probabilistic Machine Learning](https://probml.github.io/pml-book/) 中 conditional probability、maximum likelihood 与 regression 相关主题。

## 1. 输入、目标和模型参数

先用单步动作表示：

$$
\mathcal D=\{(o_i,a_i)\}_{i=1}^N,
\qquad
\pi_\theta(a\mid o).
$$

- $o_i$：第 $i$ 个训练样本中的 observation，可包含图像、state、force、history、language。
- $a_i$：对应的专家动作。
- $\theta$：模型中可学习的参数。
- $\pi_\theta(a\mid o)$：给定观测后模型对动作的条件分布。

下标 $i$ 是训练样本编号，不自动等于机器人时间 $t$。使用 action chunk 时把 $a$ 替换成 $A_t\in\mathbb R^{H\times d_a}$；是否假设不同时间/维度独立，需要由模型的 likelihood 定义决定。

BC 的监督目标是专家 action，不是一般意义上的 $(q_{actual}-q_d)$。后者通常描述控制器的 tracking error；BC loss 比较预测动作和示范动作。只有当 action 恰好是 joint position target 时，两种表达中的变量才可能有相似外观。

## 2. 从 likelihood 到 NLL

给定 dataset，maximum likelihood estimation 选择能给示范动作较高概率的参数：

$$
\theta^*
=\arg\max_\theta\prod_{i=1}^N\pi_\theta(a_i\mid o_i).
$$

取对数，把乘积变成求和；再取负号，得到适合最小化的 negative log-likelihood：

$$
\mathcal L_{NLL}(\theta)
=-\frac1N\sum_i\log\pi_\theta(a_i\mid o_i).
$$

episode 中的数据通常相关。实际训练仍经常使用逐样本或逐 chunk 的目标，但不能据此把相邻帧当作独立证据，或者随机分帧后声称验证了跨 episode 泛化。

## 3. 为什么 MSE 对应 Gaussian likelihood

假设模型预测动作的 mean，并把 uncertainty 固定为各维独立、相同方差的 Gaussian：

$$
\pi_\theta(a\mid o)
=\mathcal N\bigl(a;\mu_\theta(o),\sigma^2 I\bigr).
$$

- $\mu_\theta(o)$：模型预测的 mean，形状与 action 一致。
- $\sigma$：这里是固定 standard deviation。
- $I$：identity matrix，表示没有建模各 action 维度之间的协方差。

单个样本的 NLL 为：

$$
-\log\pi_\theta(a\mid o)
=\frac{1}{2\sigma^2}\|a-\mu_\theta(o)\|_2^2
+\frac{d_a}{2}\log(2\pi\sigma^2).
$$

当 $\sigma$ 固定时，第二项与 $\theta$ 无关，所以最小化 NLL 等价于最小化 MSE。

因此 MSE 有概率假设：一个给定 observation 对应一个单峰、固定方差的分布。它可以是有效 baseline，但不一定适合高度多峰的示范。

若学习 diagonal variance，则目标还包括：

$$
\mathcal L
=\sum_j\left[
\frac{(a_j-\mu_j(o))^2}{2\sigma_j^2(o)}
+\log\sigma_j(o)
\right]+\text{const}.
$$

残差项和 $\log\sigma$ 共同约束 uncertainty；只增加方差不能无限降低 NLL。用 log-standard-deviation 参数化可以保持 $\sigma>0$。

## 4. 为什么会生成专家没有执行过的平均动作

平方损失的总体最优预测器是 conditional mean：

$$
f^*(o)=\mathbb E[a\mid o].
$$

推导时，对固定 $o$ 的风险求导：

$$
R_o(f)=\mathbb E[(a-f)^2\mid o],
\qquad
\frac{\partial R_o}{\partial f}=2\bigl(f-\mathbb E[a\mid o]\bigr).
$$

若同一观测下专家以相同概率选择 $a=-1$ 或 $a=+1$，则：

$$
\mathbb E[a\mid o]=0.
$$

$0$ 最小化 MSE，却不属于任何一个专家 mode。对于绕障碍，两条安全轨迹的平均甚至可能撞上障碍。

不是所有看起来不同的轨迹都构成相同条件下的多峰性。若图像、history、language 或任务阶段能区分两种行为，则提供这些条件后，分布可能接近单峰。需要先检查是否遗漏了重要 observation，再决定是否需要更复杂的生成模型。

## 5. 学习 variance 能不能解决多峰

单个 Gaussian 可以把 variance 变大，但仍然只有一个 peak。对于 $-1/+1$ 的示范，它可能学成 mean 为 0、standard deviation 约为 1 的宽分布。这表示“不确定”，却不能准确表示“两条明确的动作分支”。

Gaussian mixture 则可以写成：

$$
\pi(a\mid o)
=\sum_{k=1}^K w_k(o)\,
\mathcal N\bigl(a;\mu_k(o),\sigma_k^2(o)\bigr),
\qquad
\sum_k w_k=1.
$$

Mixture、CVAE、diffusion 和 flow 都能用于构造更灵活的条件动作分布，但训练稳定性、表示容量、采样策略、时序一致性和计算代价各不相同。第 8–9 课逐项推导。

## 6. 离线 action error 与任务成功率

如果新的采样动作和 dataset 专家动作选择了不同分支，它可能有很大 pointwise MSE，但仍是有效动作。反过来，平均动作可能有更小 MSE，却在 rollout 中失败。

所以要区分：

- **fit**：held-out NLL、reconstruction/denoising/flow loss；不同训练目标的原始数值不能直接跨模型排序。
- **distribution quality**：mode coverage、样本是否接近示范支持集。
- **control quality**：rollout success、接触力、平滑性、时延和 failure modes。

Lab 01 的 mode-compatibility 指标只检验合成动作分布；它没有动力学、接触或控制器，不能等同于机器人成功率。

## 7. 从 objective 到实际训练

最小训练闭环为：

```text
episode split
  → fit normalization on training data
  → construct observation/action batches
  → forward
  → compute loss
  → backward
  → optimizer step
  → held-out evaluation
```

本课先认识梯度更新：

$$
\theta_{n+1}
=\theta_n-\alpha\nabla_\theta\mathcal L(\theta_n).
$$

下一课深入 Adam、learning rate、regularization、batch、gradient clipping、train/eval mode、overfitting 和数据质量。首先验证 shape、units、mask 与 split，再把模型容量作为诊断方向。

## 8. 本课实验

在工作区根目录运行：

```bash
python robot_learning/labs/01_probabilistic_bc.py --steps 800 --seed 7 --output robot_learning/labs/results/lab01_seed7.json
```

合成数据的真实机制是：

$$
a=0.5o+b+\epsilon,
\quad b\in\{-1,+1\},
\quad\epsilon\sim\mathcal N(0,0.05^2).
$$

$b$ 对同一 synthetic episode 保持不变，但不提供给单步 policy。两个分支均衡，train/validation 按 episode 划分。这里 episode 只是用于组织合成样本，没有真实状态转移模型。

脚本训练 deterministic MSE、single-Gaussian NLL 和 two-component-mixture NLL 三个 policy，报告 held-out mean MSE、可比较的 Gaussian/mixture NLL、采样动作接近专家 mode 的比例，以及 $o=0$ 时的分布参数。

本课任务：

1. 运行脚本，解释为何 MSE policy 的 mean 靠近两种动作中间。
2. 比较 single Gaussian 与 mixture 的 NLL 和 sampled mode-compatibility。
3. 解释为什么 mixture 的 mean prediction 仍然可能无效，但 sampled action 接近专家 mode。
4. 修改 seed，再比较结论是否保持；必要时增加 steps 检查优化是否收敛。
5. 思考：若把 branch 信息或可靠 history 加入 observation，多峰性是否仍然存在？

## 9. 检查题

1. $\pi_\theta(a\mid o)$ 与 $\mu_\theta(o)$ 有何不同？
2. MSE 对应哪种 likelihood 假设？学习 $\sigma$ 后 loss 如何改变？
3. 为什么 squared loss 的最优输出是 conditional mean？
4. 单 Gaussian 的大 variance 为什么不能准确表示两个 modes？
5. 为什么有效的 sampled action 可能比 mean action 有更大 pointwise MSE？
6. 这个实验能支持哪些结论，不能支持哪些机器人控制结论？

英文口述题：

> Explain behavior cloning as conditional density estimation. Under which assumptions does it reduce to mean squared error, and why can that objective produce an action that the expert never demonstrated?

下一课：[第 5 课：训练、数据与观测](05_training_data_and_observation.md)。
