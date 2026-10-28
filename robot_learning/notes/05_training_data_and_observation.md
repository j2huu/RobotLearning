# 第 5 课：训练、数据与观测——怎样把 demonstration 变成可靠的 policy

上一课：[概率视角的 BC](04_probabilistic_bc_and_training.md)。配套实验：[Colab Lab02：episode windows](../labs/notebooks/Lab02_episode_windows.ipynb)，直接上传运行；[原脚本](../labs/02_episode_windows.py) 保留为历史入口。

第 4 课解释了模型为什么最小化 MSE/NLL。本课继续问：loss 里的 observation 和 expert action 从哪里来？训练集、mask、时间对齐或 action 定义有错时，优化正确的公式也可能学到错误的行为。

本课采用“概念 → 数学 → 数据接口 → 代码 → 论文检查”的顺序；英文术语保留，用于之后的口述。

## 1. State、observation 与 history

为避免符号冲突，本课把完整环境状态记为 $x_t$，把机器人能测到的 observation 记为 $o_t$。

完整状态可能包括对象 pose/velocity、摩擦、接触形变、robot configuration 与其它变量。真实传感器只提供其中的一部分：

$$
x_{t+1}\sim p(x_{t+1}\mid x_t,a_t),
\qquad
o_t\sim p(o_t\mid x_t).
$$

如果 observation 不能完整决定未来，就要按 partially observable 问题考虑。history 可写为：

$$
h_t=(o_{t-L+1:t},a_{t-L:t-1}),
\qquad
\pi_\theta(a_t\mid h_t).
$$

history 帮助估计速度、任务阶段、接触趋势等，但不保证完全恢复真实状态。GRU、Transformer 或显式 state estimation 都可以利用历史。

例子：同一张末端停在孔口的图像，可能对应“尚未接触”“受到侧向卡滞”或“正在反向退出”。单帧图像不足以区分，force/history 可以补充。

第 4 课的两个 action branches 若能从可靠 history 或任务信息区分，条件分布会变得更简单。因此多峰性也可能源于缺失观测，不只是 action head 容量不足。

## 2. Episode、transition 与监督样本

一条 demonstration episode 可以表示为：

$$
\tau=(o_0,a_0,o_1,a_1,\ldots,o_{T-1},a_{T-1},o_T).
$$

常见监督数据是 transition $(o_t,a_t,o_{t+1})$ 或 history/chunk pair：

$$
O_t=o_{t-T_o+1:t},
\qquad
A_t=a_{t:t+H-1}.
$$

需要明确：$o_t$ 应是产生 $a_t$ 之前可获得的信息。硬件日志可能先发 command，再读取传感器；不能只依据两个数组的相同 index 推断其因果顺序。

同一 episode 的相邻 frames 高度相关。1000 个 frame 不等于 1000 个独立任务示例。

## 3. 什么是 observation/action contract

在写 model 前，先定义接口：

| 字段 | 必须写清的内容 |
|---|---|
| image | camera 名称、分辨率、exposure/arrival timestamp、颜色/归一化方式 |
| proprioception | joint/TCP、position/velocity、角度单位、robot configuration |
| wrench | sensor/TCP frame、origin、force/moment ordering、bias/gravity compensation、timestamp |
| language | 当前任务指令还是 subtask；什么时候更新 |
| action | position/velocity/torque/wrench；absolute/delta；joint/Cartesian；单位、frame、frequency |
| episode | task/object/scene/robot IDs、episode boundary、成功/失败或干预记录 |

一个 $[B,H,7]$ tensor 只能告诉你 shape。它可能是关节目标，也可能是 6D Cartesian motion 加 gripper；实际语义必须由 contract 决定。

FP2 中 RFM motion 与 desired wrench 是不同控制目标。日志里还要区分 policy reference、controller command、实际执行状态与 measured wrench，否则监督标签会混淆。

## 4. Train/validation/test 应如何划分

基础划分按 episode：

```text
episodes
  ├─ train: optimization + normalization statistics
  ├─ validation: model/hyperparameter selection
  └─ test: final evaluation after choices are fixed
```

如果随机把相邻 frames 分给 train 和 validation，模型可能在 validation 看到几乎相同的场景和动作。这样的 loss 不能支撑跨 episode 泛化。

更严格的划分由研究问题决定：

- 新对象：按 object ID 隔离。
- 新场景：按 scene/location 隔离。
- 新任务：按任务与指令语义隔离。
- 新 embodiment：按机器人与接口隔离。

数据量相同并不保证独立性。split 还要避免同一原始 demonstration 的重采样、增强版本或相邻 chunk 跨 partition。

Lab 02 只有 train/validation 用于演示接口；真实实验需保留独立 test，不反复依据 test 调参。

## 5. Normalization 的统计只能来自 train

对连续动作可使用：

$$
\mu_j=\mathbb E_{train}[a_j],
\quad
\sigma_j=\sqrt{\mathbb E_{train}[(a_j-\mu_j)^2]},
\quad
\tilde a_j=\frac{a_j-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

验证/测试使用同一组训练统计。模型输出后：

$$
\hat a_j=\tilde a_j\max(\sigma_j,\varepsilon)+\mu_j.
$$

如果在整个 dataset 上拟合统计，validation/test distribution 的信息就进入了 preprocessing。不同 embodiment 的统计也应结合 action 定义处理，不能仅因为维度相同就混合。

gripper、rotation、binary mask 可能需要不同变换；有些实现使用 quantile normalization。本实验只示范连续向量的 standardization。

## 6. 多频率数据如何因果对齐

假设 policy 在时刻 $t$ 使用 force，应选择当时已经到达的样本：

$$
j(t)=\max\{j:\tau_j^{available}\le t\},
\qquad
f_t=f_{j(t)}.
$$

“最近 timestamp”可能选到未来样本。离线用未来 force 预测当前 action 会造成部署无法复现的 leakage；这与 ImplicitRDP 的 causal attention 问题相关。

还要区分 acquisition time 与 arrival time：若样本采集于 100 ms、到达于 120 ms，policy 在 110 ms 不能使用它。本实验用已可用 timestamp；真实 pipeline 要记录或建模传输延迟。

这个 last-available 示例只说明因果性。高频 force 的过滤/降采样仍需考虑 aliasing、峰值、时间窗口和 filter delay。普通线性插值若依赖尚未到达的右端点，也可能违反在线因果性。

FP2 的缓存 task context 可以低频更新，force/proprio history 则必须随 fast step 更新。它们的有效期和时间语义不同。

## 7. Window、episode boundary 与 padding mask

若 $T_o=3,H=4$，在 $t$ 取：

```text
observation: [o_(t-2), o_(t-1), o_t]
action:      [a_t, a_(t+1), a_(t+2), a_(t+3)]
```

开头没有足够 history，末尾没有完整 action chunk。可以丢弃边界样本，或明确 padding 并提供 mask。对于t = 3，$A_3=[a_3,a_4,a_5,a_6]$, 没有提供$a_5, a_6$，所以需要padding，也即是$m = [1, 1, 0, 0]$，padding的Action不参与Loss。
 
本实验 history 在开头重复首帧并提供 observation mask，action 在末尾填零并提供 action mask。窗口始终从一个 episode 取数据，不能跨到下一个任务。

masked action loss 示例：

$$
\mathcal L
=\frac{\sum_{b,h,j}m_{b,h}\,\ell(\hat a_{b,h,j},a_{b,h,j})}
{d_a\sum_{b,h}m_{b,h}}.
$$

| 符号               | 含义                         |
| ---------------- | -------------------------- |
| $b$              | Batch 中第 \(b\) 个样本         |
| $h$              | Action Chunk 中第 \(h\) 个时间步 |
| $j$              | Action 的第 \(j\) 个维度        |
| $\hat a_{b,h,j}$ | 模型预测的 Action               |
| $a_{b,h,j}$      | Demonstration 中的真实 Action  |
| $m_{b,h}$        | Mask，决定该时间步是否有效            |
| $d_a$            | Action Dimension，例如 7      |
| $\ell$           | 单个 Action 分量的误差函数          |

分母只计有效元素。masked padding 不参与 loss 或训练统计；attention 是否屏蔽 padding 则由模型另外处理，单有 loss mask 并不够。

## 8. 梯度下降、Adam 与 regularization

minibatch 提供 objective 的梯度估计。基本 SGD 为：

$$
g_n=\nabla_\theta\mathcal L_{batch}(\theta_n),
\qquad
\theta_{n+1}=\theta_n-\alpha g_n.
$$

Adam 跟踪梯度的一阶/二阶滑动平均：

$$
m_n=\beta_1m_{n-1}+(1-\beta_1)g_n,
\qquad
v_n=\beta_2v_{n-1}+(1-\beta_2)g_n^2.
$$

进行 bias correction 后，使用：

$$
\theta_{n+1}=\theta_n-\alpha\frac{\hat m_n}{\sqrt{\hat v_n}+\varepsilon}.
$$

learning rate 过大可能震荡，过小可能训练缓慢。gradient clipping 限制异常梯度，但不修复错误数据。weight decay、dropout 和数据增强影响泛化，需结合验证结果选择。

`model.train()`/`model.eval()` 会改变 dropout/BatchNorm 等行为；`eval()` 不自动关闭 autograd，评价还应使用 `no_grad()` 或 inference mode。第 4 课实验使用 MLP，无需从这个例子推断所有大型模型的训练行为。

## 9. 训练诊断的顺序

| 现象 | 优先检查 |
|---|---|
| train loss 不降 | 标签、units、mask、归一化、梯度是否存在、learning rate；尝试小 batch overfit |
| train 好而 validation 差 | split、数据量/多样性、过拟合、augmentation、capacity |
| 离线 loss 好但 rollout 差 | covariate shift、动作接口、latency、history、controller、任务指标 |
| free-space 好而 contact 差 | force 因果对齐、模态是否被用、contact 样本比例、反馈频率、力控接口 |

小 batch overfit 是检查 pipeline 能否学习固定样本的诊断，不是泛化或闭环性能证明。

## 10. 把问题带回论文与 benchmark

查看本地 CR-DAgger 的数据来源与 correction 分布；查看 ImplicitRDP 的 causal force sequence；查看 FP2 的 RFM/force-policy/低层控制接口。

资源列表中的 datasets 与 simulators 用于检查数据和评价覆盖范围，具体入口见 [资源地图](../resources/resource_map.md)。[robosuite](https://robosuite.ai/) 提供 robot-learning 仿真和 benchmark；[SimplerEnv](https://github.com/simpler-env/SimplerEnv) 面向 manipulation policy 的 simulation evaluation。第 15 课再选择具体实验 protocol。

现在重点不是安装所有环境，而是先能写清任何一个环境的 observation、action、controller、reward 与 success definition。

## 11. Lab 02：从 episode 到 batch

上传 [Lab02 notebook](../labs/notebooks/Lab02_episode_windows.ipynb) 到Colab，从第一个cell顺序运行。所需数据与函数已内嵌，不要求安装本地环境；[旧脚本](../labs/02_episode_windows.py) 仅作来源参考。

Notebook展示：

1. 按 episode 分 train/validation。
2. 为每个 control step 查找最新可用的 force。
3. 构造 observation history、action chunk 和 masks。
4. 仅用 train actions 拟合 normalization。
5. 检查有效动作的 normalize/denormalize round trip。
6. 打印 batch shape、padding 数量、force age 和统计量。

这是 Lab 02 的数据窗口阶段；图像 preprocessing、异构 action/frame conversion 在第 13 课扩展。脚本没有训练 robot policy，也没有真实接触 dynamics。

## 12. 检查题与英文口述

1. 完整环境 state、robot proprioception 和 observation 有什么不同？A: $P \subseteq O \subseteq S$  
2. 同一图片对应两个动作时，应该先检查哪些遗漏条件？
3. 为什么随机 frame split 不能证明跨 episode 泛化？
4. force 的 acquisition timestamp 与 arrival timestamp 为什么需要区分？
5. 为什么 padding 需要在 loss、统计和 attention 中分别处理？
6. 为什么 validation/test 的 normalization 要复用 train statistics？
7. 小 batch overfit 成功后，仍有哪些问题不能排除？
8. action MSE 很低但机械臂 rollout 失败，你会按什么顺序定位？

英文题：

> Walk me through converting robot demonstrations into training batches. Explain the observation/action contract, causal sensor alignment, episode split, normalization, padding masks, and the gap between offline prediction error and closed-loop success.

下一步先看 [循序学习路线](../planning/vla_learning_path.md)，再进入 [第6课6A：像素到token](06_multimodal_policy_representation.md)。从本课的数据contract继续，先手算RGB到patch embeddings，再逐步加入语言、state与融合；按一个小单元推进。
