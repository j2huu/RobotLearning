# P6：补齐 VLA 工程基础，并形成可展示的项目

前课：[P5](05_openpi_engineering.md)。这节把 Pi 系列连回更通用的 Robot Learning 能力。重点是能解释并应用基础；完整 RL、表示学习和世界模型课程仍见 [主线计划](../../planning/course_plan.md)。

岗位取向与来源见 [能力地图](../../resources/vla_job_competency_map.md)；口述练习见 [面试题卡](interview_drill.md)。

## 1. 用六个问题比较 VLA，而不是背模型名

比较任何新模型时，写下：输入条件、预训练来源、动作表示、训练目标、推理循环、执行反馈。

| 方法/版本 | 动作与训练 | 最应该比较的设计 |
|---|---|---|
| ACT 原始方案 | CVAE chunk reconstruction + KL | latent 如何使用、temporal ensemble、示范数据要求 |
| Diffusion Policy | noisy chunk 的去噪目标 | 多峰建模、sampler、receding horizon |
| OpenVLA 原始方案 | 离散动作 AR / CE | VLM 迁移、动作量化与解码接口 |
| Octo | 通用机器人策略，含 diffusion action head | 从多机器人数据学表示，支持语言/goal conditioning |
| π0 | VLM + flow expert | token routing、cache、连续 chunk 与预训练配方 |
| π0-FAST | 压缩 action tokens / CE | teacher forcing、token 长度与推理耗时 |
| π0.5 v1 | 异构离散预训练 + 连续 expert post-training | co-training、subtasks、开放场景证据 |

原始资料：[ACT](https://tonyzhaozh.github.io/aloha/)、[DP](https://diffusion-policy.cs.columbia.edu/)、[OpenVLA](https://openvla.github.io/)、[Octo](https://octo-models.github.io/)。这些项目还有后续变体，比较时注明论文/代码版本，不能把“原始 OpenVLA 不支持某机制”推广到所有后续方法。

Octo 使用语言条件，但并非从大型 VLM 直接迁移而来的同类架构。说清具体输入和初始化，比争论一个宽泛的 VLA 标签更有用。

## 2. Transformer：四个必会点

**计算图**：Q/K 的点积决定读取权重，V 提供被聚合内容；residual、MLP 和 normalization 继续更新表示。只画 attention 不等于完整 Transformer layer。

**位置**：相同 token 顺序若没有位置/结构信息，模型不能自然知道“哪一个动作先执行”。真实系统还需区分相机身份、序列位置和物理 timestamp。

**复杂度**：普通 dense attention 的 score matrix 随序列长度 $S$ 按 $S^2$ 增长；推理还有参数、K/V、activation 和实现开销。减少 image tokens 与减少生成 steps 影响的是不同部分。

**mask 与因果性**：先写允许读取的逻辑关系，再转换成 API。不同 API 的 bool mask 可能用 `True` 表示允许或屏蔽，必须核对。一个最小 perturbation test 比只看矩阵形状更有说服力。

练习：画 P1 的三块 mask，加入 padding camera，再说明为什么 C 的 loss 仍可能通过 P 的 K/V 反传到 VLM。

## 3. 概率、优化与训练诊断

已有 [第 4 课](../04_probabilistic_bc_and_training.md) 推导固定方差 Gaussian NLL → MSE。这里继续区分：

- **Aleatoric uncertainty**：同样可见条件下的多解/噪声；更丰富条件有时能减少它。
- **Epistemic uncertainty**：模型对未覆盖区域缺少知识；单次 flow sample 或 token 熵并不自动提供校准的 OOD 置信度。
- **Overfitting**：训练拟合好，独立数据/rollout 差；先检查 split 和数据，再判断是否需要正则化。

优化中需要能解释 learning rate、warm-up、AdamW、gradient clipping 和 loss reduction。`zero_grad`、`backward`、`step` 的顺序，梯度累积是否正确缩放，也是常见代码追问。

`eval()` 改变 dropout/BatchNorm 等模块行为；不等于关闭 autograd。冻结参数、停止梯度、在 optimizer 中不更新参数也不是完全同义：optimizer state、weight decay、反向所需 activation 都可能不同。

**诊断顺序**：小 batch 能否 overfit → shape/单位/标签 → gradients → train/validation gap → rollout gap。不要一看到 loss 不降就先换更大 backbone。

练习：把动作单位从 rad 意外改成 degree，会怎样影响归一化、梯度尺度和执行？如果输入与标签都做了 normalization，为什么错误仍可能在部署反归一化时出现？

## 4. LoRA、冻结与 sidecar

LoRA 的基本形式：

$$
W'=W_0+\frac{\alpha}{r}BA,\qquad
A\in\mathbb R^{r\times d_{in}},\quad
B\in\mathbb R^{d_{out}\times r}.
$$

低秩增量的可训练参数为 $r(d_{in}+d_{out})$，对比 full weight 的 $d_{out}d_{in}$。原始来源：[LoRA](https://arxiv.org/abs/2106.09685)。

**教学算例**：$d_{in}=d_{out}=2048,r=8$。LoRA 两矩阵有 32,768 个参数，full matrix 有 4,194,304 个，相比为 $1/128$。这只是单矩阵参数比例，不是整模型显存或训练耗时的缩减比例。

| 决定 | 改变什么 | 验证点 |
|---|---|---|
| full fine-tuning | 大范围更新模型参数 | optimizer、显存、过拟合与语义保持 |
| LoRA | 指定层的低秩参数增量 | target modules、rank、checkpoint 装配 |
| frozen backbone + head | backbone 不更新，head 学习 | 表示是否适合目标动作、梯度路径 |
| FP2 sidecar | 原动作策略下游的 interaction regulation | 输入 context、实时 wrench、控制器接口 |

FP2 的“adaptor”不自动是 LoRA；π0.5 的 attention 隔离不自动是 KI；KI 也不等于整个 VLM 永久冻结。这三种误会可用 forward graph 与 backward graph 分别画出来纠正。

## 5. 训练工程的基础深度

以下是工程原理与学习任务，具体 API 以使用版本的官方文档为准：[PyTorch AMP](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)、[PyTorch Distributed](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)。

| 技术                       | 解决的问题                              | 不能误解成                   |
| ------------------------ | ---------------------------------- | ----------------------- |
| mixed precision          | 合适算子的精度/速度/显存权衡                    | 所有运算都能无损改低精度            |
| gradient accumulation    | 小 microbatch 累积到较大 effective batch | 降低单样本 activation 的必然方法  |
| activation checkpointing | 少存激活，以重计算换显存                       | 保存训练 checkpoint 到磁盘     |
| DDP                      | 各副本处理不同数据，同步梯度                     | 每卡只保存一小片完整参数            |
| FSDP                     | 分片参数/梯度/optimizer state 等          | 自动消除通信与所有 activation 成本 |

模型能放进 inference 显存，不等于 full fine-tuning 也放得进：训练还需要 gradients、optimizer states 和 activation。LoRA 会减少部分成本，仍需实测。

本阶段验收是能读一张 profiler trace，分辨 dataloader、CPU preprocessing、device transfer、attention、expert loop 和通信。分布式集群/CUDA kernel 优化属于进一步岗位方向，不是开始学习 π0 的必要前置。

## 6. 机器人基础：action contract 后面必须有几何

末端位姿 $T\in SE(3)$ 包含旋转和平移。若 $T_{AB}$ 将 B 系坐标映到 A 系，则组合为 $T_{AC}=T_{AB}T_{BC}$；左乘/右乘增量的含义不同，不能直接给旋转矩阵各元素加一个 delta。

关节速度和末端 twist 的局部关系：

$$
\xi=J(q)\dot q.
$$

反解时可能遇到奇异性、冗余和限位，不能随意把 $J^{-1}$ 当作总是存在。常见阻尼最小二乘示意：

$$
\dot q=J^\top(JJ^\top+\lambda^2I)^{-1}\xi.
$$

Twist/wrench 的 frame、参考点与功率配对见 [第 3b 课](../03b_fp2_adaptor_twist_wrench.md)。F/T sensor 的 force 与 torque 不能只旋转前三维；参考点变化带来力臂项。

**控制接口**：策略输出 joint position、Cartesian delta、desired wrench、stiffness 时，controller 要完成的事情不同。PD、impedance、admittance、hybrid force-position 并非同义词。50 Hz 的 learned regulator 也不能代替 1 kHz loop 的稳定性论证。

练习：π0 生成粗略插入轨迹，但孔口存在 2 mm 横向偏差。列出几何重规划、force-conditioned correction 和显式法向力调节各能修复什么；说明固定粗轨迹何时已不可行。

## 7. IL、DAgger、RL 与 world model 的最低知识闭环

| 方法 | 监督/信息来源 | 主要问题 |
|---|---|---|
| BC | 固定示范的 expert actions | 拟合专家行为，rollout 分布可能偏移 |
| DAgger | 在学习者访问状态上询问 expert label，并聚合数据 | 扩展训练覆盖，减少序列分布偏移 |
| RL | 与环境交互或已有 transition/reward 数据 | 最大化长期回报，需处理探索/估值 |
| offline RL | 固定 transition/reward dataset | OOD actions 与价值过估计风险 |
| dynamics / MPC | 转移模型与任务成本 | 预测候选动作后果，滚动重规划 |

DAgger 原始来源：[Ross et al.](https://proceedings.mlr.press/v15/ross11a.html)。本仓库的 CR-DAgger 加上顺应干预与 residual 接口，不能仅因有人类纠正就叫 RL。

最小 Bellman 关系：

$$
Q^\pi(s,a)=\mathbb E[r+\gamma\mathbb E_{a'\sim\pi}Q^\pi(s',a')].
$$

Actor 选择动作，critic 估计回报。监督训练的 π0 action expert 不需要 critic；“expert”这个名字不是 actor-critic 中的 critic。

Forward dynamics 预测 $s'$，policy 预测 action；world model 的生成对象可以是未来状态/latent/图像，不能因为策略生成未来 action chunk 就叫它 world model。MPC 每次执行前缀后用新观测重规划，其模型误差还会影响长 rollout。

本节仅建立接口地图。需要独立 RL 推导与实验时，回到主线第 16–18 课；不能把会背 Bellman 方程写成已掌握在线机器人 RL。

## 8. 泛化与评估：把“好”拆成可检验假设

先固定 benchmark contract：observations、action semantics、controller、reward、termination、success 和 reset distribution。再分开测试 seen/unseen instance、object category、scene、task composition、embodiment、摩擦/质量/时延。

公平比较尽量控制数据、训练预算、初始化、推理预算和低层 controller；无法全部匹配时明确报告。不要把更大的模型、更多数据、不同 controller 同时改变后的差异全部归功于新 loss。

对 rollout 报告试验次数、seed/reset 条件、完整成功率和 progress、耗时、干预次数、失败分型；接触任务再报告 force peak、tracking error、contact loss 和反馈延迟。

视频帧不是独立样本，同一家庭里的许多 episodes 也不等于同样数量的新家庭。置信区间/重采样单位应匹配 episode、object、scene 或 robot 的研究问题。

## 9. Capstone：一份可以被追问的作品

项目题目示例：**VLA action pipeline 的因果性、时间方向与执行时延审计**。这是课程项目规格，尚未代表学习者已完成。

**基础交付，不需要大模型/GPU：**

1. 完成本专题机制实验，解释每个“正确结果”和故意构造的反例。
2. 为固定 openpi revision 交付 raw data→loss→sampling→command 的接口报告。
3. 在现有 Lab 02 上提出一个实际可检查的数据错误；给出失败证据和修复验证。
4. 提交 π0/FAST/π0.5 比较表、六张论文证据卡，以及两种失败的定位流程。
5. 准备 10 分钟演示：问题 1 分钟、公式/结构 3 分钟、代码和结果 3 分钟、局限与下一步 3 分钟。

**模型扩展，有资源时再做：** 同一小数据集比较 baseline 与一种明确变化，例如 full vs LoRA、不同 flow steps、语言扰动或 state 表示。记录参数、split、训练/推理预算和多个受控试验。

**硬件扩展，有机器人时再做：** 固定 controller 与场景，测量延迟与 repeated rollout；对比无 force、低频 force、高频 feedback。不得把机制实验的数值当成这些真实实验结果。

建议报告字段：

```text
Question / hypothesis:
Source revision and data contract:
What I implemented:
Baseline and controlled variables:
Command / seed / config / environment:
Metrics and independent evaluation unit:
Observed result and failure case:
What the evidence does not establish:
My contribution versus existing code/team work:
```

## 10. 求职准备的完成标准

独立画图、推导、跑代码和解释失败；再做 [面试题卡](interview_drill.md)。每题按“定义 → 机制 → 取舍 → 证据”组织，而不是背一个模型宣传句。

简历先写真实完成的动作和可复查证据。例如完成标准库实验后可以说“实现并验证 flow 时间约定、attention 泄漏和 delay arithmetic 的教学检查”；不能写“复现 π0.5 并实现开放世界机器人控制”。

完成专题后，把薄弱点映射回原第 6–19 课；求职主线优先补数据/训练/部署，再把已有力觉与 FP2 作为差异化研究经验。还未掌握的部分保留为待学习，不用扩充论文数量掩盖基础缺口。
