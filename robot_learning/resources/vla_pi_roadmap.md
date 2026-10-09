# VLA 与 π 系列系统学习路线

2026-10-09 更新：完成前 3 课的讲解后，当前按 [第 4–19 课执行计划](../planning/course_plan.md) 推进，每周 8–10 小时，补概率/优化、表示、RL、模型学习与代码实验。本文件保留为 VLA 主题地图；具体课号、节奏和验收以执行计划为准。

## 目标

完成这条路线后，应当能够：

- 把 VLA 写成条件策略，并说清 observation、language、state、action chunk 和 embodiment 的含义。
- 从训练目标、动作表示、推理方式和控制频率四个角度比较 ACT、Diffusion Policy、RT/OpenVLA、Octo、π0 与 π0-FAST。
- 推导 π0 的 conditional flow matching loss，并把每个数学量对应到 `openpi` 张量。
- 解释 π0.5 如何利用异构 co-training、high-level subtask prediction 和 web data增强开放世界泛化。
- 判断力/触觉信息应放在 VLM、action expert、residual controller 还是低层力控器，并说明相应的时延和控制后果。
- 面对一个新机器人数据集，设计 observation/action schema、归一化、action horizon、训练与评估方案。

## 先建立统一坐标系

把策略统一写成：

$$
\pi_\theta(a_{t:t+H-1}\mid o_{\le t},\ell,e),\qquad
o_t=\{I_t,s_t,f_t,\ldots\}
$$

- $I_t$：一个或多个相机观测。
- $s_t$：关节、末端位姿、夹爪等本体状态。
- $f_t$：可选的力、力矩或触觉观测。
- $\ell$：自然语言任务指令或中间子任务。
- $e$：embodiment 信息，包括机器人形态、传感器和动作约定。
- $a_{t:t+H-1}$：长度为 $H$ 的 action chunk。

学习任何模型时都用五个问题检查：输入是什么、动作怎样表示、损失是什么、推理怎样进行、动作以什么频率闭环执行。

## 8 个模块

### M0：必要的机器人学习语言（2-3 天）

内容：MDP/POMDP、trajectory、policy、behavior cloning（BC）、distribution shift、DAgger、位置/速度/力矩动作、笛卡尔空间与关节空间。

产出：能从数据集 $D=\{(o_t,\ell,a_t)\}$ 写出 BC 目标；能解释为什么 one-step BC 在多峰动作和闭环误差下容易失败。

本地阅读：

- [CR-DAgger](../../paper/NeurIPS-2025-compliant-residual-dagger-improving-real-world-contact-rich-manipulation-with-human-corrections-Paper-Conference.pdf)：用它理解 covariate shift、on-policy correction 和 residual policy。
- [Making Sense of Vision and Touch](../../paper/1810.10191v2.pdf)：把多模态表征与策略学习分开看。

### M1：Action chunk 与序列策略（3-4 天）

内容：ACT、Transformer policy、chunking、temporal aggregation、open-loop horizon 与 replanning frequency。

产出：能解释为何 chunking 降低有效预测长度，也会引入反馈延迟；能画出“预测 50 步、执行前 $k$ 步、再规划”的时序图。

必补论文：ACT / *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*。

### M2：生成式动作建模（1 周）

内容：多峰动作分布、Diffusion Policy、denoising、conditional flow matching、ODE sampling、autoregressive action tokenization。

核心比较：

| 方法 | 动作表示 | 训练 | 推理 | 关键代价 |
|---|---|---|---|---|
| ACT | 连续 chunk | CVAE reconstruction + KL | 一次 decoder 前向 | latent 使用方式与重叠预测平滑影响多峰表达 |
| Diffusion Policy | 连续 chunk | 去噪/score 类目标 | 多步迭代去噪 | 延迟随采样步数增加 |
| π0 | 连续 chunk | flow matching velocity | 数值积分 ODE | 积分步数与实时性权衡 |
| RT/OpenVLA 类 | 离散 action token | next-token CE | 自回归解码 | 量化误差、序列较长 |
| π0-FAST | 压缩后的离散 action token | next-token CE | 自回归解码 | tokenizer 与序列压缩决定效率 |

必补论文：Diffusion Policy、Flow Matching for Generative Modeling、FAST。

本地阅读：

- [Adaptive Compliance Policy](../../paper/2410.09309v2.pdf)：diffusion action 与动态顺应参数如何组合。
- [Reactive Diffusion Policy](../../paper/2503.02881v3.pdf) → [ImplicitRDP](../../paper/2512.10946v1.pdf)：为什么 chunk-level policy 需要高频修正，以及显式层级与结构性 slow-fast 的取舍。
- [PhaForce](../../paper/2603.08342v1.pdf)：slow planner / fast corrector 与 phase schedule。

### M3：从 VLM 到 VLA（1 周）

内容：image encoder、language tokens、multimodal transformer、action head、RT-1/RT-2、Open X-Embodiment、cross-embodiment normalization、Octo、OpenVLA。

产出：能区分“语义预训练带来的迁移”与“机器人动作数据教会的控制”；能说明 Open X 解决的是数据接口和跨机器人训练问题，而不是自动消除 embodiment gap。

建议顺序：RT-1 → RT-2 → Open X-Embodiment/RT-X → Octo → OpenVLA。

### M4：π0（1 周）

内容：PaliGemma/VLM prefix、robot state、action expert、双向注意力结构、action chunk、conditional flow matching。

需要掌握的最小推导：令 $a\sim p_{\text{data}}(a\mid c)$、$\epsilon\sim\mathcal N(0,I)$，采用线性路径

$$
a^\tau=(1-\tau)\epsilon+\tau a,\qquad \tau\sim U[0,1].
$$

路径速度为 $a-\epsilon$，训练 action expert：

$$
\mathcal L_{\text{FM}}(\theta)=
\mathbb E\left[\left\|v_\theta(a^\tau,\tau,c)-(a-\epsilon)\right\|_2^2\right].
$$

推理时从噪声出发，积分

$$
\frac{d a^\tau}{d\tau}=v_\theta(a^\tau,\tau,c),\quad a^0\sim\mathcal N(0,I),
$$

得到 $a^1$。这里的 $c$ 不是一个抽象标签，而是图像、语言和机器人状态经过模型得到的条件表示。

精读：[π0 论文](https://arxiv.org/abs/2410.24164)。

### M5：FAST 与 π0-FAST（3-4 天）

内容：连续动作频域变换、量化、BPE 压缩、autoregressive decoding；区分“FAST tokenizer”和“π0-FAST policy”。

产出：解释 π0 与 π0-FAST 并不是同一种 action head 换个采样器，而是连续 flow policy 与离散自回归 policy 的差别。

精读：[FAST 论文](https://arxiv.org/abs/2501.09747)。

### M6：π0.5 与开放世界泛化（1 周）

内容：heterogeneous co-training、机器人动作、verbal instruction、object detection、semantic subtask、web data、high/low-level inference。

产出：能解释以下三个层次为何不同：

1. 语义泛化：知道“杯子”“水槽”和“清理”是什么意思。
2. 任务泛化：能把长任务分解成合适的子任务。
3. 物理泛化：能在新的几何、摩擦、接触与机器人动力学下成功执行。

精读：[π0.5 论文](https://arxiv.org/abs/2504.16054)。

### M7：openpi 代码闭环（1-2 周）

按数据流而不是文件顺序阅读：

```text
raw episode
  → dataset config / transforms
  → observation + prompt + normalized action chunk
  → model prefix/suffix
  → loss (flow matching or token CE)
  → checkpoint
  → policy preprocessing
  → action sampling / decoding
  → denormalization
  → robot execution and replanning
```

代码入口：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)。

实践验收：

- 打印一个 batch 中每个字段的 shape、dtype、mask 和归一化范围。
- 从 `compute_loss` 追到 flow target，逐项对应 M4 的公式。
- 从 policy 输入追到 action denormalization，确认机器人真正收到的单位和坐标系。
- 修改一个已有 data config 接入小型自有数据，并跑通单 batch overfit。

## 力觉 / 接触专题线

完成 M0-M2 后插入这条线，效果最好。

### S1：感知与数据采集

阅读：Making Sense of Vision and Touch → ForceMimic → FACTR → ManipForce。

问题：力信号的采样率、坐标系、bias、接触阶段和视觉同步如何影响可学习性？

### S2：动作接口与顺应控制

阅读：Admittance Visuomotor Policy → Adaptive Compliance Policy → EquiContact → Force Policy。

问题：策略输出 pose、wrench、stiffness、admittance gain 或 residual action 时，低层 controller 各自承担什么？

### S3：力觉进入 VLA

阅读：CompliantVLA-adaptor → FD-VLA → CRAFT → FAVLA → ForceVLA2 → AT-VLA。

问题：力觉应该早融合、晚融合、进入 action expert，还是只驱动 fast residual stream？预训练 VLM 表征会不会被新模态破坏？

### S4：统一研究问题

阅读：Reactive Diffusion Policy → ImplicitRDP → Force Policy → FP2 → PhaForce → Multi-Modal Policy Consensus → CR-DAgger。

最终问题：低频语义规划、高频接触反馈、动作 chunk 与人在环纠正怎样组成稳定的闭环系统？

## 推荐节奏与验收

每周按“2 篇精读 + 1 次推导/代码 + 1 次口述复盘”推进。精读笔记只回答六项：问题、接口、方法、损失、证据、局限。每个模块结束时，不靠论文完成一张架构图和一次 10 分钟口述；无法从输入一路讲到实际执行频率，就还没有真正掌握。

## 当前资料缺口（按优先级添加）

1. ACT、Diffusion Policy。
2. RT-1、RT-2、Open X-Embodiment/RT-X。
3. Octo、OpenVLA。
4. π0、FAST、π0.5。
5. Flow Matching 原始论文。

现有论文已足够覆盖力觉 VLA 专题。先补前四组主干论文，比继续增加同类 force-aware VLA 更能完善知识结构。
