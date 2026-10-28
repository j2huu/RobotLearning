# P4：π0.5——异构数据、分层推理与开放环境泛化

前课：[P3](03_fast_action_representation.md)。主读：[本地 π0.5 v1](../../../paper/2504.16054v1.pdf) 全文，方法 §IV，证据 §V，局限 §VI，附录 B–E。以下数字和机制均指该版本；公开模型的区别见 [P5](05_openpi_engineering.md)。

## 1. 新问题不是只让动作更精细

π0 展示通用策略和 dexterous manipulation；π0.5 进一步问：怎样在没有采集过训练数据的新家庭中，完成多阶段操作？

这需要同时迁移：对象语义、场景理解、子任务选择、动作技能、机器人接口和物理交互。某一层泛化不自动推出其余层泛化。

论文采用异构任务 co-training，并用同一模型做高层语义预测与低层连续动作推理。它不是仅把 action head 放大，也不是简单追加更多同分布示范。

## 2. 六类数据：有什么输入、什么监督

依据 §IV-C/D 和 Fig. 4：

| 缩写 | 来源与监督 | 对应能力 | 在 v1 配方中的位置 |
|---|---|---|---|
| MM | 多家庭移动操作示范；动作标签 | 目标机器人在目标类型任务中的技能 | pre/post |
| ME | 多环境静态单/双臂机器人示范 | 较容易采集的场景/对象多样性与动作迁移 | pre/post |
| CE | 实验室跨机器人操作，含 OXE | 多任务和跨 embodiment 的行为先验 | pre；post 移除这一类 |
| HL | 机器人观测与总任务 → bounding boxes / subtask labels | 场景到子任务的选择 | pre/post，post 使用相关切片 |
| WD | 网络图文、VQA、localization | 视觉语言语义与对象知识 | pre/post |
| VI | 人实时用语言指挥已学低层策略 | 在实际执行状态中选择高层命令 | post 加入 |

HL 通常来自对已有机器人轨迹的语义标注；VI 来自语言层的实时示范。两者不应合称为“更多动作 teleoperation”。WD 也没有为每张 web image 提供机器人关节动作标签。

§I/§IV-C 报告约 400 小时 MM，约 100 个家庭环境；场景数量实验最大为 104 个 locations。它们是不同精度的表述。第一阶段 97.6% 的训练 examples 来自 MM 以外的来源，不是说 97.6% 都是 web data，也不是机器人数据时长占比。

## 3. 分层概率分解

总任务为 $\ell$，预测 subtask 为 $\hat\ell$：

$$
p_\theta(A,\hat\ell\mid o,\ell)
=p_\theta(\hat\ell\mid o,\ell)\,
p_\theta(A\mid o,\hat\ell).
$$

这是 §IV-A 的设计：低层以 subtask 为文本条件，不直接依赖原始总任务。因此它是带有条件独立假设的建模选择，不是任意联合概率都必须满足的形式。

例如：

```text
总任务: 把餐具收进水槽
   ↓ 当前图像与状态
subtask: 拿起杯子
   ↓ 当前观测 + “拿起杯子”
50 个连续动作
   ↓ 执行部分、重新观察
subtask: 把杯子放进水槽
```

同一模型权重承担两种推理职责，不代表每个 servo tick 都先生成一段文本。§II 明确高层推理低于低层动作推理频率；v1 没有给出适用于所有场景的固定高层 Hz，不能自行补一个。

高层与低层共享权重也不意味着任务永不出错。若 subtask 错了，低层可能很精确地执行错误目标；若夹爪抓取失败，高层需要根据新观测调整。

## 4. 两阶段训练：这才是 v1 的完整故事

依据 Fig. 3、§IV-B/D：

```text
预训练 VLM
  ↓ 280k updates
MM + ME + CE + HL + WD
离散 text / localization / FAST action tokens
next-token CE，尚无 flow expert 目标
  ↓ 加入随机初始化 action expert
  ↓ 80k updates
MM + ME 的筛选动作示范 + 相关 HL + WD + VI
CE + flow matching，alpha = 10
  ↓ 最终推理
AR subtask text → 条件化 flow action sampling
```

post-training action episodes 筛选成功且长度低于阈值的示范；这与预训练覆盖较广行为的目的不同。不能照搬成“所有失败数据都应删掉”：恢复/纠错数据的用途还取决于标签和训练目标。

用本专题的自洽 flow 符号写示意目标：

$$
\mathcal L
=\lambda_{txt}\mathcal L_{CE}^{text/subtask}
+\lambda_{act}\mathcal L_{CE}^{FAST}
+\alpha\mathcal L_{FM}.
$$

这里把论文 Eq. (1) 的 CE 项拆开只是为了教学；它不新增一组论文超参数。实际每种样本具有什么监督、mask 和权重，应以配方为准。没有动作标签的 web sample 不能被强行赋予机器人 flow target。

训练混合还需决定按 example、token、dataset 还是 loss component 归一化。仅写“CE + 10 FM”不足以重建真实训练规模与有效梯度权重。

## 5. 与 π0 的结构差异

依据附录 E：

| 机制 | π0 v1 | π0.5 v1 |
|---|---|---|
| 本体状态 | 连续投影为 action expert 的 state token | 离散化，以文本 tokens 进入 VLM prefix |
| flow time | 与每个 noisy action embedding 拼接后经 MLP | 单独 time MLP，经 adaptive RMSNorm 条件化 expert 各层 |
| 输出任务 | 主要低层连续动作 | 同一模型支持文本/高层预测与连续动作 |
| action learning | flow matching | 离散预训练，再加入联合连续目标 |

adaptive RMSNorm 可概念性理解为“根据生成时间调制归一化后的 hidden representation”；不要把它当成学习物理机器人时间，也不必假设它与所有 AdaLN 实现完全相同。

论文用 $a_{t:t+H}$ 的闭区间写法，附录明确 50 个动作用 $H=49$；π0 写 $a_t$ 到 $a_{t+H-1}$，$H=50$。本专题一律称 $N_a=50$，它们不相差一个实际动作。

## 6. FAST 与 flow 怎样防止互相泄漏答案

依据附录 E、Fig. 18。$P$ 是 images/prompt/tokenized state，$D$ 是 FAST action tokens，$C$ 是 continuous noisy-action embeddings：

| Query \ Key | P | D | C |
|---|---|---|---|
| P | 双向 | 禁止 | 禁止 |
| D | 允许 | 因果 | 禁止 |
| C | 允许 | 禁止 | 双向 |

教师给出的 FAST actions 是 clean trajectory 的编码。若 flow branch 可以读它们，模型能绕开视觉语言条件而重建答案；部署时只运行 flow 时，这条捷径不存在。

还要检查**间接路径**：若 C 不能直接读 D，但 P 能读 D，那么多层网络中 D→P→C 仍可泄漏。因此 prefix 也不能依赖答案。机制实验会构造这个反例。

**Attention mask 不等于 stop-gradient。** C 可以从 P 读取信息，而 C 的 loss 在反向传播时仍可能更新产生 P 的参数。禁止 P 的 forward attention 读 C，不自动阻断这个梯度。

后来的 [Knowledge Insulation](https://www.pi.website/research/knowledge_insulation) 明确讨论阻止 action-expert 梯度破坏 VLM 表示，同时由离散动作目标继续训练 backbone。这是后续配方说明，不能从 v1 的 Fig. 18 自动推断同样的 gradient routing 已被实现。

## 7. 动作和控制接口也在配方里

§IV-C 使用 dataset-specific 的 1% / 99% quantile 将动作缩放，并用 prompt 中的 control mode 区分 joint 与 end-effector 目标；不同维度做 padding。

§IV-E 的移动平台包含双臂、夹爪、base 和 torso，物理 state/action 为 18 或 19 维。高层读取四相机，低层使用 wrist 与 front 相机。它输出 arm/gripper/torso 的目标，以及 base velocity，由 PD controller 跟踪，command rate 为 50 Hz。

论文说未添加额外 trajectory planning/collision detection，是对其实验系统的描述，不意味着一般部署应省去这些模块。这里最应该学的是 action contract：pose 与 velocity 在同一向量里，也必须保留不同单位和执行语义。

## 8. 六张证据卡：结论需要对应实验

| 位置 | 实验与观察 | 能支持什么 | 不能扩大成什么 |
|---|---|---|---|
| Fig. 7，§V-A | 三个未见真实家庭，任务进度评估 | 在这些新环境中的迁移能力 | 任意家庭、物理条件、任务都成功 |
| Fig. 8/9，§V-B | 3→104 locations；post-train 控制样本暴露量 | 场景多样性在该协议下有收益 | 与数据总量、对象变化完全无关的普遍 scaling law |
| Fig. 10，§V-C | 移除 ME/CE 退化；no WD 的 aggregate 差异不显著 | 跨机器人数据的重要性 | WD 对所有任务都显著提升或完全无效 |
| Fig. 11，§V-C | 移除 WD，OOD 对象语言/执行表现受损 | web semantics 对部分泛化有价值 | web data 已教会所有接触动力学 |
| Fig. 12，§V-D | π0.5 优于 π0、FAST+Flow 对照 | 整套训练/数据组合更有效 | 单独更换一项结构就解释全部增益 |
| Fig. 13/17，§V-E | implicit HL 也受益；移除 VI/WD 变差 | 训练中的高层监督与部署分层各有作用 | 必须显式语言推理才能得到全部收益 |

§V-E 中 full model 的均值甚至高于 human-HL 对照，也不能据此宣称“规划普遍优于人类”。那是有限任务、采样和接口下的特定比较。

## 9. 评估指标与样本单位

附录 B 按各任务 rubric 给分，多数图是 task progress，通常每 policy/任务约 10 次评估；标准四任务比较约 40 次。它还说明部分因硬件/时间等原因取消的 episodes 被移除，统计需结合实际样本数理解。

例如 Dishes in Sink：4 件物体，每件“拿起 + 放入水槽”各 1 分，共 8 分。拿起两件、放入一件是 $3/8=37.5\%$ progress；它不等于 37.5% 的完整 episode 成功率。

附录 C 将 language following（选择正确物体）与 task success（放到正确地点）分开。把目标放得比 distractors 更远，使“总拿最近物体”的捷径更难成功。

10–15 分钟长任务展示，与若干定量评估中约 2–5 分钟的子任务，也不是同一统计口径。面试中要说清视频能力展示和受控实验各支持什么。

## 10. 局限和后续研究问题

§VI 提到陌生把手、难打开的柜子、遮挡、反复开关抽屉、prompt 复杂度和有限记忆等问题。它展示的是有范围的开放环境迁移，没有消除 partial observability、长期记忆和接触控制难题。

联系本仓库：π0.5 能选对“擦桌子”并生成运动，不意味着持续维持合适法向力。[FP2](../03b_fp2_adaptor_twist_wrench.md) 可以补 interaction regulation；高层目标错或轨迹严重不可行仍需要重规划/恢复。

## 11. 练习与验收

1. 画六类数据到 CE/FM 的监督路径，指出哪些数据没有动作标签。
2. 为什么验证 WD 的作用时，仅看四任务 aggregate progress 不够？需看 OOD 对象、高层行为和分任务结果。
3. 设计 `full / no WD / implicit HL / no HL` 四组对照，分别控制什么、检验什么？
4. 用两层 attention 说明为什么只屏蔽 C→D 仍不够；再运行机制实验核对。
5. 设计一个见过任务、没见过场景的 split，以及一个没见过对象类别的 split；解释两者结论的区别。

英文口述：

> Explain the two-stage training recipe of pi-zero-point-five. Which data sources supervise semantics, subtasks, and motor actions, and what do the ablations actually establish?

通过标准：能用六张证据卡陈述结论，并把 v1 方法与后续 KI / 公开 flow 接口分开。下一课：[P5：工程数据流](05_openpi_engineering.md)。
