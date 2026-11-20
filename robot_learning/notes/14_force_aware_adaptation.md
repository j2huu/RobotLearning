# 第 14 课：力觉融合、foundation adaptation 与 FP2

[上一课](13_action_and_control_interfaces.md) · [下一课：评估](15_vla_evaluation_and_generalization.md) · [论文tensor推导](03c_fp2_tensor_walkthrough.md) · [3d源码精读](03d_fp2_source_architecture_and_training.md)

目标：在已经理解 VLA 数据与动作 head 后，决定 force 应进入哪里、训练谁、输出什么，并说明这种选择解决不了什么。

按 [循序路线](../planning/vla_learning_path.md) 分三次：**14A读§1比较force入口；14B结合§2/4/5与FP2详解追compressor和冻结图；14C结合§2/3/6/7串完整系统、对照和边界**。前置是第7课Q/K/V与第13课物理接口。课后做 [第14课面试卡](vla_interview_workbook.md)，独立说明每个箭头的来源，而非统称为“融合”。

2026-11-09：在14B加入3d §2–5，14C加入3d §6–11。先讲论文职责，再用peg/RLT1024/no-wrist/no-action配置检查具体算子、标签与执行；不把该配置推广成所有FP2。

## 1. 三种不同的 force 使用方式

| 方案 | Force 进入哪里 | Learned output | 必须解决的问题 |
|---|---|---|---|
| force-conditioned action model | observation encoder / action expert | 原生 motion action 或其分布 | 新模态对齐、因果性、训练数据与频率 |
| residual/correction | 下游根据 action/context 与反馈 | motion correction | correction 单位/frame、稳定性、与原目标冲突 |
| structured regulation | context + interaction histories | frame、mask、desired wrench | hybrid controller contract、监督标签与时序 |

这不是严格互斥的模型分类；实际系统可以组合。不能因为观测里有 force，就直接称为显式闭环 force control。

## 2. FP2 的职责分解

$$
A_k=\pi_{RFM}(o_k,\ell),\quad C_k=\operatorname{Context}(o_k,\ell),
\quad z_k=\psi(C_k).
$$

$$
(\Sigma_t,s_t,W_t^{ref})=\pi_F(z_{k(t)},\mathcal W_t,\mathcal P_t).
$$

RFM 不新增 force input；原始 action generation 保留。Compressor 用 learned query cross-attention 压缩 RFM context，本身不读 wrench。Fast force policy 才读取近期 wrench/proprioception。

Controller使用处理后的RFM motion reference、三项结构参数和实际反馈。RGB不直接输入hybrid controller；论文基线和所选no-wrist配置没有快策略独立wrist encoder，上游VLA仍可读取wrist。源码另有wrist/action-head消融。

所选OpenPI路径从上下文化后的prefix中取图像位置`[B,N,2048]`，RLT输出`[B,1,1024]`后flatten。近期pose9+wrench6组成`[B,100,15]`，经FiLM→两层GRU→FiLM，再与context做两路gated fusion，得到`[B,128]`；这条融合不是compressor中的cross-attention。

## 3. 三个参数的含义

- Σ：在哪个局部 interaction frame 中解释 motion/force。
- s∈{0,1}⁶：哪些平移/转动方向控力，其余方向跟踪运动。
- Wref∈R⁶：被选轴的期望 wrench，区别于 measured wrench。

FP2主文未给全raw head配置；当前选定源码已确认三个独立头：frame9、force6、mask6。frame由xyz+rotation6D反变换为7D xyz+wxyz，mask从logits经sigmoid/threshold变bool。21个raw数不等于21D可执行动作；设备层还会清除mask后三个旋转轴。

## 4. 三阶段训练与梯度

| 阶段 | 更新参数 | Force 的位置 |
|---|---|---|
| 原生 task adaptation | RFM 原生可训练部分 | 不新增 force modality |
| context reconstruction | compressor + decoder，RFM冻结 | 不输入 force |
| force-control learning | force policy，RFM/compressor冻结 | histories 与 demo 恢复的交互监督 |

Decoder 训练后丢弃；force loss 不反向重塑 compressor。相比 end-to-end force-conditioned model，这种冻结接口降低更新范围，也限制高层计划直接利用实时 force 重新选择动作的能力。

上表是论文阶段。源码的OpenPI trainer在alpha=0时stop-gradient prefix且仅优化RLT，重建用MSE；alpha>0另加VLA action loss，允许上游经该分支更新。模型类支持mask不表示实际调用用了mask：本trainer给RLT传None。快策略离线读取token，选定基线loss为`L1(frame)+L1(force)+.1 BCE(mask)`，不含action。

未来50步控制标签被重组为单步reference：接触预告、保持与释放分别选标签或置零。`pre_contact/pre_release=40`、`rho=.9`是本配置的标签索引规则；第一项因remove_first通常是锚点后20ms，不能当作在线接触检测阈值。详见3d §9的35/40与36/40手算。

## 5. Adaptation 不等于 LoRA

RFM task adaptation 可以用 full fine-tuning 或 LoRA；FP2 sidecar 是结构位置和职责的描述，不等于低秩更新。先画 forward/backward graph，再说哪些参数被训练。

任务微调在前，冻结训练FP2在后。论文Table I对照是`π0 (LoRA)`与`π0.5 (full)`，不是同一个π0.5的两个微调版本；具体run是否全量更新哪些模块须看上游filter和训练记录，而非YAML名字。

小规模 force 数据还要检查 sensor bias、gravity compensation、frame、contact phase coverage 和 demonstration label 来源。未见 contact state 时的稳定性不能由正常数据 loss 推断。

## 6. 应做的对照

控制同一 backbone、原生动作、task data 与 low-level controller，比较：没有 regulation、仅 context、仅 histories、二者都有、stale context、force delay。记录 task success、force error/peak、contact loss 和 latency。

如果允许 downstream 重新生成动作，参数量、训练目标与延迟都变了；这个 ablation 不能再叫“只加一个 force input”。

从相同task-adapted checkpoint分出baseline/+FP2，才能把任务微调收益先控制住。但同一批demonstrations仍不等于相同输入和监督：FP2多用wrench、结构标签与实时闭环。整体增量不能全归给compressor；RLT/global_mean比较还可能改变feature来源与宽度，需逐项记录。

## 7. 能力边界

若 RFM 指向错误孔、严重偏离表面或选择错误目标，force regulation 不能保证恢复。它更适合在 task-level motion 大体可行时改善接触执行。实际位姿可因控力改变，因此“保留 action generation”不等于逐点执行原参考不变。

实验：[ML05](../labs/notebooks/ML05_attention_compressor.ipynb) 训练合成 compressor/sidecar，不含 controller；[VLA02](../labs/notebooks/VLA02_multirate_force_execution.ipynb) 展示规则式接触闭环，不训练真实 FP2。两者各自支持不同机制，不能拼成真实模型复现声明。

## 验收

给同一任务选一种 force 接口，列数据需求、输出 shape、训练/冻结、时序与失败边界。英文：Explain how downstream interaction regulation differs from regenerating the foundation policy's actions.

源码验收再加四问：context两条FiLM/gate路径如何连接？target stop-gradient为何不足以证明冻结？35/40与36/40接触怎样生成标签？normalized force=1.4为何不自动被限制为50N？答案与证据路径见3d。
