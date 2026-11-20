# FP2：从架构图到张量与控制闭环

本文针对本仓库的 [FP2 原文](../../paper/ICRA27_2601_MS.pdf)，重点对应第 3 页 Fig. 2、式 (1)–(4)、第 4 页 III-C/III-D 和第 7 页 Table II。前作的实现细节单独标为“Force Policy 前作”，来源是 [Force Policy](../../paper/2602.22088v2.pdf) 第 18 页 Supplement III；不能把它们自动当作 FP2 的已确认配置。

**2026-11-09补充：** 现在已有选定FP2源码的 [3d：peg-in-hole架构与训练精读](03d_fp2_source_architecture_and_training.md)。本文保留论文层推导与教学例子；“主文未说明”不再等于“当前源码未知”。所选RLT1024/no-wrist/no-action配置可确认100点history、15→128的FiLM/GRU、两路gating及9/6/6三头；其他配置不能直接套用。

## 1. 先纠正最容易形成的误解

FP2 并没有使用 force history 去引导 context compressor，也没有把 force 输入 RFM。

```text
视觉、语言、原生机器人状态 → RFM → contextual tokens → compressor → z
                               └→ action chunk ──────────────────┐
实时 wrench history ─────────────────────────┐                    │
实时 proprioception history ─────────────────┼→ force policy       │
缓存 z ──────────────────────────────────────┘   ↓                 │
                                        frame、mask、desired wrench
                                                └────┬────────────┘
                                                 hybrid controller
                                                        ↓
                                                    机器人执行
                                                        ↓
                                         新状态、新 wrench，形成反馈
```

这里有两个不同的“将大量信息变成紧凑表示”：

1. **上下文压缩**：把 RFM 的多个上下文 token 汇聚成一个 token；这一步不读取力觉。
2. **交互历史编码/融合**：force policy 根据 wrench、proprioception 历史与缓存上下文预测控制参数。主文未完整展开；所选源码为input FiLM→两层GRU→output FiLM→gated fusion，详见3d §7。这与compressor的cross-attention是不同操作。

图 2 右侧上下叠放的 encoder / decoder 是上下文压缩的训练结构。下方 decoder 用于重建，不是部署时的 force policy。

## 2. 符号与形状：先分清“多少个 token”和“每个 token 多宽”

为避免论文中 sequence length 与 selection mask 都用 S 的冲突，本文用 N 表示上下文 token 数，用 s 表示六轴 mask。省略 batch 维时，论文式 (2) 是 N×D → 1×d。

| 符号 | 含义 | 带 batch 的形状 | 已确认程度 |
|---|---|---|---|
| B | batch size | 标量 | 本文补充的通用记法 |
| N | RFM context token 数 | 标量 | 论文记为 S；具体值未列出 |
| D | RFM hidden width | 标量 | 与 backbone 有关；具体值未列出 |
| d | compressor 输出宽度 | 标量 | 论文未列具体值 |
| C | RFM contextual tokens | B×N×D | FP2 明示 |
| X | 投影后的 contextual tokens | B×N×d | FP2 明示投影，可推导形状 |
| Q₀ | 可学习的压缩 query | 1×d，batch 时广播 | FP2 明示 |
| z | 压缩后的单个 context token | B×1×d | FP2 明示 |
| W_hist | 实测 wrench 历史 | B×L×6 | wrench 的六维定义；L 的具体值未说明 |
| P_hist | proprioception 历史 | B×L×dₚ | dₚ 的具体组成未说明 |
| A | RFM action chunk | B×Kₐ×dₐ | Kₐ、dₐ 由原生 action 接口决定 |
| s | force-position selection mask | B×6 | 六轴混合力位控制接口 |
| W_ref | desired wrench | B×6 | 力三维 + 力矩三维 |
| Σ | interaction frame | 取决于 frame 的数值编码 | FP2 未说明具体 raw head 宽度 |

L 是历史采样点数。论文用 W_{t-H:t} 表示一段历史；若 H 是离散步数并包含两端，点数可能是 H+1，因此本文不直接把 H 当成张量第二维。

上表“未说明”均指论文披露范围。所选源码另有具体合同：`C=[B,N,2048]`，`Z=[B,1,1024]`后flatten，`P=[B,100,9]`，`W=[B,100,6]`；raw frame/force/mask分别为`[B,1,9/6/6]`。它不是所有backbone共同的配置。

**一个 token 不是一个标量。** z=[z₁,…,z_d] 仍有 d 个实数。它们是学习到的 latent coordinates，没有指定某一维必须是任务名、接触状态或目标力。

## 3. 第一条输出：RFM 的动作支路

任务适配后的 RFM 按原生接口读取 oₜ 和指令 ℓ，输出动作序列：

$$
A_{k}=\pi_{\mathrm{RFM}}(o_{t_k},\ell)
\in\mathbb R^{B\times K_a\times d_a}.
$$

oₜ 可以包含图像与机器人状态，具体组织取决于 backbone。不能给四种 backbone 强行指定相同数量的图像 token、相同 state vector 或相同 action width。

Fig. 2 中绿色区域及 noise 表示 RFM 原生动作生成支路的示意。FP2 不重写该支路。蓝色 contextual tokens 进入压缩器；不能把所有绿色 action tokens 一并视为 compressor 输入。具体取哪层 hidden states、是否包含各类 special/state tokens，主文没有逐 backbone 列明。

所选OpenPI源码先计算`prefix_out`，再以`image_only=True`截取图像位置；因此是上下文化后的image-prefix hidden states，不是原始patch，也不是直接把全部图文位置压进RLT。文本仍可能经上游attention影响图像位置。

论文III-D描述原始action chunk经过waypoint dropout、DTW对齐和quintic spline重采样，再交给controller：

```text
原生动作 waypoints，15 Hz
    → 根据最近执行轨迹选起点，丢弃过时 waypoints
    → 五次样条，重采样到 50 Hz
    → motion reference
```

“保留 RFM action generation”指不由 force policy 再生成动作；它不意味着机器人在控力轴上的实际位姿会与原始 action 完全相同。控力闭环可以为满足目标力而改变实际运动。

当前peg源码配置则选择linear位置插值+姿态SLERP，将15Hz source重采样到50Hz planner；不能把上面的论文五次样条当作该checkout已经实现的分支。

## 4. 第二条输出：context compressor 的完整形状变化

### 4.1 输入 C 已经是 contextual representation

$$
C=[c_1;\ldots;c_N],\qquad c_i\in\mathbb R^D.
$$

这些 token 经过 RFM 内部的上下文化计算，不能简单理解为“一维是红色、一维是擦桌子”。每个 token 可以混合视觉、任务与状态信息，取决于原生模型。

### 4.2 逐 token 投影：改变宽度，不减少 token 数

用线性层说明论文的 Proj：

$$
X=CW_p+b_p,\qquad W_p\in\mathbb R^{D\times d}.
$$

$$
\underbrace{B\times N\times D}_{C}
\longrightarrow
\underbrace{B\times N\times d}_{X}.
$$

每个 token 独立用同一个投影层变换；这一步还没有 N→1。

### 4.3 Learned query 对整个 token sequence 做 cross-attention

论文引入 Q₀∈R^{1×d}，是全局可学习参数，并不是实时 wrench 的变换。下面是标准多头 attention 的形状展开，用于解释论文模块，不代表所有 LayerNorm/位置编码实现细节均已披露。

设 h 个 attention heads，dₕ=d/h。单个 head：

$$
q=Q_0W_Q\in\mathbb R^{1\times d_h},\quad
K=XW_K\in\mathbb R^{N\times d_h},\quad
V=XW_V\in\mathbb R^{N\times d_h}.
$$

$$
\alpha=\operatorname{softmax}\left(\frac{qK^\top}{\sqrt{d_h}}\right)
\in\mathbb R^{1\times N},
\qquad
u=\alpha V\in\mathbb R^{1\times d_h}.
$$

因此 u=Σᵢαᵢvᵢ，是对 N 个 value vectors 的加权汇总。多头拼接回 1×d，再经过输出投影、残差和 MLP；论文说明 encoder block 使用 multi-head cross-attention 和 MLP residual connections。

```text
Q₀                         1×d
投影后的 K、V              N×d
各 head 的 attention 权重  1×N
各 head 的加权输出         1×dₕ
拼接 + 输出投影 + 残差/MLP  1×d
```

**N→1 发生在 query 只有一个、它对 N 个 context tokens 读取信息这一步。**

这不是挑一个原始 token 直接保留，也不是固定平均。α 随 C 改变；多头与多层可以提取不同信息。不过训练目标是重建上下文，不能声称 α 已被明确监督成“与力控最相关的 token 权重”。

### 4.4 一个明确标为示例的数值演练

仅为理解，假设 B=2、N=128、D=2048、d=256、h=8；这些不是论文报告的超参数。

```text
C                 [2, 128, 2048]
Proj(C)           [2, 128, 256]
每个 head 的 q    [2, 1, 32]
每个 head 的 K,V  [2, 128, 32]
每个 head 的 α    [2, 1, 128]
每个 head 输出    [2, 1, 32]
八个 head 拼接    [2, 1, 256]
最终 z            [2, 1, 256]
```

这次演练有两个变化：2048→256 是 feature width 变化；128→1 是 sequence length 变化。这仍是教学尺寸。所选源码实际使用2048→1024、一个learned query，详见3d；还支持其他token数/宽度变体。

## 5. 没有 force supervision，压缩器怎样学会保留有用信息

FP2 的第二阶段使用 reconstruction decoder：

$$
z=\psi(C),\qquad
\hat C=f_{dec}(z)\in\mathbb R^{B\times N\times D}.
$$

decoder 的 learned reconstruction queries 只读取 z，重建原始 contextual token sequence。输入 C 作为重建目标存在，但没有 C→decoder 的直接信息旁路，否则无法强制信息通过 bottleneck。

$$
\mathcal L_{comp}
=\mathcal L_{MSE}(C,\hat C)
+\lambda\mathcal L_{cos}(C,\hat C).
$$

理解上，token-wise MSE 约束数值，cosine distance 约束 feature direction。具体归约和实现细节仍以代码为准。这里的监督来自 C 本身，不是人工定义的接触阶段标签，也不是未来 wrench。

**论文与实现的区别：** 上式为论文目标；所读OpenPI/JAX `RLTokenModel.loss`实际只有MSE，trainer传入的mask为`None`。通用PyTorch compressor可配置cosine项，但不能据此断言JAX训练也用了它。冻结还需检查prefix输入的stop-gradient和optimizer filter，不能只看重建target，详见3d §4。

```text
训练：C → compressor → z → reconstruction decoder → Ĉ
      └──────────────────────── reconstruction target ┘

部署：C → compressor → z → cache → force policy
                        decoder 已丢弃
```

这种重建一般是有损的。N·D 个数经过 d 维瓶颈，不保证任意 C 都能精确复原；方法依靠任务数据中上下文的结构与冗余。重建效果好，也不等价于每个接触细节都已保留。下游消融实验提供的是实际有用性的证据。

## 6. Force Info 的向量组成与作用

### 6.1 实测 wrench：每个时刻六维

$$
W_t=[f_x,f_y,f_z,\tau_x,\tau_y,\tau_z]^\top\in\mathbb R^6.
$$

前三项是力，单位 N；后三项是力矩，单位 N·m。这里的 τ 是末端参考点上的力矩，不是 n 个机器人关节的 torque vector。

$$
\mathcal W_t=[W_{t-L+1};\ldots;W_t]\in\mathbb R^{L\times6}.
$$

历史能提供接触力的变化趋势、建立/丢失接触的时序线索，以及与运动同步的响应。历史本身并不能保证唯一确定几何形状或接触状态。

### 6.2 Proprioception：告诉模型刚才怎样运动

$$
\mathcal P_t=[P_{t-L+1};\ldots;P_t]\in\mathbb R^{L\times d_p}.
$$

FP2 主文没有完整列出 P 每维含义。其物理作用是把力变化与自身运动联系起来，例如区分“向前运动受到阻力”和“主动加压导致力上升”。

**Force Policy 前作的具体实现**：第 18 页写明使用末端 pose history，pose 为 9D continuous-rotation representation，通常可拆成 position 3D + rotation 6D；wrench 为 6D。将同一时刻的二者拼接得到 15D，是这套前作表示下的推导。前作使用 GRU 编码历史，最终 proprioceptive embedding 为 128D，并由 global context 通过 FiLM 调制。FP2 没有重列这些具体宽度，不能直接声称 FP2 必然是“15→128”。

**现已核对的FP2配置：** peg/RLT1024基线确实使用100点、relative TCP pose9+wrench6，经input FiLM变128维，再进两层GRU、output FiLM并取末时刻。这一结论来自当前源码，不是把前作参数默认继承到所有FP2。Wrench仍按各时刻TCP约定处理，不可假设所有历史已搬到同一最新原点。

### 6.3 与 z 融合之后才预测力控参数

论文确认的整体映射是：

$$
(\Sigma_t,\;s_t,\;W_{t}^{\mathrm{ref}})
=\pi_{F}\!\left(z_{k(t)},\;\mathcal{W}_t,\;\mathcal{P}_t\right).
$$

k(t) 是最近完成的 slow update。z 是缓存的上下文，W/P 则随 fast step 更新。论文式 (1) 没有把原始 action chunk 列为 π_F 的直接输入；action 在 controller 处与 π_F 的输出汇合。

为了理解，可以把其内部抽象为：

```text
W_hist、P_hist → HistoryEncoder → h_phys
缓存 z、h_phys → ContextConditioning/Fusion → h_fused
h_fused → frame head / mask head / wrench head
```

这张抽象图表达论文层的信息依赖关系。所选源码现可具体化为FiLM/GRU历史支路与context直接支路的两路gated fusion；ML05的GRU+concat是另一个教学实现。

z 让相似的力觉在不同任务或阶段下有不同含义；W/P 让相同的上下文能对不同的实际交互作出不同反应。例如插入中的阻力与擦拭中的法向力，未必应触发相同的调节策略。

## 7. 三个输出究竟是什么

### 7.1 Σ：在哪组局部轴上解释力与运动

几何上一个 frame 可以用旋转 R 和原点 p 描述。它定义局部 x/y/z 以及相应的转动轴。Σ 不等于下一步机器人动作，也不等于图像特征。

FP2 没有说明 raw head 用 quaternion、6D rotation 或其他编码。**Force Policy 前作**使用相对末端的 9D frame-pose 表示：position 3D + continuous rotation 6D。9 个编码数字不代表 9 个物理自由度。

这里“没有说明”指主文；当前所选FP2源码的frame head确认为9D，processor反变换后为TCP-relative xyz+wxyz quaternion，共7D。

### 7.2 s：六个轴分别控位置还是控力

$$
s=[s_x,s_y,s_z,s_{r_x},s_{r_y},s_{r_z}]\in\{0,1\}^6,
\quad M=\operatorname{diag}(s).
$$

sᵢ=1 是 force/torque control，sᵢ=0 是 motion/position control。不能把三维线性轴的选择与三个转动轴的选择混为一谈。

例如沿平面擦拭、局部 z 为法向时，可用：

$$
s=[0,0,1,0,0,0].
$$

这是帮助理解的典型模式，不是所有任务固定使用的 mask。自由空间可全部为 0；插入则可能在横向与某些转动方向上用力/力矩控制。

### 7.3 W_ref：想要达到的 wrench

W_ref∈R⁶ 是目标，不是当前传感器读数，也不是最终关节命令。controller 主要在 mask 选中的轴上使用它。

论文未确认raw width；当前选定源码现已确认三个独立头输出9+6+6=21个数，frame/force用L1、mask用BCEWithLogits，推理sigmoid后以0.5阈值转bool。设备适配层还会清除mask后三个旋转轴，模型输出与实际启用轴必须分开检查。

## 8. controller 如何把两条支路合成真实动作

在同一 interaction frame 和参考点中计算：

$$
e_W=W^{ref}_{\Sigma}-W^{meas}_{\Sigma}.
$$

当只是同一点的坐标轴旋转时：

$$
f_{\Sigma}=R^\top f_{world},\qquad
\tau_{\Sigma}=R^\top\tau_{world}.
$$

如果参考点也改变，还需要力矩平移项，不能只旋转三个力和三个力矩。force、pose、twist 必须遵守统一的坐标系与参考点约定。

概念上，把 motion branch 与 force branch 都转成同一种控制量后：

$$
u_\Sigma=(I-M)u_{motion}+M u_{force}.
$$

u_motion 由 RFM motion-reference tracking 产生；u_force 由 wrench error 等闭环量产生。不能直接把“米单位的位置向量”和“牛顿单位的力向量”相加。论文没有把全部低层增益和执行律展开，上式用于说明互补轴选择，不冒充 FP2 的精确控制律。

举一个非论文实测的例子：局部 z 轴目标接触力为 +8 N，实测为 +5 N，且 s_z=1。此时该轴的 force error 为 +3 N，controller 按其控制律调节接触；局部 x/y 仍跟踪 RFM 提供的擦拭路径。曲面法向转动时，Σ 也要更新，让“控 z 轴力”始终对应新的局部法向。力的正负取决于传感器与控制器的约定。

从学习模块看，measured wrench 是 π_F 的输入；从完整闭环看，显式力控制还需要低层比较 desired/measured wrench。这里必须区分插值后指令流的频率、传感器反馈更新率与低层控制律的执行率，不能仅由 1 kHz 插值输出推断传感器也以 1 kHz 提供新测量。

## 9. 多速率：论文、历史说明与当前源码分别记录

| 环节 | 频率 | 理解与来源 |
|---|---:|---|
| RFM 推理 | 2 Hz | 约每 500 ms 产生一次 chunk 和 context |
| compressor | 随 RFM 更新 | 重新计算 z，随后缓存 |
| chunk 原始 waypoints | 15 Hz | 相邻 waypoint 的时间间隔约 66.7 ms |
| force policy | 50 Hz | 每 20 ms 读取更新的历史并预测结构参数 |
| motion reference | 50 Hz | 对齐与样条重采样的结果 |
| 论文内部低层控制律 | 1 kHz | III-D的底层口径，不是新模型预测或传感器到达率 |

上表为论文系统；理想2Hz/50Hz同步下可数出一次slow期间25个fast steps，但异步延迟会改变实际时序。此前用户补充的“1kHz是插值输出”作为相应实现的历史口径保留，不能自动回填为当前Python线程频率。

2026-11-09所选源码：source15Hz、planner/fast目标50Hz、F/T buffer轮询200Hz，100点history按50Hz取样；`max_infer_dt=.5s`是距上次slow完成后的请求条件，不是实测固定2Hz。当前应用层不证明有50→1000Hz命令插值或1kHz Python反馈，需另核对SDK、实验版本与时间日志。详见3d §10。

15 Hz 是 chunk 内 waypoint 的采样频率，不是 RFM 推理频率。2 Hz 是生成一整段轨迹的频率，也不意味着机器人只以 2 Hz 移动。

## 10. 三阶段训练与梯度去向

| 阶段 | 输入/目标 | 更新参数 | 力信息是否参与 |
|---|---|---|---|
| 1. task adaptation | RFM 原生任务数据与动作目标 | RFM 的可训练部分，可按原生流程 full fine-tune/LoRA | 不作为新增 sensing modality 输入 RFM |
| 2. compression | C→z→Ĉ，以 C 为重建目标 | compressor + reconstruction decoder | 不读取 wrench，不受 force-policy loss 更新 |
| 3. force-control learning | z、W/P histories→demo 恢复的 frame/mask/wrench targets | force-control policy | 在此作为输入与交互监督来源使用 |

阶段 3 冻结 RFM 和 compressor；阶段 2 的 reconstruction decoder 不部署。接触结构标签沿用前作从 demonstration 恢复的办法；不等于 compressor 自动提供了 frame/mask 标签，也不等于在线需要人工标注。

上表描述论文训练流程。当前OpenPI trainer额外支持`alpha>0`的联合action-loss分支：重建输入仍stop-gradient，VLA可通过额外action loss更新。alpha=0只训练RLT；离线token后的force训练不回传RFM/RLT。当前peg标签消费与未来序列重组已读，但原始标注生成全过程尚未审计，不能直接把前作labeler当来源证明。

FP2 主文没有重列完整 π_F 的各项损失和权重，不能把前作含动作生成项的总 loss 整体照搬给 FP2。

选定源码现在可确认`L_frame + L_force + .1 L_mask`，基线没有action loss。50个未来标签经过接触预告/释放规则重组为1步监督；规则和严格边界手算见3d §8–9。

## 11. 什么证据支持这些模块的分工

FP2 第 7 页 Table II 是 π0.5 backbone 上 Flip Box 任务的设计消融：

| force-policy 输入 | 训练物体 SR | 新物体平均 SR |
|---|---:|---:|
| 只有 z | 4% | 5% |
| 只有 W/P histories | 36% | 18% |
| average-pooled context + W/P | 84% | 75% |
| action latent + W/P | 88% | 80% |
| learned compressed context + W/P | 92% | 88% |

这些结果支持上下文与物理反馈互补、该 learned bottleneck 在这项消融中有效。它们不能证明任何任务上都应使用单 token，不能证明 z 具有可解释的逐维语义，也不能证明 learned compression 是无损的。

该论文的 limitation 还指出当前 force regulation 在 task level 学习，不能把“可接多种 RFM”直接解释成“一个通用 force-policy checkpoint 零样本适配所有任务和机器人”。

## 12. 阅读 Fig. 2 的自检

1. compressor 的输入是 C，而不是 W/P。
2. learned query Q₀ 是可训练参数，不是实时 force query。
3. N→1 与 D→d 是两种不同的形状变化。
4. reconstruction decoder 只在压缩训练中使用。
5. z 是 learned latent representation，不是显式六维力。
6. force policy 的直接输入是 z、W/P history，输出是 frame、mask、desired wrench。
7. RFM action 与结构参数在 controller 处汇合。
8. desired wrench、measured wrench、实际低层 command 是三种不同量。
9. 原生 action generation 被保留，实际执行仍会受控力闭环调节。
10. FP2 主要修复交互执行问题，无法保证修复错误目标或严重几何错位。

## 13. 从 RGB 到 regulated action 的端到端 tensor 实例

### 13.1 先明确接口：RGB 不直接输入 hybrid controller

论文图 2 的依赖关系是：

```text
RGB / language / 原生 state → RFM → raw action chunk
                                  → contextual tokens → compressor → cached z
cached z + wrench history + proprioception history → force policy
force policy → interaction frame / selection mask / desired wrench

处理后的 action reference + 三项力控参数 + 当前实际状态 + 实测 wrench
    → hybrid force-position controller → 实际控制命令 / regulated execution
```

RGB 的信息通过 RFM action 与 compressed context 间接影响执行。FP2 去掉了 force policy 的独立 wrist-camera input；controller 本身也不是一个将 RGB、action、state 拼接起来的视觉网络。Context token 主要供 force policy 使用，controller 使用的是已经具有控制含义的结构参数。

Proprioception 有两处用途：原生机器人状态可以进入 RFM；较新的状态历史进入 force policy，而当前状态还用于 controller 的跟踪误差与机器人运动学计算。三处不必使用相同编码或更新率。

### 13.2 一个完整但不冒充论文超参数的形状配置

以下数值全部为教学示例：batch B=1，两路 224×224 RGB，16 个 action waypoints，128 个 context tokens，context width D=2048，compressed width d=256，历史长度 L=10。

为方便展示，假设 native action 是 Cartesian pose + gripper：position 3D、continuous rotation 6D、gripper 1D，合计 dₐ=10；proprioception history 只取同样格式的末端 pose，不含 gripper，dₚ=9。真实 backbone 可能使用 joint actions、不同旋转编码或不同 state composition，需按其接口做坐标转换、反归一化或运动学映射，不能直接套用这些数字。

| 数据 | 示例 shape | 内容 |
|---|---|---|
| RGB | [1, 2, 3, 224, 224] | batch、camera、RGB channel、height、width |
| native state | [1, dₛ] | 由 RFM 原生 observation 接口决定 |
| raw action chunk | [1, 16, 10] | 每个 waypoint 为 position 3 + rotation 6 + gripper 1 |
| contextual tokens C | [1, 128, 2048] | RFM 内部的上下文特征 |
| projected tokens X | [1, 128, 256] | 逐 token 改变宽度 |
| compressed z | [1, 1, 256] | 单 token 上下文，随 slow update 缓存 |
| P_hist | [1, 10, 9] | pose history |
| W_hist | [1, 10, 6] | force 3 + moment 3 的历史 |
| hist = concat(P_hist, W_hist) | [1, 10, 15] | 沿 feature 维拼接，时间长度不变 |

一种形状完整的教学用 force-policy 内部可以写成：

```python
# 此处 history encoder、concat、head 宽度是教学选择，并非 FP2 已公布配置。
hist = cat([P_hist, W_hist], dim=-1)       # [1, 10, 15]
h_phys = history_encoder(hist)            # [1, 128]
z_flat = z.squeeze(1)                     # [1, 256]
fused = cat([z_flat, h_phys], dim=-1)      # [1, 384]
h = fusion_mlp(fused)                     # [1, 128]

frame_code = frame_head(h)                # [1, 9]，示例采用前作的 frame 编码
mask_logits = mask_head(h)                # [1, 6]
mask = sigmoid(mask_logits) > 0.5         # [1, 6]，示例离散规则
w_ref = wrench_head(h)                   # [1, 6]
```

若输出训练时做了归一化，交给物理 controller 前需按对应尺度还原。期望力的单位、参考点、坐标系必须明确，不能把 normalized latent 数值直接当成 N 或 N·m。

### 13.3 frame、mask、wrench 的数值含义

在上述 frame 编码示例中，frame_code 的前 3 个数定义相对位置，后 6 个数定义旋转编码。旋转编码经正交化得到 R∈R^{3×3}，再与位置组成 T∈R^{4×4}。若 frame 相对于末端 E，且机器人基座为 B，则：

$$
T_{B\Sigma}=T_{BE}T_{E\Sigma}.
$$

9D 是表示宽度，不是 9 个物理自由度。FP2 主文没有确认其具体 raw frame encoding；此处借前作的表示解释 tensor 如何成为实际坐标系。

mask 的六个元素依次对应局部三个平移轴与三个转动轴。mask=[0,0,1,0,0,0] 表示只有局部 z 轴控力。w_ref=[0,0,8,0,0,0] 则在这个示例中要求该轴达到 +8 N，其余项在当前 mask 下不作为力控制目标。

这三项是每步预测的控制设置，不是模型权重，也不是论文所称的 stiffness/damping gain prediction。它们分别回答：在哪组轴上控制、各轴采用哪种模式、目标 wrench 是多少。

### 13.4 raw chunk 先成为当前时刻的 motion reference

raw chunk 的 [1,16,10] 不需要与 RGB 或 frame_code 一起做神经网络式拼接。轨迹管理模块先做对齐、丢弃过时点、插值/重采样，得到当前执行时刻的参考：

```text
position_ref       [1, 3]
rotation_ref       [1, 3, 3]
gripper_ref        [1, 1]
必要时还有参考 twist / acceleration
```

controller 比较这些参考与实际状态。它通常需要当前的 state/wrench，而 force policy 使用一段历史：这是“执行闭环”与“预测交互结构”的不同需要。

### 13.5 用速度形式演示混合控制：不是 FP2 精确低层控制律

为了把 tensor 如何合成讲清楚，这里选择一种示意性的速度/导纳实现。FP2 主文未完整披露实际低层增益与输出 API，下式不当作实际机器人源码。

在统一的 interaction frame 中，构造 pose error eₓ=[位置误差 3，旋转误差 3]∈R⁶。旋转误差是局部三维旋转误差，不能直接将两个 6D rotation codes 相减后当作三个旋转自由度。

两条分支先都变成 twist：

$$
u_{\mathrm{motion}}=K_p e_x,
\qquad
u_{\mathrm{force}}=K_f(W^{\mathrm{ref}}-W^{\mathrm{meas}}).
$$

前三维是 m/s，后三维是 rad/s。Kₚ、K_f 是这个示例 controller 的增益，其单位按平移/转动分量分别定义；它们不是这里的 FP2 force policy 输出。

按轴选择：

$$
u_{\mathrm{reg}}=(1-s)\odot u_{\mathrm{motion}}+s\odot u_{\mathrm{force}}.
$$

具体示例约定正 z 运动使被控接触力读数增大：

```text
pose_error       [1,6] = [[0.002, 0.001, 0.005, 0, 0, 0]]
u_motion         [1,6] = [[0.020, 0.010, 0.050, 0, 0, 0]]
w_ref            [1,6] = [[0, 0, 8, 0, 0, 0]]
w_meas           [1,6] = [[0, 0, 5, 0, 0, 0]]
wrench_error     [1,6] = [[0, 0, 3, 0, 0, 0]]
u_force          [1,6] = [[0, 0, 0.003, 0, 0, 0]]
mask             [1,6] = [[0, 0, 1, 0, 0, 0]]
u_reg            [1,6] = [[0.020, 0.010, 0.003, 0, 0, 0]]
```

这里取平移 position gain 为 10 s⁻¹，法向导纳增益为 0.001 (m/s)/N。x/y 保留 motion branch；z 舍弃 motion branch 的 0.050 m/s，使用 force branch 的 0.003 m/s。不是把 3 N 直接加到原始 action 上。

在这个常速简例中，20 ms 对应平移增量 [0.0004,0.0002,0.00006] m，即 [0.4,0.2,0.06] mm。是否真的达到 8 N，还由后续环境响应和反馈闭环决定。

### 13.6 regulated actions 的 shape 取决于机器人控制接口

本教学实现先得到 u_reg=[1,6]。再将它变换回机器人控制坐标系，可积分为下一时刻的 Cartesian pose reference，或经机器人运动学映射为 joint command。gripper 可按其独立接口继续执行原生参考；不能在未确认实现时断言 FP2 对 gripper 的具体处理。

因此 regulated actions 不一定是另一个 [B,Kₐ,dₐ] 的 learned action chunk。它可以指 controller 形成的指令流及其受力调节的执行过程。实际 API 也可能直接接收 pose、frame、mask、wrench 这一组结构化目标，由机器人内部完成控制。

50 Hz 参考/指令进一步插值到 1 kHz 输出流，是前文区分过的输出采样率问题。不能据此声称 π_F 在 1 kHz 推理，也不能不加区分地对 frame rotation、binary mask 和 wrench 全部做逐元素线性插值；各字段的插值/保持方式需实现定义。

相关基础见 [FP2 adaptor 与 twist–wrench](03b_fp2_adaptor_twist_wrench.md)。
