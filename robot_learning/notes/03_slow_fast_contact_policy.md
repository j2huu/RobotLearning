# 第 3 课：Slow-Fast 接触策略

## RDP、ImplicitRDP、Force Policy 与 VLA 的连接

## 本课目标

完成本课后，应当能够：

1. 解释为什么把 force 直接拼进普通 Diffusion Policy，并不能得到 chunk 内闭环控制。
2. 从输入、输出、训练方式、推理频率和低层控制接口比较 RDP、ImplicitRDP 与 Force Policy。
3. 解释 latent bottleneck、causal masking、modality collapse、interaction frame 和 hybrid force-position control。
4. 判断一个系统是在“根据力修正位置”，还是在“显式调节接触力”。
5. 说明这些 slow-fast 结构怎样与 π0/π0.5 一类 VLA 组合。

本课论文：

- [RDP：Reactive Diffusion Policy](../../paper/2503.02881v3.pdf)
- [ImplicitRDP](../../paper/2512.10946v1.pdf)
- [Force Policy](../../paper/2602.22088v2.pdf)

---

## 1. 三篇论文共同解决的问题

标准 action-chunk policy 学习：

$$
p_\theta(A_t\mid O_t),
\qquad
A_t=(a_t,\ldots,a_{t+H-1}).
$$

如果模型生成整个 chunk 后连续执行，在执行期间新到达的 force/tactile observation 没有进入 policy：

```text
vision/state/force at t
        ↓
generate action chunk [a_t, ..., a_{t+H-1}]
        ↓
execute ------------------------------→
        ↑ new force exists, but policy does not use it
```

即使训练输入里包含 force，这个系统在 chunk 内仍可能是 open loop。区别不在“有没有 force tensor”，而在：

$$
\boxed{\text{new force at }t+i\text{ can change action }a_{t+i}\text{ before execution?}}
$$

视觉适合全局定位、对象语义和长时程进度；force/tactile 适合接触、滑移、卡滞和力峰值。两者的信息尺度和更新频率不同：

| 模态 | 强项 | 常见频率 | 主要问题 |
|---|---|---:|---|
| Global vision | 几何、对象、任务阶段 | 1-10 Hz policy inference | 模型大、时延高、接触常被遮挡 |
| Wrist vision | 局部几何、接触邻域 | 中频 | 仍难直接观察力与微小卡滞 |
| F/T or tactile | 接触、阻力、滑移、冲击 | 20 Hz 到 kHz | 非接触阶段噪声大，缺少全局语义 |
| Low-level state | 位姿、速度、关节状态 | 高 | 单独无法判断任务意图 |

三篇论文给出三种答案：

```text
RDP:         显式 slow policy + fast decoder
ImplicitRDP: 一个带因果结构的 end-to-end diffusion policy
Force Policy: global vision policy + local force policy + explicit hybrid controller
```

---

## 2. RDP：在 latent action 与真实动作之间插入快速触觉反馈

### 2.1 核心结构

RDP 有两个模块：

```text
Slow policy, 1-2 Hz
low-frequency observation O_t
        ↓
latent diffusion policy
        ↓
latent action chunk Z

Fast policy, about 24 Hz in the paper
Z + latest tactile/force F_t
        ↓
asymmetric tokenizer decoder
        ↓
current executable action a_t
```

Slow policy 负责复杂、多峰、非马尔可夫的轨迹结构；fast policy 在每个执行步读取最新 force/tactile，修正真实动作。

### 2.2 为什么叫 asymmetric tokenizer

令真实动作 chunk 为：

$$
A\in\mathbb R^{T\times D}.
$$

Encoder 只读取动作：

$$
Z=E(A),qquad Z\in\mathbb R^{t\times d}.
$$

Decoder 同时读取 latent action 和当前触觉/力序列：

$$
\hat A=D\big([Z,F]\big).
$$

不对称性在于：

```text
Encoder: A → Z
Decoder: Z + F → A_hat
```

force 不进入 encoder，因此 $Z$ 被迫保留较粗的高层动作意图；接触相关的精细调整交给 decoder 根据最新 $F$ 产生。

Tokenizer 训练目标为：

$$
\mathcal L_{AT}
=
\mathbb E\|A-\hat A\|_1
+\lambda_{KL}\mathcal L_{KL}.
$$

这里的 KL 系数较小，目标主要是平滑 latent space，而不是把 tokenizer 训练成完整生成模型。

### 2.3 Slow latent diffusion

冻结或完成 tokenizer 训练后，将专家动作编码为：

$$
Z^0=E(A^0).
$$

Slow policy 在 latent space 做 diffusion：

$$
\mathcal L_{LDP}
=
\mathbb E
\left\|
\epsilon^k-epsilon_\theta(O,Z^k,k)
\right\|_2^2.
$$

推理时 slow policy 低频生成一个 latent chunk；fast GRU decoder 高频、自回归地结合新 force/tactile，逐步产生真实动作。

### 2.4 RDP 的收益与代价

收益：

- diffusion 只在压缩后的 latent action 上运行，计算更省。
- fast decoder 小于 1 ms，可高频利用 force/tactile。
- 明确分工，使普通 action chunk 获得 chunk 内反馈。

代价：

- 两阶段训练与模块调参更复杂。
- slow policy 只输出 latent intent，fast decoder 可能丢失精确空间几何。
- 若 slow latent 与真实接触状态不匹配，fast decoder 只能在错误意图附近修正。
- OOD force 可能使 decoder 输出异常动作，因此论文强调 relative trajectory 和 latency matching。

重要判断：RDP 的 fast policy 被论文类比为 learnable impedance controller，但它实际输出的是 reactive motion action；它没有显式输出 stiffness matrix 或通过动力学计算 torque command。

---

## 3. ImplicitRDP：用因果结构把 slow-fast 合入一个网络

### 3.1 它对 RDP 的批评

RDP 的显式 hierarchy 可能造成：

1. latent compression 丢失精确空间信息；
2. slow/fast 模块误差累积；
3. 两阶段训练和 hand-designed handoff；
4. fast decoder 只能围绕 slow latent 进行局部修正。

ImplicitRDP 保持原始 action space，在一个 Transformer diffusion policy 内共同处理：

- slow observations：vision + proprioception；
- fast observations：与 action sequence 时间对齐的 force history；
- noisy action sequence。

### 3.2 Temporally causal structure

训练时，对于第 $s$ 个 action token，只允许它读取不晚于该动作时刻的 force：

$$
a_s
\text{ attends to }
\{f_1,\ldots,f_s\},
\qquad
a_s\not\text{ attend to }\{f_{s+1},\ldots\}.
$$

否则训练时会发生 future-force leakage：模型用执行该动作之后才可能测到的 force 预测当前动作，部署时无法复现。

实现上：

- GRU 编码 force sequence，保持时间因果性；
- causal attention mask 限制 action-force attention；
- 训练时仍可并行处理整段序列。

### 3.3 Consistent inference

每个 fast control step 都加入最新 force，再生成截至当前时刻的动作序列。若每次独立 diffusion sampling，随机性会让相邻输出不一致。因此论文：

- 在 chunk 开始时只采样一次 initial noise；
- cache slow visual tokens 和 initial noise；
- 使用 deterministic DDIM，设置 $\eta=0$；
- 每个 fast step 只延长 force/action prefix；
- 执行重新生成序列的最后一个动作。

简化流程：

```text
chunk start:
  encode slow observation once
  sample noise once

fast step s:
  append latest force f_s
  deterministic denoise with same cached context/noise
  execute last predicted action a_s
```

所以 ImplicitRDP 的“implicit slow-fast”不是两个独立 policy，而是由输入频率、因果 attention 和 inference schedule 形成的结构性分工。

### 3.4 为什么需要 Virtual-target Representation Regularization

End-to-end 模型可能出现 modality collapse：视觉足以降低大部分训练 loss，于是模型忽略低维、嘈杂但关键的 force。

论文从准静态 compliance 关系出发：

$$
f_{ext}=K(x_{vt}-x_{real}),
$$

得到 virtual target：

$$
x_{vt}=x_{real}+K^{-1}f_{ext}.
$$

训练时把它作为辅助预测量，与 action 拼接：

$$
a_{aug,t}=\operatorname{concat}(a_t,x_{vt},k_{adp}).
$$

推理时丢弃 $x_{vt}$ 和 $k_{adp}$，只执行 action。

相较直接预测 force，virtual target 的两个动机是：

1. 它和 Cartesian action 位于同一坐标/表示空间，更容易与动作学习对齐。
2. adaptive stiffness 使非接触阶段 $x_{vt}\approx x_{real}$，接触阶段偏差增大，相当于提高关键接触样本的训练权重。

需要保持批判性：这仍是基于 compliance 假设和 heuristic stiffness schedule 的 auxiliary target，不等于模型恢复了真实环境 stiffness。

---

## 4. Force Policy：显式学习接触结构并执行混合力位控制

RDP 和 ImplicitRDP 主要根据 force 生成或修正 motion action。Force Policy 更进一步：它显式决定哪些方向控制力、哪些方向控制位置。

### 4.1 Interaction frame

在接触点附近，环境对不同方向的约束不同。例如沿桌面切向可以移动，法向受到强约束。定义局部 interaction frame：

$$
\Sigma=(R_{\Sigma},p_{\Sigma}),\qquad R_{\Sigma}\in SO(3),
$$

这里的 IF 是三维空间坐标架；它诱导六维 twist/wrench 的坐标表示，不能把任意六维特征向量基都视作可实现的 TCP 坐标旋转。FP1 v2 附录 Supp. I 在局部保守接触、接触分布对称及合适参考点等条件下，讨论同一三维旋转对平移/转动刚度的解耦。

在该局部模型与任务兼容假设下，将六维空间分为：

- constraint subspace：高 stiffness，需要 force regulation；
- admissible-motion subspace：低 stiffness，允许 position/motion control。

论文利用 demonstration 中的 twist：

$$
\xi=[v^\top,\omega^\top]^\top
$$

和 wrench：

$$
W=[f^\top,m^\top]^\top
$$

恢复 interaction frame，并把 interaction 分成 free、surface、insertion、rotation 等模式。

### 4.2 Local policy 的输出

Local force policy 输出：

$$
a_t^{local}
=
(\Sigma_{t+1},S_{t+1},W_{t+1}^{ref},\Delta s_{t+1:t+T_a}).
$$

- $\Sigma$：interaction frame。
- $S\in\{0,1\}^6$：selection mask。
- $S_i=1$：该轴使用 force control。
- $S_i=0$：该轴使用 position control。
- $W^{ref}$：force-controlled axes 的 reference wrench。
- $\Delta s$：position-controlled axes 的 desired motion chunk。

例如沿表面滑动：

```text
tangential x/y: position control
surface normal z: force control
```

这比“policy 看到 force 后预测一个位置修正”更明确，因为低层 hybrid controller 真正闭环调节 reference wrench。

### 4.3 Global-local 分工

```text
Global vision policy, about 5 Hz
  global RGB-D / point cloud
  → task progress, reaching, global action, visual feature

Local force policy, 50 Hz
  wrist image + proprioception + wrench history + global feature
  → interaction frame + mask + reference wrench + motion chunk

Hybrid force-position controller
  → force control on selected axes
  → position control on remaining axes
```

论文实验使用 RISE-2 作为 global vision policy，但该接口允许换成 VLA。Force Policy 本身不是 VLA；“可以插入 VLA backbone”与“实验已经验证 VLA 端到端系统”是不同主张。

### 4.4 Router 与异步 scheduler

Selection mask 同时充当 implicit routing signal：

- 无预期接触、mask 全零：global policy 保持控制权；
- local policy 激活 force axis：切换到 hybrid force-position control。

两套 policy 异步运行。为解决 inference latency 导致的新旧 chunk 跳变，scheduler：

1. 将 global/local 输出统一重采样到 50 Hz；
2. 用 hysteresis 抑制频繁切换；
3. 用 DTW 将新 chunk 与最近执行轨迹对齐；
4. 丢弃新 chunk 中已经过时的 waypoint；
5. 生成 acceleration-continuous trajectory。

这部分是面试常被忽略的 deployment contribution：模型输出正确，不代表异步切换后机器人就会平滑。

---

## 5. 三篇论文横向比较

| 维度 | RDP | ImplicitRDP | Force Policy |
|---|---|---|---|
| 核心思想 | slow latent diffusion + fast tactile decoder | causal end-to-end diffusion | global vision + local force + hybrid controller |
| 是否 VLA | 否 | 否 | 否；global policy 可替换为 VLA |
| slow 输入 | vision、state、低频 tactile/force | vision、proprioception | global visual observation |
| fast 输入 | latest tactile/force | growing causal force prefix | wrist image、motion、wrench、global feature |
| policy 输出 | reactive motion action | reactive motion action | frame、mask、reference wrench、motion chunk |
| force 的角色 | fast decoder conditioning | causal token conditioning + auxiliary target | 显式 force-control objective |
| chunk 内闭环 | fast decoder 每步更新 | 每步加入 force 后一致性重采样 | local policy + hybrid controller 高频闭环 |
| 训练 | 两阶段 | end-to-end | global/local 模块化训练 + IF labels |
| 主要风险 | latent bottleneck、模块耦合 | modality collapse、重复 denoising 成本 | IF/模式恢复、router 与控制接口复杂 |
| 控制可解释性 | 中 | 较低 | 高 |

可以把演化理解成：

```text
RDP
  explicit learned hierarchy
  "coarse latent intent + force-conditioned motion decoding"
        ↓ remove latent bottleneck / unify training
ImplicitRDP
  structural hierarchy inside one causal diffusion model
        ↓ make contact objectives explicit
Force Policy
  learned interaction structure + explicit hybrid force-position control
```

这不是严格的单线继承关系，而是三种不同设计取舍。

---

## 6. 与 π0 / VLA 的关系

### 6.1 它们到底属于什么模型类别？

VLA 通常要求部署时的动作策略显式接受 language instruction：

$$
\pi_{VLA}(A_t\mid I_t,\ell,s_t).
$$

RDP、ImplicitRDP 和本文实验中的 Force Policy 都没有这样的运行时文本条件，因此更准确的分类是：

| 方法 | 更准确的类别 | 简化策略形式 |
|---|---|---|
| RDP | visual-tactile/force visuomotor diffusion policy | $p(Z\mid I,s,F_{slow})$，再由 $D(Z,F_{fast})$ 输出动作 |
| ImplicitRDP | end-to-end visual-force diffusion policy | $p(A\mid I,s,F_{\le t})$ |
| Force Policy | global-local vision-force sensorimotor/control policy | $\pi_{global}(A\mid I)$ + $\pi_{local}(\Sigma,S,W^{ref},\Delta s\mid I_{wrist},s,W)$ |

它们可以宽泛地被描述为 vision-action 或 multimodal sensorimotor policies，但“VA”并不是一个像 VLA 那样定义稳定、普遍使用的类别名称。面试中优先使用 `visuomotor policy`、`visual-force policy` 或 `visual-tactile policy`。

如果每个任务使用独立 checkpoint 或任务在部署前已固定，模型可以在没有语言输入的情况下完成技能；但同一场景下，用户通常无法仅靠文本让它在多个目标之间切换。Force Policy 在离线 interaction-frame 标注或恢复中使用 task description/外部语义模型，也不等于部署时的 action policy 是 language-conditioned VLA。

整个机器人系统仍然可以是 VLA 系统，即使 fast controller 不直接读取文本。例如：

$$
c_t,A_t^{coarse}=\pi_{VLA}(I_t,\ell,s_t),
$$

$$
a_t^{fast}=\pi_{local}(c_t,F_{\le t},s_t).
$$

只要语言通过 global context $c_t$ 或 coarse action 对下游动作产生因果影响，fast module 无需在每个 20 ms control step 重复编码原始 text prompt。

π0 类 VLA 解决：

```text
image + language + state
        ↓
semantic and task-conditioned action chunk
```

本课三篇解决：

```text
slow semantic/geometric intent
        +
fast contact observation
        ↓
reactive execution or explicit force control
```

可能的组合有三种：

### 组合 A：VLA + RDP-style fast decoder

VLA/action expert 输出 latent or coarse chunk，fast decoder 用 force 修正。优点是轻量；风险是 fast decoder 被 slow latent 限制。

### 组合 B：VLA action expert 内的 causal force stream

类似 ImplicitRDP，将 force tokens 注入 action expert，并通过 causal mask 支持 chunk 内更新。优点是端到端；风险是计算成本和模态塌缩。

### 组合 C：VLA global policy + Force Policy local controller

VLA 负责语言、对象与全局进度；local policy 输出 interaction frame、force objective 和 motion objective。优点是控制接口清晰；风险是模块切换、标注和系统复杂度。

面试时不要说“给 π0 加一个 force token 就实现了高频力控”。必须回答：

- force token 以什么频率更新？
- 新 force 能否在 chunk 内改变当前动作？
- 是否要重新运行完整 VLM？
- low-level controller 接收 position、wrench、stiffness 还是 torque？
- 新旧 chunk 和不同 policy 怎样对齐？

---

## 7. FP2：把 slow-fast interaction regulation 接到 foundation model

2026-11-09同步：本节的单token、无快wrist/action头和2/50/1000Hz描述针对论文基线。实际所选peg配置的context读取位置、RLT1024、FiLM/GRU/gating、alpha训练分支和部署时钟见 [3d源码精读](03d_fp2_source_architecture_and_training.md)，不再用论文未披露参数代替当前实现。

FP2 将本课的 slow-fast 思想扩展到 robotic foundation model（RFM）。当 foundation policy 是 π0、π0.5 或 GR00T 一类语言条件模型时，整个系统是 **VLA/RFM + downstream force control**；FP2 模块本身仍是 force-control policy。

### 7.1 不是把 force 塞进 foundation model

Task-adapted foundation policy 保持原生接口：

$$
A_{t:t+H_a}=\pi_{RFM}(o_t,\ell),
$$

并产生内部 contextual tokens：

$$
C_t\in\mathbb R^{S\times D}.
$$

论文基线的RFM不读取wrench，action generation不由下游模块重做。Compressor将选定context压缩为一个token；源码支持不同token数和动作头消融。单token写法为：

$$
z_t=\psi(C_t)\in\mathbb R^{1\times d}.
$$

50 Hz force-control policy 读取缓存的 task context、wrench history 和 proprioceptive history：

$$
(\Sigma_t,S_t,\hat W_t)
=
\pi_F(z_t,W_{t-H:t},P_{t-H:t}).
$$

它只输出：

- $\Sigma_t$：interaction frame；
- $S_t$：force-position selection mask；
- $\hat W_t$：desired wrench。

Foundation action 与这些参数共同进入 hybrid force-position controller：

```text
RFM action trajectory ─────────────────────────┐
                                               ↓
RFM context → compressor → z ─┐       hybrid force-position controller
wrench + proprio history ─────┴→ π_F           ↓
                         Σ, S, W_ref     regulated robot execution
```

因此 FP2 的准确描述是：

$$
\boxed{\text{preserve action generation; add interaction regulation}}
$$

而不是“force model 重新生成一个更好的 action chunk”。

### 7.2 Multi-rate structure

论文系统的频率链为：

| 层 | 频率 | 职责 |
|---|---:|---|
| Foundation policy | 2 Hz | 语言/视觉条件的 task-level action chunk 与 context |
| Native RFM action samples | 15 Hz | 原始 chunk waypoint |
| Context compressor | 随 RFM 在 2 Hz 更新 | 将大量 contextual tokens 压缩为 $z_t$ |
| Force-control policy | 50 Hz | 根据新 wrench/proprioception 更新 $\Sigma,S,\hat W$ |
| Aligned RFM trajectory | 50 Hz | DTW 对齐、quintic spline 重采样后的 position reference |
| Low-level hybrid controller | 1 kHz | 在 $S$ 指定轴上调力，其余轴跟踪 RFM motion |

Slow context 在两次 RFM inference 之间缓存；fast policy 无需重复运行完整 VLM。

### 7.3 三阶段训练

1. **Task adaptation**：按 RFM 原生流程用 task data fine-tune/LoRA，不加入 force。
2. **Context compression**：冻结 RFM，用 reconstruction objective 自监督训练 compressor；训练后丢弃 reconstruction decoder。
3. **Force-control training**：冻结 RFM 和 compressor，用 demonstration 恢复的 interaction-frame targets 训练 $\pi_F$。

Compressor 的重建目标为：

$$
\hat C_t=f_{dec}(z_t),
$$

$$
\mathcal L_{comp}
=
\mathcal L_{MSE}(C_t,\hat C_t)
+\lambda\mathcal L_{cos}(C_t,\hat C_t).
$$

这使 force policy 获得“正在执行什么任务、处于哪个动作阶段”的 RFM context，同时避免直接处理长 token sequence。

上式是论文的MSE+cosine；本地OpenPI/JAX实现为MSE，`alpha=0`只训RLT，`alpha>0`还允许VLA action loss更新上游。所选force基线的具体loss为frame L1 + force L1 + .1 mask BCE。训练目标与哪个参数更新，须按实际配置分别核查。

### 7.4 它与前三种方法的关键区别

| 方法 | Slow output | Fast output | 是否改变原 action |
|---|---|---|---|
| RDP | latent action chunk | force-conditioned motion action | 是，fast decoder 生成动作 |
| ImplicitRDP | cached slow context/noise | 重新 denoise 的 motion action | 是，最新 force 改变动作 |
| Force Policy | global feature/action | frame、mask、wrench、local motion chunk | local policy 也生成 motion |
| FP2 | RFM action chunk + compressed context | frame、mask、desired wrench | 否，保留 RFM action generation |

FP2 相对原 Force Policy 做了两项针对 RFM 的收缩：

1. 移除 local force policy 的 wrist-camera input，复用 RFM contextual representation。
2. 移除 local action re-generation，只学习 interaction regulation parameters。

这样职责边界更清晰，也避免下游小数据训练破坏 foundation model 的运动先验和 novel-object generalization。

### 7.5 能做什么，不能做什么

FP2 擅长修复：

- 接触力不足或过大；
- 接触不稳定、滑落；
- jamming 与错误 force direction；
- 曲面跟随中的 contact loss。

但它假设 RFM 已生成可行的 task-level motion。若 RFM 找错位置、几何对齐错误或任务推理错误，interaction regulator 通常不能把错误轨迹变成正确任务。这是 action-regulation decomposition 的明确能力边界。

### 7.6 面试中的一句话定义

> FP2 is a multi-rate downstream interface that freezes a task-adapted robotic foundation policy for language-conditioned action generation, compresses its contextual tokens into a cached task representation, and runs a lightweight 50 Hz policy to predict interaction-frame force-control parameters for a 1 kHz hybrid force-position controller, without re-generating the foundation-policy actions.

关于 context compressor、twist–wrench 对偶、interaction frame 构造和 hybrid controller 的逐层推导，见 [FP2 adaptor 与 twist–wrench 机制](03b_fp2_adaptor_twist_wrench.md)。

---

## 8. 本课练习

1. 为什么标准 DP 即使把 force 拼进 observation，也可能在 chunk 内保持 open loop？
2. RDP 为什么不让 force 进入 asymmetric tokenizer encoder？
3. RDP 的 latent bottleneck 为什么会损害 precise free-space motion？
4. ImplicitRDP 的 causal mask 防止了哪一种训练作弊？
5. 为什么 ImplicitRDP 使用固定 initial noise 和 $\eta=0$ 的 DDIM？
6. virtual target 为什么比 raw force prediction 更接近 action learning objective？
7. Force Policy 的 $S_i=1$ 和 $S_i=0$ 分别意味着什么？
8. RDP/ImplicitRDP 的“force-aware motion correction”和 Force Policy 的“force regulation”有什么本质区别？
9. 如果 global VLA 为 2 Hz、local force policy 为 50 Hz，应怎样处理 global feature stale、chunk transition 和意外接触？
10. FP2 与 RDP 都是 slow-fast，为什么一个 fast module 生成 motion action，另一个只生成 force-control parameters？
11. 为什么FP2论文基线不把force直接输入RFM、不让快策略重新生成action？当前源码哪些变体可以改变这条约束？

通过标准：能在 10 分钟内不看笔记画出三张数据流图，并准确说出每篇论文的 force 在哪里进入、最终 action 是什么、谁负责闭环。
