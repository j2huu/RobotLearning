# 第 3 课面试题卡：RDP / ImplicitRDP / Force Policy

## 使用方法

每题先用 30 秒给结论，再用 90 秒解释信号流、设计取舍和局限。答案中至少出现一次具体的 input/output 或 frequency，不要只复述论文摘要。

## 1. 为什么接触任务需要 slow-fast policy？

**30 秒答案骨架**

视觉提供全局几何、对象和任务进度，但大模型推理较慢；force/tactile 提供局部接触状态且变化快。普通 action chunk 在 chunk 内通常不读取新反馈。Slow-fast policy 让低频模型规划连贯动作，让高频模块用最新 force 修正执行，从而兼顾多峰轨迹建模与接触反应。

**追问**

- 高频 force sensor 是否自动带来高频 closed loop？
- policy frequency、command frequency、sensor frequency 各是多少？

## 2. 用一分钟解释 RDP

RDP 先训练 asymmetric tokenizer：encoder 将真实 action chunk 压缩成 latent，decoder 用 latent 和 tactile/force 重建动作。再训练 latent diffusion policy，根据低频视觉等 observation 生成 latent chunk。部署时 slow diffusion 以约 1-2 Hz 更新 latent intent，fast decoder 以约 24 Hz 读取最新触觉/力并输出动作。优点是快速反馈和低 latent diffusion 成本；局限是两阶段训练、latent bottleneck 和 OOD force 下的 decoder 稳定性。

**危险回答**

“RDP 就是给 Diffusion Policy 加 force。”这没有说明 chunk 内怎样闭环。

## 3. RDP 的 asymmetric tokenizer 为什么 asymmetric？

Encoder 只读 action，decoder 读 latent action 和 force/tactile。这样 latent 表示 coarse intent，接触相关的快速变化留给 decoder 根据实时反馈恢复。如果 encoder 也读取 force，latent 可能把训练时 force history 编进去，部署时 stale latent 会削弱 fast feedback 的作用。

## 4. ImplicitRDP 相对 RDP 改了什么？

它取消显式 tokenizer hierarchy，在原始 action space 用一个 Transformer diffusion policy 统一建模 slow visual context 与 fast force sequence。GRU force encoder 和 causal attention 防止未来 force 泄漏；部署时缓存 slow tokens 和 initial noise，每个 control step 加入最新 force prefix，用 deterministic DDIM 重新生成当前 prefix 的动作。

**追问**

- 为什么还叫 slow-fast？
- end-to-end 一定优于 modular 吗？

## 5. 为什么必须有 causal mask？

训练数据包含整段 action 和整段 force。若 action token $a_s$ 能读取未来 $f_{s+1:}$，模型会用由当前动作造成的未来接触结果预测当前动作，形成 information leakage。部署时未来 force 不存在，性能会崩溃。

## 6. 为什么 ImplicitRDP 使用 deterministic DDIM？

它在同一 chunk 内多次、逐步加入新 force 进行推理。如果每次采样使用不同随机噪声，相邻 action 会同时受到 feedback 和 sampling randomness 的影响，产生不连续。固定 initial noise、缓存 slow context、设置 $\eta=0$ 后，相邻输出差异主要来自新 force。

## 7. 什么是 modality collapse？

多模态模型可能找到只靠视觉就能降低平均训练 loss 的 shortcut，从而忽略低维、嘈杂、只在接触阶段有用的 force。检测方法包括 modality ablation、attention/gradient sensitivity、遮蔽实验和 contact-phase 分段评估。解决方法包括 curriculum、gating/MoE、auxiliary objective、结构性 fast path。

## 8. ImplicitRDP 为什么预测 virtual target？

由准静态 compliance：

$$
x_{vt}=x_{real}+K^{-1}f_{ext}.
$$

Virtual target 与 Cartesian action 在同一表示空间，比 raw wrench 更容易与动作目标对齐。Adaptive stiffness 让自由空间偏差小、接触时偏差大，强化关键 contact sample。它是训练辅助量，部署时丢弃。

## 9. Force Policy 与 RDP 最根本的差别是什么？

RDP 根据 force 生成 motion correction，实际 force 通过机器人与环境相互作用间接变化。Force Policy 显式输出 interaction frame、force/position selection mask、reference wrench 和 motion chunk，再由 hybrid controller 在选定轴上闭环调节力。因此 Force Policy 的控制结构更明确、更可解释。

## 10. 什么是 interaction frame？

它是接触点附近与任务约束对齐的局部 6D 坐标基。高 stiffness 轴属于 constraint subspace，适合 force control；低 stiffness 轴属于 admissible-motion subspace，适合 position control。它让“沿哪个方向移动、沿哪个方向施力”成为明确的 policy output。

## 11. Selection mask 的作用是什么？

$$
S\in\{0,1\}^6.
$$

$S_i=1$ 表示第 $i$ 个 interaction-frame axis 使用 force control；$S_i=0$ 表示使用 position control。Force Policy 还把 mask 作为 router signal，决定何时把控制权从 global vision policy 交给 local force policy。

## 12. 为什么需要 asynchronous scheduler？

Global 与 local policy 的频率和推理延迟不同。直接替换 chunk 会造成位置/速度跳变、jerk 和虚假惯性力。Force Policy 将输出重采样到统一频率，用 hysteresis 稳定路由，用 DTW 对齐新 chunk 与已执行轨迹，并生成 acceleration-continuous trajectory。

## 13. 这三篇都是 VLA 吗？

不是。RDP 和 ImplicitRDP 是 visual-force/tactile diffusion policies；Force Policy 是 modular global-local control framework，论文实验的 global backbone 是 RISE-2。它们的 slow-fast、causal fusion 和 explicit force-control interface 可以与 VLA 结合，但有 language/VLM backbone 才能称为 VLA。

## 14. 你会怎样给 π0 加 force feedback？

一个合格回答要先明确目标：语义层理解 force，还是控制层高频反应。可选：

1. 在 action expert 中加入 causal fast force stream，保持 VLM context cache；
2. 让 π0 输出 coarse chunk，外接 fast force-conditioned residual decoder；
3. 将 π0 作为 global policy，外接 interaction-frame local policy 和 hybrid controller。

随后比较 latency、训练数据、force/action 坐标系、chunk consistency、controller interface 和 safety limit。

## 15. 用一分钟解释 FP2

FP2 保留 task-adapted RFM 的原生 action generation，并冻结 RFM。它把 RFM contextual tokens 自监督压缩为一个 cached token，与实时 wrench/proprioception history 一起输入 50 Hz force-control policy。Fast policy 只预测 interaction frame、force-position selection mask 和 desired wrench；原始 RFM action 经过对齐与重采样后，和这些参数共同进入 1 kHz hybrid force-position controller。因此它添加的是 interaction regulation，不是 action re-generation。

## 16. FP2 和 RDP 都是 slow-fast，区别是什么？

RDP 的 slow policy 产生 latent action，fast decoder 根据 force 把 latent 解码成可执行 motion，因此 fast module 参与 action generation。FP2 的 slow RFM 已产生完整 action chunk；fast policy只输出 force-control parameters，低层 controller 在选定方向调节力，但保留 RFM motion reference。

## 17. 为什么 FP2 不直接把 force 输入 VLA？

Force 通常不在 foundation pretraining interface 中，直接加入会改变 observation distribution，并要求小规模 downstream data 对齐 force 与既有视觉语言表征。FP2 冻结 task-adapted RFM，把 force 留在 downstream regulator，降低灾难性干扰风险并允许 50 Hz interaction regulation。不过代价是物理反馈不能立刻反向改变当前 RFM 的高层计划。

## 18. FP2 的能力边界是什么？

它要求 foundation action 大体可行。FP2 可以减少过大力、失去接触或 jamming，但无法可靠修复错误目标、严重位置偏差或错误的 task-level motion。面试中应把它描述为 interaction failure suppressor，而不是通用 motion recovery policy。

## 19. 开放问题

- 如何证明 fast policy 使用的是有因果意义的 force，而不是相关性 shortcut？
- 遇到训练分布外的冲击力，learned fast decoder 如何保证稳定？
- 如何同时保持 VLM 语义能力与新增 force modality？
- interaction frame 是否能在不可逆形变、软体操作中成立？
- 是否可以让 model uncertainty 决定何时打断 chunk 或切换 controller？

## 口述评分表

| 维度 | 0 分 | 1 分 | 2 分 |
|---|---|---|---|
| 问题定义 | 只说接触任务难 | 提到频率差异 | 明确 chunk 内反馈缺失 |
| 系统接口 | 不清楚输入输出 | 能说主要模态 | 能说 policy output 与 controller interface |
| 数学/结构 | 只背模型名 | 能解释一个机制 | 能解释因果 mask、latent 或 selection mask |
| 取舍 | 只说优点 | 提到一个局限 | 能比较 computation、stability、generalization |
| VLA 连接 | 把三者都称 VLA | 知道可组合 | 能设计具体组合与频率路径 |

每题 8/10 以上，才进入下一篇或下一模块。
