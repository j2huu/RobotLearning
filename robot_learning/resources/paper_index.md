# 本地论文索引

当前共 24 篇。优先级含义：A = 当前路线精读；B = 围绕研究问题选读；C = 查表式对照。这里按概念组织，不按年份排序。

PDF 集中存放在项目根目录的 [paper/](../../paper/README.md)。

## 通用 VLA / Pi 主干

| 优先级 | 论文与本地版本 | 核心贡献 | 阅读时关注 |
|---|---|---|---|
| A | [π0: A Vision-Language-Action Flow Model for General Robot Control](../../paper/2410.24164v1.pdf)，arXiv:2410.24164v1，2024-10-31；[来源](https://arxiv.org/abs/2410.24164v1) | 预训练 VLM + continuous flow action expert，跨机器人 pre/post-training | §IV/V、附录 B/D/E；expert/mask、flow 符号、执行时序、部分进度评分 |
| A | [π0.5: a Vision-Language-Action Model with Open-World Generalization](../../paper/2504.16054v1.pdf)，arXiv:2504.16054v1，2025-04-22；[来源](https://arxiv.org/abs/2504.16054v1) | 异构 co-training、FAST 预训练与 flow post-training、同模型高低层推理 | §IV/V、附录 B–E；六类数据、注意力隔离、场景/对象泛化与消融范围 |

对应 [Pi 系列独立专题](../notes/pi_series/README.md)。精读依据为本地 v1 全文；公开 openpi 与 KI 的版本差异在课程中单独标明。FAST 当前仍是 [外部阅读](https://arxiv.org/abs/2501.09747)，不计入本地 PDF 数量。

## 多模态感知与数据采集

| 优先级 | 论文 | 核心贡献 | 阅读时关注 |
|---|---|---|---|
| A | [Making Sense of Vision and Touch](../../paper/1810.10191v2.pdf) | 自监督学习视觉、触觉、本体感觉的紧凑表示 | 表征学习怎样提高真实机器人样本效率 |
| B | [ForceMimic](../../paper/2410.07554v3.pdf) | robot-free force-motion demonstration + hybrid force-position IL | 数据采集接口与执行接口是否一致 |
| A | [FACTR](../../paper/2502.17432v2.pdf) | 通过视觉扰动 curriculum 迫使 BC policy 使用力信息 | 模态失衡、shortcut learning |
| B | [ManipForce](../../paper/2509.19047v1.pdf) | 高频 F/T 与 RGB 的自然人类示范、频率感知融合 | 异步多频信号如何编码与对齐 |

## 生成式策略、顺应与混合力位控制

| 优先级 | 论文 | 核心贡献 | 阅读时关注 |
|---|---|---|---|
| A | [Admittance Visuomotor Policy](../../paper/2409.14440v2.pdf) | diffusion 规划动作与期望接触力，admittance controller 执行 | learned policy 与经典控制的边界 |
| A | [Adaptive Compliance Policy](../../paper/2410.09309v2.pdf) | 从示范学习时空变化的近似 compliance | stiffness/compliance 的可辨识性与代理目标 |
| A | [Reactive Diffusion Policy](../../paper/2503.02881v3.pdf) | slow latent diffusion + fast force/tactile-conditioned asymmetric tokenizer | chunk 内闭环、显式层级与 latent bottleneck |
| B | [EquiContact](../../paper/2507.10961v4.pdf) | SE(3)-equivariant 层级 vision-to-force policy | equivariance、局部坐标与空间泛化 |
| B | [ImplicitRDP](../../paper/2512.10946v1.pdf) | 结构化 slow-fast visual-force diffusion policy | chunk planning 与高频 contact reaction |
| A | [Force Policy](../../paper/2602.22088v2.pdf) | interaction frame 下的 hybrid force-position policy | global decision 与 local interaction 的解耦 |
| A | [FP2（ICRA 2027 审稿稿）](../../paper/ICRA27_2601_MS.pdf) | 冻结 task-adapted RFM，以压缩的 foundation context 条件化 50 Hz force-control sidecar | action generation 与 interaction regulation 的解耦 |
| B | [PhaForce](../../paper/2603.08342v1.pdf) | phase-scheduled slow planner + fast residual correction | 何时使用 force，在哪个子空间修正 |
| B | [Multi-Modal Policy Consensus](../../paper/2509.23468v3.pdf) | 多模态 policy 通过 consensus 组合 | 模态失效和动态可靠性 |
| C | [Learning When to See and When to Feel](../../paper/2604.01414v1.pdf) | diffusion policy 中视觉/力觉融合策略对比 | 公平消融、contact-aware gating |
| B | [CR-DAgger](../../paper/NeurIPS-2025-compliant-residual-dagger-improving-real-world-contact-rich-manipulation-with-human-corrections-Paper-Conference.pdf) | 顺应干预接口 + residual DAgger | correction 分布、残差标签与安全干预 |

## Force-aware VLA

| 优先级 | 论文 | 核心贡献 | 阅读时关注 |
|---|---|---|---|
| A | [CompliantVLA-adaptor](../../paper/2601.15541v2.pdf) | VLM 引导 variable impedance action | adapter 如何连接语义与阻抗控制 |
| B | [FD-VLA](../../paper/2602.02142v2.pdf) | 训练期 force distillation，部署时不依赖力传感器 | privileged information 的收益与上限 |
| A | [CRAFT](../../paper/2602.12532v1.pdf) | force-aware curriculum fine-tuning 与信息瓶颈 | 如何避免 V/L 高熵输入压制关键力信号 |
| A | [FAVLA](../../paper/2602.23648v1.pdf) | slow VLM + fast force-aware action expert | 多频率闭环、action chunk 被打断/修正的机制 |
| A | [ForceVLA2](../../paper/2603.15169v1.pdf) | force prompt + cross-scale MoE + hybrid force-position output | 力觉同时进入任务概念与实时控制的证据 |
| B | [AT-VLA](../../paper/2605.07308v1.pdf) | adaptive tactile injection + fast tactile stream | 新模态注入对预训练能力的干扰 |
| C | [VLF-DiT（匿名审稿稿）](../../paper/26-1958_01_MS.pdf) | vision-language-force diffusion transformer | 结果尚处审稿稿状态，侧重方法对照 |

## 这个目录呈现出的研究脉络

```text
是否能感知接触？
  → 怎样采集同步、高频、可迁移的 force/tactile demonstrations？
  → 怎样避免视觉占据表示、模型忽略低维力信号？
  → policy 应输出运动、力、阻抗，还是它们的组合？
  → 低频生成式 action chunk 怎样接收高频闭环修正？
  → 怎样把这些机制接到预训练 VLA，同时保留语义泛化？
```

这批论文最有价值的横向比较轴包括：force 输入位置、融合频率、action space、低层 controller、训练时是否需要 force、部署时是否需要 force，以及是否真正进行闭环反馈。
