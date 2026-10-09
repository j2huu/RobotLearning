# 本地论文索引

当前共 22 篇。优先级含义：A = 当前路线精读；B = 围绕研究问题选读；C = 查表式对照。这里按概念组织，不按年份排序。

## 多模态感知与数据采集

| 优先级 | 论文 | 核心贡献 | 阅读时关注 |
|---|---|---|---|
| A | [Making Sense of Vision and Touch](../../1810.10191v2.pdf) | 自监督学习视觉、触觉、本体感觉的紧凑表示 | 表征学习怎样提高真实机器人样本效率 |
| B | [ForceMimic](../../2410.07554v3.pdf) | robot-free force-motion demonstration + hybrid force-position IL | 数据采集接口与执行接口是否一致 |
| A | [FACTR](../../2502.17432v2.pdf) | 通过视觉扰动 curriculum 迫使 BC policy 使用力信息 | 模态失衡、shortcut learning |
| B | [ManipForce](../../2509.19047v1.pdf) | 高频 F/T 与 RGB 的自然人类示范、频率感知融合 | 异步多频信号如何编码与对齐 |

## 生成式策略、顺应与混合力位控制

| 优先级 | 论文 | 核心贡献 | 阅读时关注 |
|---|---|---|---|
| A | [Admittance Visuomotor Policy](../../2409.14440v2.pdf) | diffusion 规划动作与期望接触力，admittance controller 执行 | learned policy 与经典控制的边界 |
| A | [Adaptive Compliance Policy](../../2410.09309v2.pdf) | 从示范学习时空变化的近似 compliance | stiffness/compliance 的可辨识性与代理目标 |
| A | [Reactive Diffusion Policy](../../2503.02881v3.pdf) | slow latent diffusion + fast force/tactile-conditioned asymmetric tokenizer | chunk 内闭环、显式层级与 latent bottleneck |
| B | [EquiContact](../../2507.10961v4.pdf) | SE(3)-equivariant 层级 vision-to-force policy | equivariance、局部坐标与空间泛化 |
| B | [ImplicitRDP](../../2512.10946v1.pdf) | 结构化 slow-fast visual-force diffusion policy | chunk planning 与高频 contact reaction |
| A | [Force Policy](../../2602.22088v2.pdf) | interaction frame 下的 hybrid force-position policy | global decision 与 local interaction 的解耦 |
| A | [FP2（ICRA 2027 审稿稿）](../../ICRA27_2601_MS.pdf) | 冻结 task-adapted RFM，以压缩的 foundation context 条件化 50 Hz force-control sidecar | action generation 与 interaction regulation 的解耦 |
| B | [PhaForce](../../2603.08342v1.pdf) | phase-scheduled slow planner + fast residual correction | 何时使用 force，在哪个子空间修正 |
| B | [Multi-Modal Policy Consensus](../../2509.23468v3.pdf) | 多模态 policy 通过 consensus 组合 | 模态失效和动态可靠性 |
| C | [Learning When to See and When to Feel](../../2604.01414v1.pdf) | diffusion policy 中视觉/力觉融合策略对比 | 公平消融、contact-aware gating |
| B | [CR-DAgger](../../NeurIPS-2025-compliant-residual-dagger-improving-real-world-contact-rich-manipulation-with-human-corrections-Paper-Conference.pdf) | 顺应干预接口 + residual DAgger | correction 分布、残差标签与安全干预 |

## Force-aware VLA

| 优先级 | 论文 | 核心贡献 | 阅读时关注 |
|---|---|---|---|
| A | [CompliantVLA-adaptor](../../2601.15541v2.pdf) | VLM 引导 variable impedance action | adapter 如何连接语义与阻抗控制 |
| B | [FD-VLA](../../2602.02142v2.pdf) | 训练期 force distillation，部署时不依赖力传感器 | privileged information 的收益与上限 |
| A | [CRAFT](../../2602.12532v1.pdf) | force-aware curriculum fine-tuning 与信息瓶颈 | 如何避免 V/L 高熵输入压制关键力信号 |
| A | [FAVLA](../../2602.23648v1.pdf) | slow VLM + fast force-aware action expert | 多频率闭环、action chunk 被打断/修正的机制 |
| A | [ForceVLA2](../../2603.15169v1.pdf) | force prompt + cross-scale MoE + hybrid force-position output | 力觉同时进入任务概念与实时控制的证据 |
| B | [AT-VLA](../../2605.07308v1.pdf) | adaptive tactile injection + fast tactile stream | 新模态注入对预训练能力的干扰 |
| C | [VLF-DiT（匿名审稿稿）](../../26-1958_01_MS.pdf) | vision-language-force diffusion transformer | 结果尚处审稿稿状态，侧重方法对照 |

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
