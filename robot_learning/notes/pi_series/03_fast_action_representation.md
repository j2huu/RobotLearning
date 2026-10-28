# P3：FAST——动作表示为什么会改变训练和推理

前课：[P2](02_flow_matching.md)。原始来源：[FAST](https://arxiv.org/abs/2501.09747)，重点 §IV–V；[π0.5 v1](../../../paper/2504.16054v1.pdf) §IV-B/D。本课补齐理解 π0.5 配方需要的动作表示知识，不要求训练完整 tokenizer。

## 1. 先分清四个名字

| 名称      | 是什么                        | 动作怎么生成                              |
| ------- | -------------------------- | ----------------------------------- |
| π0      | flow VLA                   | action expert 数值积分生成连续 chunk        |
| FAST    | action tokenizer           | 把连续 chunk 编码成较短离散序列，并可解码回去          |
| π0-FAST | 使用 FAST 的 AR policy        | 自回归生成 action tokens，再解码             |
| π0.5 v1 | 异构 co-training 的 VLA 与训练系统 | 训练使用离散动作和连续动作目标；最终低层推理用 flow expert |

FAST 不是给 π0 减少 ODE steps 的 sampler。FAST+ 指通用 tokenizer 变体；它也不是 π0.5 或后者的 FAST+Flow baseline。

## 2. 朴素离散化的成本从哪里来

**教学计算**：50 个动作、每个 14 维，如果每个标量各用一个 token，一次 chunk 就有 $50\times14=700$ 个符号，尚未计 prompt、边界和 special tokens。

高频数据相邻动作经常接近。逐标量 binning 的序列既长，又有大量容易预测的局部变化。一个好的动作表示应压缩重复结构，同时保留对控制有用的精度。

“相邻很像”不代表可以随意删去关键动作：夹爪切换、接触瞬间和快速修正可能只占少量时间，却决定任务成败。

## 3. FAST 的编码路径

FAST 原文 §IV 的关键链路为：

```text
normalized chunk
   → 每个动作维度沿时间做 DCT
   → 量化系数
   → 按约定序列化
   → BPE 压缩为离散 tokens
```

解码逆转这条链路。DCT 将时间冗余变成容易压缩的频域结构；量化控制精度，BPE 合并常见符号模式。具体归一化、系数顺序、量化尺度和词表由 tokenizer 定义。

DCT 本身在保留全部精确系数时是可逆线性变换；量化会有损。BPE 压缩不等于必须截去所有高频系数，也不是说任何动作都由低频成分决定。这些区分来自编码链路本身。

本课不把 VQ-VAE、RDP tokenizer 和 FAST 混在一起：前两者可有 learned encoder/decoder 或 latent bottleneck；FAST 的核心是变换、量化与序列压缩。比较时始终说明“学习的是 policy、codebook、压缩词表，还是重建网络”。

## 4. Token policy 的训练

给定 $z_{1:L}=T(A)$，AR 目标为：

$$
\mathcal L_{CE}=-\sum_{i=1}^{L}m_i\log p_\theta(z_i\mid z_{<i},c).
$$

$m_i$ 是目标 loss mask。训练时 teacher forcing 提供正确前缀 $z_{<i}$，因果 mask 允许并行计算各位置的 logits；推理时前面的 tokens 来自模型，需要顺序生成。

**因此，“AR 训练能并行”和“AR 推理要逐步解码”可以同时成立。** Teacher forcing 不允许位置 $i$ 偷看自己的答案或后续答案。

一个常见工程错误是：attention mask 正确，但 label shift 错了，模型在当前位置重建了输入 token；另一个是把 prompt/padding 也计入 loss，低平均 CE 掩盖动作预测不佳。

## 5. 训练快，不自动意味着部署快

π0.5 v1 §IV-B/D 使用 FAST 离散预训练来学习表示，再加入 flow expert 以获得连续动作和实时推理。这是特定训练配方的选择，不是所有离散模型都必然慢于所有连续模型的定理。

**教学成本模型**：

$$
T_{AR}\approx T_{prefix}+\sum_{i=1}^{L}T_{decode,i}+T_{decode\_action},
$$

$$
T_{flow}\approx T_{prefix}+K\,T_{expert}+T_{postprocess}.
$$

减少 $L$、减少 $K$、缩小 expert、改 cache 都可能降低耗时；收益需实际 profiling。Tokens/sec、每 chunk latency 和每秒完成任务数是不同指标。

## 6. 三种误差不能混成一种

先只测 tokenizer 的 round trip：

$$
A\xrightarrow{T}z\xrightarrow{T^{-1}}\tilde A,
\qquad E_{codec}=\|\tilde A-A\|.
$$

然后测 policy 对 token 序列的预测，最后测闭环执行。误差来源依次是：

1. **表示误差**：正确 token 解码后已丢失控制精度。
2. **预测误差**：模型生成了不同 token 序列。
3. **系统误差**：动作定义、时延、controller 或状态估计有问题。

这是教学诊断分解，不意味着三个误差可以直接线性相加。接触系统的非线性可能放大小变化；aggregate reconstruction MSE 好，不代表接触阶段也足够好。

## 7. 面试中的选型回答

如果数据少、硬件有限，先选能验证端到端接口的 baseline，再比较更复杂模型。若考虑 AR，应测 token 长度分布、codec error 和解码耗时；若考虑 flow，应测 expert 步数、动作连续性和在线性能。

避免“连续一定精确”“离散一定表达差”“FAST 一定最快”三种回答。报告动作单位、数据、运行配置和测量口径，才构成可比较的证据。

## 8. 练习与验收

1. 20 Hz、7 维、1 秒 chunk，朴素序列有几个标量 tokens？答案：140，不含附加 tokens。
2. 训练并行预测 140 个位置，为什么部署仍可能慢？答案必须提 teacher forcing 与真实解码的差异。
3. 对插接任务，除平均 codec MSE 外，应检查什么？分维最大误差、接触阶段、夹爪切换、重建时序和执行结果。
4. 若 tokenizer round trip 很好、rollout 却差，应再看哪些路径？输入条件、语言利用、预测误差、OOD、单位和 latency。
5. π0.5 推理时是否先生成 FAST actions，再交给 flow 精修？按 v1 所述最终低层推理并非这个串联流程；两种 action representations 在训练 attention 中隔离，见下一课。

英文口述：

> Distinguish an action tokenizer from a policy. Explain how compressed action tokens affect teacher-forced training, autoregressive inference, and control precision.

通过标准：写出 CE、画训练/推理流程，并用延迟模型解释训练与部署为何可能选择不同表示。下一课：[P4：π0.5](04_pi05_generalization.md)。
