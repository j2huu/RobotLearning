# T2：Attention 怎样组成完整 Transformer

[目录](README.md) · [上一课](01_attention_from_scratch.md) · [下一课](03_positions_and_rope.md)

## T2A：一次信息交换还不等于一层网络

将 token 想成各自维护一份特征表。Attention 在位置之间交换信息；FFN 对每个位置的特征做非线性变换；residual 保留原表示并加入更新；normalization 调整特征尺度。层与层重复，才形成多步推理和间接信息路径。

一个简化 Pre-LN block：

$$
U=X+\operatorname{Attention}(\operatorname{LN}_1(X)),\qquad
Y=U+\operatorname{FFN}(\operatorname{LN}_2(U)).
$$

这里 Attention 包含 Q/K/V 和输出投影；省略 dropout。两个 LN 有各自参数，标准深层模型各层参数也通常独立。Pre-LN stack 常在最终输出再加一个 norm，不能只看一个 block 就推断整个模型。

普通 FFN：

$$
\operatorname{FFN}(x)=\sigma(xW_1+b_1)W_2+b_2,
\quad W_1:[d,d_{ff}],\ W_2:[d_{ff},d].
$$

它对每个 token 使用同一套权重，但标准逐位置 FFN 不混合 token 轴。“逐位置”不是各 token 拥有不同参数。没有非线性时，两层线性可合并为一层，扩宽本身不能提供相同的非线性表达力。

Gated FFN 的示意式为 $(\operatorname{SiLU}(xW_g)\odot xW_u)W_d$，比普通两投影 FFN 多一个投影；中间宽度需要配合参数预算选择。不要给所有模型套用固定 $d_{ff}=4d$。

## T2B：把 normalization 真正算一遍

对单个 token $x\in\mathbb R^d$：

$$
\mu=\frac1d\sum_rx_r,\qquad v=\frac1d\sum_r(x_r-\mu)^2,
\qquad \operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{v+\epsilon}}+\beta.
$$

`x=[1,3]` 时，均值 2、方差 1；忽略 epsilon 且 gamma=1、beta=0，输出 `[-1,1]`。这里使用特征总体方差，分母为 d。对 `[B,N,d]` 使用 `LayerNorm(d)` 时，每个样本每个 token 各算一组统计量，既不跨 batch，也不跨时间。

RMSNorm 的基本式：

$$
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac1d\sum_rx_r^2+\epsilon}}.
$$

同一 `[1,3]` 输出约 `[0.4472,1.3416]`，均值不是 0。RMSNorm 不减均值，不能称为“完全一样的 LayerNorm”。[RMSNorm 论文](https://arxiv.org/abs/1910.07467)

Post-LN 则写成 $U=\operatorname{LN}(X+\operatorname{Attention}(X))$，第二子层同理。Pre-LN 的 residual 通道给梯度提供较直接的路径；Post-LN 的路径会经过 norm Jacobian。优化差异与深度、初始化和学习率共同有关。“Pre-LN 永不需要 warmup”或“Post-LN 无法训练”都过度概括。[Pre-LN/Post-LN 分析](https://arxiv.org/abs/2002.04745)

**手算参数：** d=12、普通 FFN 中间宽度 48，忽略 bias/norm，FFN 有 $2\times12\times48=1152$ 参数，attention 有 576，总计 1728。Norm 和 bias 另算。这解释为什么面试中不能只统计 attention。

## T2C：Encoder、Decoder 和任务

| 常见结构 | 典型读取方式 | 常见目标 | 不应混淆 |
|---|---|---|---|
| Encoder-only | 有效输入内双向 self-attention | 表示学习、分类、masked prediction | 双向可见不等于提前读取未提供的真值 |
| Decoder-only | 常用 causal self-attention | next-token prediction | 名称不表示内部必有 cross-attention |
| Encoder-decoder | encoder 双向；decoder causal + cross | 条件生成、翻译 | decoder 的 self 和 cross 有不同 K/V |

以翻译为例：encoder 读完整源句，decoder 在第 t 步读已生成目标前缀，并 cross-attend 到源句。Cross 的 $N_q$ 是目标长度，$N_k$ 是源长度，通常不需要对源句强加目标的下三角 mask；仍要遮住源 padding。若源数据是流式、未来部分不可用，则要另外设计时间可见性约束。

一个 seq2seq decoder block 常有三个子层：masked self-attention、cross-attention、FFN，每个配 residual/norm。Decoder-only block 常只有 self-attention 和 FFN。模型命名不能代替检查结构。

BERT 的双向上下文与 masked language modeling 见 [BERT](https://arxiv.org/abs/1810.04805)；统一 text-to-text 的 encoder-decoder 案例见 [T5](https://arxiv.org/abs/1910.10683)。这些是任务与结构的实例，不是所有 encoder/decoder 都只能使用一种 loss。

## 自测与答案

1. 标准逐 token LN 会把未来 token 混进当前 token 吗？**不会**；若误把 N 轴也纳入 norm，结论会改变。
2. 只用 residual 就一定稳定？**不一定**，累积尺度、初始化、优化器和深度仍重要。
3. Self-attention 是动态卷积吗？可类比内容相关聚合，但**不能直接等同**；连接范围、共享方式和位置归纳偏置不同。
4. 画一个 decoder-only block，要画哪些部分？**Norm、Q/K/V、mask、softmax、V 聚合、O 投影、residual、FFN 及第二 residual**，并标明 norm 位置。
