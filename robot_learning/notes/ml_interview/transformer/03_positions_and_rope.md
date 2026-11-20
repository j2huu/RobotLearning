# T3：Transformer 如何知道位置和顺序

[目录](README.md) · [上一课](02_blocks_norm_and_architectures.md) · [下一课：Mask](04_masks_and_information_flow.md)

## T3A：先理解“没有位置信息”意味着什么

考虑没有位置编码、没有位置相关 mask/bias 的标准 self-attention。令 P 是 token 置换矩阵：

$$
Q'=PQ,\quad K'=PK,\quad V'=PV,\quad
Q'K'^T=P(QK^T)P^T.
$$

按行 softmax 后再乘 $PV$，结果为原输出的同样置换：$f(PX)=Pf(X)$。这是**置换等变**，不是输出逐元素完全不变。若再做对称 mean pooling，整个 pooled 表示才可能置换不变。

“狗 咬 人”和“人 咬 狗”包含相同 token 集，但角色不同；仅靠不带位置结构的对称汇聚难以区分。需要把顺序信息带进计算。

反过来，有 causal mask 时，可见集合本身已经与位置有关，不能继续直接使用上述无 mask 的置换等变证明；这也不意味着 mask 足以替代所有位置表示。证明结论必须带上假设。

## T3B：从加法位置到旋转位置

绝对位置的一种方式是 $x_i=e(token_i)+p_i$。$p_i$ 可学习，也可用固定正弦/余弦。两者都让同一 token 在不同位置具有不同输入。学习表的位置范围有限；固定函数能计算更大 i，不保证模型在超出训练长度后仍准确。

经典正弦编码：

$$
PE(i,2r)=\sin(i/10000^{2r/d}),\qquad
PE(i,2r+1)=\cos(i/10000^{2r/d}).
$$

不同频率为不同跨度提供变化尺度。Relative bias 则直接给 score 加一个位置差相关项 $b(i-j)$，不一定改 token embedding。

### RoPE 的二维推导

对 Q/K 的每一对分量做旋转，先看二维：

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix},
\quad \tilde q_i=R(i\theta)q_i,\quad \tilde k_j=R(j\theta)k_j.
$$

由于 $R(a)^TR(b)=R(b-a)$：

$$
\tilde q_i^T\tilde k_j=q_i^TR((j-i)\theta)k_j.
$$

位置差进入了 attention score。一般按多对分量使用不同频率；常见设计旋转 Q/K，V 不按同样方式旋转。不要把 RoPE 说成“往 embedding 上加一个位置向量”。[RoFormer](https://arxiv.org/abs/2104.09864)

**手算：** $q=k=[1,0]$，$\theta=\pi/2$。i=0,j=1 时旋转后内积为 0；i=1,j=2 仍为 0。相同相对位移保持这里的分数。若 i=j，内积为 1。

这个演示固定了原始 q/k。真实模型的 q/k 内容会变；不能推出“任意相隔相同距离的 tokens 都有相同分数”。

## T3C：位置编码怎样影响工程正确性

Cache 中的 K 往往已经带有对应位置的旋转。新增 token 必须用它在当前序列中的正确逻辑位置；每步都重置为 0 会改变相对相位，使缓存和全量计算不一致。

```text
有效文本：     A B C
左 padding：  _ _ A B C
逻辑位置：        0 1 2    （一种常见策略）
物理列号：    0 1 2 3 4
```

逻辑位置与 tensor 列号不一定相同。模型若使用另一种一致的位置约定也可能正确，但不能训练时一种、带 padding 的推理时另一种。需要同时核对 position IDs、key validity、cache length 和读取的最终 logits。

长上下文不仅是“位置函数能算到”。还涉及训练长度分布、频率/缩放策略、相关信息检索能力以及计算资源。不要仅凭 RoPE 数学可延展，就宣称无需额外训练即可无限延长。

视觉 patch 还有二维位置，多帧视频增加时间轴，机器人状态也有时间语义。展平后的 token index 未必等于物理时间。模态 embedding、空间位置与采样时间各解决不同歧义。

## 自测与答案

1. 无位置 attention 的输出是否不变？**一般等变，不是逐位置不变**。
2. 同时给所有逻辑位置加常数，二维 RoPE 内积为什么保持？**相对位置差不变**，前提是原始 q/k 不变且旋转规则一致；完整模型可能还有其他绝对位置信号。
3. 单步 cache 推理和全量推理不同，除了 mask 查什么？**位置、层级 cache、padding、模型模式、dtype 和权重一致性**。
4. 能计算长度 100000 的位置编码就能可靠理解 100000 tokens 吗？**不能**；需要任务评测和实际资源验证。
