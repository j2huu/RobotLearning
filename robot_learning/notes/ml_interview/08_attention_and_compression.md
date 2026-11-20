# M8：Attention、Transformer、mask 与 FP2 compressor

[目录](README.md) · [上一课](07_representation_learning.md) · [下一课](09_generative_models.md) · [Colab ML05](../../labs/notebooks/ML05_attention_compressor.ipynb)

本课保留 Attention 与 FP2 compressor 的应用概览。需要系统准备通用 Transformer 面试时，进入 [T1–T8 渐进专题](transformer/README.md)：从手算、完整 block 和 RoPE，学到 [Mask 专课](transformer/04_masks_and_information_flow.md)、训练标签对齐、KV cache、GQA/FlashAttention 与 coding 诊断，不必从 FP2 场景倒推所有基础。

## 1. Q、K、V 是信息检索的三个角色

$$
Q=X_qW_Q,\quad K=X_kW_K,\quad V=X_vW_V,
\qquad Attention(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

Self-attention 的输入序列来源相同；cross-attention 的 query 与 K/V 来自不同序列或表示。不是靠 Q/K/V 的矩阵尺寸是否相同来区分。

## 2. 完整 shape

若 B 为 batch、h 为 head 数，query 长度 Nq、key/value 长度 Nk，每头宽度 dₕ：

```text
Q：[B,h,Nq,d_h]
K：[B,h,Nk,d_h]
V：[B,h,Nk,d_h]
Q @ K.transpose(-1,-2)：[B,h,Nq,Nk]
softmax over key axis：[B,h,Nq,Nk]
attention @ V：[B,h,Nq,d_h]
拼接 heads / 输出投影：[B,Nq,d]
```

输出 token 数由 query 长度 Nq 决定。`softmax` 轴写错可能仍得到合法 tensor，但机制已改变。

## 3. FP2 为什么 N→1

$$
C:[B,N,D]\rightarrow Proj(C):[B,N,d],
\quad Q_0:[1,1,d]\rightarrow z:[B,1,d].
$$

learned query 在 batch 轴广播，cross-attend 到 projected context。一个 query 汇聚 N 个 tokens。它不是 wrench 编码；compressor 的 forward 只接收 C。

训练时 decoder 用 learned reconstruction queries 读取 z，重建 C；MSE + cosine distance 提供自监督。完成后丢弃 decoder，冻结 compressor，再训练 force policy。详见 [FP2 tensor walkthrough](../03c_fp2_tensor_walkthrough.md)。

## 4. Causal mask、padding mask 与间接泄漏

Causal mask 禁止当前读取未来；padding mask 排除无效 slots。Bool mask 的 True 表示“允许”还是“禁止”依赖 API，不能在不同 attention 函数间盲目复用。

Notebook 的 `nn.MultiheadAttention` 用上三角 True 表示禁止读取未来。检查方式是扰动未来 token，比较过去输出，而不只目视一个 mask 矩阵。

多层网络还可能存在间接路径。例如 C 不能直接读 target D，但 prefix P 先读 D，下一层 C 再读 P，仍会泄漏。Pi 机制 notebook 保留了这个两层反例。

## 5. Position、residual、normalization 与 cache

没有顺序信号的标准 attention 对输入置换具有相应对称性；位置信息可以来自 embedding、relative bias、RoPE 或其他结构，不必全是绝对位置编码。

Residual 和 MLP 是 Transformer block 的组成部分；cross-attention 不是整块网络的全部。Norm 的位置影响优化，但仅凭图示不能补写论文未公布的具体层数和 Pre-LN/Post-LN 配置。

KV cache 复用既有计算，不使已过时的视觉/物理信息自动更新。FP2 的 cached z 是压缩上下文缓存，与 autoregressive KV cache 不是同一个对象。需要分别记录生成时间、可用时间与 sensor age。

## 6. Notebook 实现边界

ML05 用合成低秩 context 训练 cross-attention bottleneck，再冻结它，训练读取 pose/wrench history 的三头 sidecar。frame/mask/wrench targets 是合成规则，不是 FP2 数据；没有真实 RFM、机器人或 force-control controller。

它可以验证形状、重建目标、freeze 与信息依赖，不能复现论文成功率。一个 token 重建表现好也不保证对所有任务保留了足够信息。

## 验收

手算 [B,h,1,N] 权重；解释为何属于 cross-attention；写出未来 token perturbation check；分清参数冻结、context cache 与高频 feedback。

参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)。
