# T7：复杂度、GQA、FlashAttention 与部署取舍

[目录](README.md) · [上一课](06_kv_cache_and_decoding.md) · [下一课](08_coding_debugging_and_robotics.md)

## T7A：先分清参数、Activation、Cache 和计算

标准 dense MHA，$d=Hd_h$，忽略常数及 bias：

| 项目 | 单层主要量级 | N 翻倍时 |
|---|---|---|
| Q/K/V/O 参数 | $4d^2$ | 不变 |
| 投影计算 | $O(BNd^2)$ | 约 2 倍 |
| $QK^T$ 与 AV 计算 | $O(BN^2d)$ | 约 4 倍 |
| 显式 attention matrix | $BHN^2$ 个元素 | 4 倍 |
| 普通 FFN 计算 | $O(BNdd_{ff})$ | 约 2 倍 |
| KV cache | $2LBNH_{kv}d_h$ 个元素 | 2 倍 |

“Transformer 是 O(N²)”描述的是标准 dense attention 对序列长度的主要项，不代表所有长度、宽度和 batch 下 attention 都是主要耗时。短序列大宽度时，投影或 FFN 也可能占较多计算。训练还需要梯度、优化器状态和保存的中间量，不能拿推理 KV 容量当训练显存。

**手算：** B=1,H=8,N=2048，显式 score 矩阵有 33,554,432 元素；FP16 下单份约 64 MiB。N=4096 时为 256 MiB。实际实现不一定保存这张完整矩阵；训练也可能需要更多中间量。

Prefill 同时处理许多 queries；decode 每次通常只有少量 queries，却要读权重和历史 KV。在某些负载下 decode 更受带宽影响，prefill 更能利用矩阵计算；这不是无条件定律，需结合 batch、硬件、精度和 kernel 测量。

## T7B：三种不同的优化方向

### MHA、MQA 和 GQA 改变 KV head 数量

| 结构 | $H_q$ | $H_{kv}$ | 共享关系 |
|---|---:|---:|---|
| MHA | H | H | 每个 query head 对应独立 KV head |
| MQA | H | 1 | 所有 query heads 共享一组 K/V |
| GQA | H | G，通常整除 H | 每组 query heads 共享一组 K/V |

例如 Hq=8,Hkv=2，每 4 个 query heads 用同一 KV head，但各自 Q 仍不同，所以各自 attention 权重仍可不同。缓存理论容量相对 Hkv=8 降到 1/4；Q 和输出宽度并没有随之变成 1/4。模型权重形状与训练设计需要匹配，不能任意删掉已有 heads 并保证质量不变。[GQA](https://arxiv.org/abs/2305.13245)

### FlashAttention 改变计算的组织方式

核心是分块计算，并用可合并的 softmax 统计量避免把完整 N×N attention matrix 反复写入显存。它保持 dense attention 的数学目标，主要改善 IO 和中间存储；并没有普遍把 dense pairwise 算术量从二次变为线性。浮点舍入使结果无需逐 bit 相同。[FlashAttention](https://arxiv.org/abs/2205.14135)

**在线 softmax 的手算理解：** 一块保留最大值 m、指数和 l、未归一化加权和 u。已有块 `(m,l,u)` 和新块 `(m',l',u')` 合并时，取 $M=\max(m,m')$：

$$
l_{new}=e^{m-M}l+e^{m'-M}l',\quad
u_{new}=e^{m-M}u+e^{m'-M}u',\quad y=u_{new}/l_{new}.
$$

对 `[0,ln3]`、values `[2,10]` 分成两个单元素块：第一块 `(0,1,2)`，第二块 `(ln3,1,10)`，合并得 $l=4/3,u=32/3$，输出仍为 8。这是在线归一化的教学演示，不是 GPU kernel 实现。

### Sparse/local attention 改变连接范围

局部窗口宽 w 时，注意力部分可降至约 $O(Nwd)$，但一些远距离边被删掉，模型的信息路径已经改变。跨多层可以逐步扩大可达范围，也可配全局 token；不能默认与原 dense 模型等价。

所谓 linear attention 往往更改核函数或归一化形式以重排运算，也不能简单认为“精确计算任意 softmax attention，只是更快”。面试时先说保留什么、改变什么，再谈收益。

## T7C：从瓶颈选择方案

| 观察到的问题 | 可考虑的方向 | 应验证的代价 |
|---|---|---|
| 长 prompt 的中间 attention 显存高 | fused/Flash attention | mask、dtype、设备支持与数值一致性 |
| 长序列 decode 的 KV 容量高 | GQA/MQA、KV 量化、窗口 | 模型支持、质量、量化误差和依赖变化 |
| 权重装不下 | 权重量化、分片 | 精度、通信、kernel 支持 |
| 训练 activation 高 | activation checkpointing | 用额外重算换显存，不等于 KV cache |
| padding 浪费大 | 长度分桶、packing | 样本边界、position 和 label 对齐 |
| 单请求延迟高 | 定位 prefill/decode/排队时间 | batch 提吞吐可能增加等待 |

Paged KV 管理着重处理分配、碎片与共享；不意味着模型在数学上少需要所有有效 KV。具体实现是否支持某种优化，要查正在使用的版本并测量，不在讲义里预设收益。

报告性能时至少说明：模型、硬件、精度、batch、输入/输出长度、warmup、是否包含预处理和排队。区分首 token 时间、后续 token 时间、总延迟、吞吐量与显存峰值。

## 自测与答案

1. FlashAttention 与 GQA 是同一种优化吗？**不是**，前者组织 dense attention 的 IO，后者改变 KV head 共享。
2. 参数量不变，N 翻倍显存为何仍上涨？**Activation 和 KV 都依赖 N**。
3. KV 少 4 倍就端到端快 4 倍吗？**不能推出**，还有投影、FFN、权重读取与其他开销。
4. 利用率低就一定计算不够吗？**不一定**，也可能等待数据、受内存带宽或小矩阵效率限制。
