# Transformer 渐进专题：T1–T8

[ML 面试目录](../README.md) · [M8 概览与 FP2 应用](../08_attention_and_compression.md) · [40 道专题题卡](interview_drill.md) · [速查表](cheatsheet.md)

新增：2026-11-18。本专题补足通用 Transformer 面试基础，独立使用 T 编号，保留 M1–M10。每课分 A/B/C 三个单元，共 24 单元。先把一个数算对，再把一层图画清，再讨论模型和系统。术语是 **causal mask（因果掩码）**，也可说 causality；不是 casual。

## 学习路线

| 课 | A：建立直觉 | B：手算与推导 | C：实现与面试 | 完成标准 |
|---|---|---|---|---|
| [T1 Attention](01_attention_from_scratch.md) | 从平均读取到 Q/K/V | softmax、缩放、梯度 | 多头、shape、投影参数 | 手算一个输出并解释每个轴 |
| [T2 完整网络](02_blocks_norm_and_architectures.md) | attention/FFN/residual 分工 | LayerNorm、Pre/Post-LN | encoder、decoder、cross-attention | 画出一个完整 block |
| [T3 位置与顺序](03_positions_and_rope.md) | 置换与顺序 | 绝对位置、RoPE 旋转 | cache 位置、长上下文 | 推导相对位移如何进入内积 |
| [T4 Mask 专课](04_masks_and_information_flow.md) | causal/padding/loss | prefix、packing、块依赖 | API、空行、多层泄漏 | 写出并验证信息可达关系 |
| [T5 训练与目标](05_training_objectives_and_generation.md) | 自回归分解与 shift | CE、有效 token 平均 | teacher forcing、生成、诊断 | 对齐 input/label/logit 三行表 |
| [T6 KV cache](06_kv_cache_and_decoding.md) | prefill/decode | 多层 cache 与显存 | 非方阵 mask、失效条件 | 比较全序列和逐步计算 |
| [T7 效率与变体](07_efficiency_and_variants.md) | 时间、显存、带宽 | MHA/MQA/GQA、FlashAttention | 长上下文、部署决策 | 用数字定位瓶颈 |
| [T8 Coding 与系统面试](08_coding_debugging_and_robotics.md) | 最小 attention 实现 | 扰动和等价性检查 | VLA 迁移、综合题 | 能解释 bug 为什么发生 |

前置：知道向量内积、矩阵乘法、softmax；不熟悉时回看 [M4](../04_tensors_and_backprop.md)。不需要先学完整 LLM，也不要求本地 GPU。

## 每次只学一个单元

1. 读 A/B/C 对应内容之前，先尝试回答该节问题。
2. 在纸上计算文中的小例子，写 shape 和 mask 的行列含义。
3. 关闭答案，用 60–90 秒讲“目的 → 公式 → 例子 → 限制”。
4. 每课最后有自测与答案；答错后回到对应机制，不直接背题卡。

每单元预留 30–50 分钟，全部约 12–20 小时，均为学习预算。若只有两小时：T1B → T4A/C → T5A → T6B/C → 题卡中的 mask/cache 题。已有 VLA 主线[第 7 课](../../07_policy_transformer_and_causality.md)可用于机器人应用复习，本专题负责更完整的通用基础。

## 统一符号

| 符号 | 含义 |
|---|---|
| $B,N,d$ | batch、token 数、模型宽度 |
| $H_q,H_{kv},d_h$ | query heads、KV heads、每头宽度；标准 MHA 中 $H_q=H_{kv}=H$ |
| $N_q,N_k$ | query 和 key 长度；解码时经常不同 |
| $L,d_{ff},V$ | 层数、FFN 中间宽度、词表大小 |
| $A_{ij}$ | 第 i 个 query 是否允许读取第 j 个 key；本文 True/1 始终表示允许 |
| $M_{ij}$ | 加到 logits 的 mask：允许为 0，禁止为 $-\infty$ |

数学投影采用 $XW$；PyTorch `nn.Linear` 的 weight 存储方向相反。特别注意：允许矩阵 A 不一定能原样交给所有 API，T4 有转换表。

## 配套代码与证据

[TR01 机制 Notebook](notebooks/TR01_attention_masks_and_cache.ipynb) 自带标准库代码和固定小输入：attention 手算、未来扰动、padding、两层间接泄漏、RoPE 相对位置、cache 等价性及错位 mask 反例。不下载权重，不依赖仓库路径。它不是训练完整 Transformer，也不测机器人能力。执行与检查范围见 [VALIDATION](VALIDATION.md)。

[T8](08_coding_debugging_and_robotics.md)另提供 PyTorch 参考实现及验收思路；Notebook 的标准库验证不等于该 PyTorch 代码已经运行，更不等于已在 Colab 验证。

## 阅读资料与学习进度

[资料索引](resources.md)按问题列出原始论文和官方 API 文档。先读本地例子，再查来源。新增内容表示材料已准备，个人掌握程度从“待验收”开始：

| 范围 | 阅读 | 手算 | 独立口述 | 本人代码执行 |
|---|---|---|---|---|
| T1–T3 | 待记录 | 待记录 | 待记录 | 待记录 |
| T4–T6 | 待记录 | 待记录 | 待记录 | 待记录 |
| T7–T8 | 待记录 | 待记录 | 待记录 | 待记录 |
