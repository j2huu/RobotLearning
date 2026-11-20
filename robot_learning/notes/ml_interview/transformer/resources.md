# Transformer 原始资料与阅读顺序

[目录](README.md)

资料核对日期：2026-11-18。以下采用原始论文与官方文档；网页的 stable 文档会随版本变化，实际编码应同时记录本机 `torch.__version__`。本专题的小数字、任务与 toy 实验是独立教学材料，不是论文报告的实验。

| 资料 | 对应课 | 阅读时要回答 |
|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | T1–T3 | attention、FFN、残差、位置和 encoder-decoder 怎样连接 |
| [BERT](https://arxiv.org/abs/1810.04805) | T2/T5 | 双向上下文怎样与遮盖预测目标结合 |
| [T5](https://arxiv.org/abs/1910.10683) | T2/T5 | 架构、训练目标和 text-to-text 任务怎样区分 |
| [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | T2 | Pre/Post-LN 的梯度与优化条件 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | T2 | 去掉均值中心化保留了什么 |
| [RoFormer](https://arxiv.org/abs/2104.09864) | T3 | 旋转如何把相对位移带进 Q/K 内积 |
| [GQA](https://arxiv.org/abs/2305.13245) | T7 | 共享 KV 的范围、质量与资源取舍 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | T7 | IO-aware exact attention 与近似稀疏变体的区别 |
| [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) | T4/T8 | bool 语义、2D/3D attn mask、key padding shape |
| [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | T4/T6/T8 | True=允许、非方阵 causal 对齐、dropout 参数 |

第一遍无需逐篇通读。先完成 T1 的手算、T4 的矩阵、T5 的 shift 和 T6 的 cache 推导；再用论文验证机制与适用条件。原论文架构、现代实现变体和机器人具体模型应分别标明来源。
