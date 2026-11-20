# Transformer 速查表

[目录](README.md) · [详细题卡](interview_drill.md)

## 一次 forward 的主链

`token/patch/state → embedding + position → 多层 attention/FFN/residual/norm → task head → logits 或连续输出`

$$
Q=X_qW_Q,\ K=X_kW_K,\ V=X_vW_V,\qquad
Y=\operatorname{softmax}(QK^T/\sqrt{d_h}+M)V.
$$

Scores `[B,Hq,Nq,Nk]`，softmax 在 Nk 上；输出长度由 Nq 决定。标准 MHA 投影约 $4d^2$ 参数，普通 FFN 约 $2dd_{ff}$，均忽略 bias。

## Mask 四问

| 先问 | 核对点 |
|---|---|
| 行列是什么？ | 行 query，列 key |
| True 是什么？ | 本文/SDPA 允许；MHA 禁止 |
| 谁在推理时可得？ | 条件、右移历史、候选变量、训练真值分别判断 |
| 间接路径是否存在？ | 逐层检查，不能只删直接边 |

Causal：$key\_pos\le query\_pos$。Padding：有效 key。Packing：独立 segment。Loss：监督位置，不能阻止 attention 泄漏。有效 query 不应出现全禁行。

```text
输入    BOS A B       label A B EOS
允许    1 0 0
        1 1 0
        1 1 1

cache长度3，新增2个query，key总长度5：
        1 1 1 1 0
        1 1 1 1 1
```

## 高频公式和区别

| 主题 | 核心 |
|---|---|
| Pre-LN | $x+F(LN(x))$；Post-LN 为 $LN(x+F(x))$ |
| LayerNorm | 每 token feature 均值/方差；RMSNorm 不减均值 |
| RoPE | $R(i\theta)^TR(j\theta)=R((j-i)\theta)$ |
| CE | $-z_y+\log\sum_v e^{z_v}$；对有效 targets 平均 |
| KV bytes | $2LBNH_{kv}d_hs$；不含其他显存 |
| Dense attention | $O(BN^2d)$；投影为 $O(BNd^2)$ |
| GQA | 减少 KV heads；Q heads 仍可很多 |
| FlashAttention | 改善 IO/中间存储，标准 dense 目标不变 |
| Local/sparse | 改读取边；不默认等价于 dense |
| eval / no_grad | 模块行为 / 梯度记录，不是同一开关 |

## Debug 优先级

先对齐 input/target → 再核对 mask 语义和位置 → 检查有限性及空行 → 做未来/padding 扰动 → 比较 full/cache → 最后定位性能。正确性不靠“loss 在下降”或“生成文字看起来合理”判断。

资料及接口说明见 [resources](resources.md)；表内符号和推导条件见各课，不能脱离假设直接套用。
