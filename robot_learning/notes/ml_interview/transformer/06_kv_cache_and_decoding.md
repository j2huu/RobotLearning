# T6：KV Cache——从依赖关系推导缓存

[目录](README.md) · [上一课](05_training_objectives_and_generation.md) · [下一课](07_efficiency_and_variants.md)

## T6A：Prefill 和 Decode 在做不同的工作

Prompt 有 P 个 tokens。Prefill 一次处理整个 prompt，产生各层 K/V，并用最后有效位置的 logits 采出第一个新 token。随后把这个新 token 输入模型，在每层生成它自己的 Q/K/V：新 Q 读取旧 K/V 加当前 K/V，得到下一个 token 的 logits。

```text
prefill: 输入 [A B C] → cache 长度3 → 采出 D
decode:  输入 [D]     → cache 长度4 → 采出 E
decode:  输入 [E]     → cache 长度5 → 采出 F
```

刚采出的 token 还没有自动进入 cache；需要作为下一次 forward 的输入。这是常见的 off-by-one 来源。

为什么缓存 K/V 而不是所有旧 Q？新输出需要当前 query 对历史 keys 打分，并读取历史 values；标准增量解码不再重算旧位置输出，因此旧 Q 不参与这次计算。

### 为什么旧 K/V 可以不变

第 1 层的旧 token 输入不变；因果约束使旧位置不能读后来 token，因此其第 1 层输出不变。递推到每一层，旧 hidden 和旧 K/V 都不变。这是依赖图上的归纳，而不是“缓存总是近似可用”。条件包括参数固定、位置和 mask 规则一致、随机性受控，以及没有其他把新 token 混入旧 token 的操作。

双向 encoder 加入新 token 后，旧位置可以读取它，旧 hidden 通常改变，不能直接套用这个缓存证明。固定 encoder 输出的 cross-attention K/V 则可以有另一套复用理由。

## T6B：形状、层数和显存

每层缓存：

```text
past_K, past_V             [B,Hkv,P,dh]
new_Q                     [B,Hq,1,dh]
new_K, new_V               [B,Hkv,1,dh]
concat K/V                [B,Hkv,P+1,dh]
scores（映射对应KV组后）    [B,Hq,1,P+1]
```

每层 K/V 来自该层自己的输入和投影，不是把第一层 K/V 复制 L 份。GQA 中多个 query heads 共享一个 KV head，详见 T7。

固定长度 N、元素字节数 s 的理论 KV 容量：

$$
\text{bytes}=2LBNH_{kv}d_hs.
$$

教学例子 L=12、B=1、N=1024、$H_{kv}=8$、$d_h=64$、FP16 s=2：共 25,165,824 bytes，即 24 MiB。这里不含权重、activations、allocator、临时 buffers、分页元数据，也不代表某模型实测显存。Batch 翻倍或上下文翻倍，基础 KV 容量线性翻倍。

缓存减少历史投影和层计算，但每个新 query 对 N 个历史 key 的 dense attention 仍需约 $O(Nd)$ 工作；生成长度增加时，注意力部分的累计工作仍可能是二次量级。不能说“有 cache 后每个 token 都是 O(1)”或“所有生成复杂度都变线性”。

## T6C：最容易错的非方阵 mask

已有 3 个 cached tokens，当前一次处理 2 个新 tokens。Query 全局位置为 `[3,4]`，key 位置 `[0,1,2,3,4]`。

正确允许矩阵：

```text
q at 3:  1 1 1 1 0
q at 4:  1 1 1 1 1
```

若直接取一个 2×5 左上对齐下三角，则变成：

```text
wrong:   1 0 0 0 0
         1 1 0 0 0
```

它不是允许了未来，而是**错误屏蔽了大部分合法历史**，同样破坏结果。单个新 query 且 K/V 恰好只含合法过去和当前时，应允许它读所有这些有效 keys。

```python
# 示意：q_pos/k_pos 是全局逻辑位置，不是局部张量行号。
allowed = k_pos[None, :] <= q_pos[:, None]
# 还需要叠加 batch 各自的 key validity / segment 限制。
```

SDPA 文档对 `is_causal=True` 的非方阵采用左上对齐语义；有 cache 时不要盲目套用。在教学实现中显式传正确位置 mask，关闭额外 `is_causal`。框架包装器可能替你处理 offset，必须检查实际入口。[SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

### 什么变化会让 cache 失效

| 改变 | 判断 |
|---|---|
| 追加 token，旧依赖和参数不变 | 满足标准因果缓存条件 |
| 修改 prompt 中间 token | 该位置及依赖它的后缀各层 cache 需更新 |
| optimizer step、更换 LoRA/权重 | 旧投影不再对应当前模型 |
| position IDs、RoPE 规则变化 | 旧位置表示可能不兼容 |
| 重新排序 batch/beam | cache 必须跟着样本或 beam 重排 |
| 图像/机器人 state 更新 | 依赖这些条件的缓存需要重算 |

滑动窗口删掉旧 cache 是模型语义的一部分：不能默默把所有剩余 token 的绝对位置重新归零，并假定等价于原模型。需要按其位置策略处理。

## 自测与答案

1. 最重要的正确性验收？**同一输入在全量 causal forward 与分步 cached forward 的对应输出一致**，固定 eval、dropout、位置、dtype，并设数值容差。
2. 为什么只有一层等价还不够？**第二层 cache 必须来自正确的第一层输出**，逐层管理或 mask 错误可能只在深层出现。
3. Cache 是否存储新传感器信息？**不会**；没有新输入，缓存无法更新环境状态。
4. 只检查生成文本一样够不够？**不够**；argmax 相同可掩盖 logits 偏差，应比较数值输出。
