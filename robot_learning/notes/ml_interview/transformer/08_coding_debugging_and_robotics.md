# T8：从手写 Attention 到系统面试

[目录](README.md) · [上一课](07_efficiency_and_variants.md) · [题卡](interview_drill.md)

## T8A：先写能解释的参考实现

下面是普通等宽 MHA 中“已分头 Q/K/V 的 attention 核心”，不含投影、残差、FFN、RoPE 或 cache 管理。它不是完整 Transformer。Bool `allowed=True` 始终表示允许。

```python
import math
import torch

def attention_reference(q, k, v, allowed):
    # q [B,H,Nq,dh]; k/v [B,H,Nk,dh]
    # allowed: bool，可广播到 [B,H,Nq,Nk]
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape[:2] == k.shape[:2] == v.shape[:2]
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[-2] == v.shape[-2]
    assert allowed.dtype == torch.bool
    # 参考路线在float32计算scores和聚合；不是高性能kernel。
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    allowed = torch.broadcast_to(allowed, scores.shape)
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("Every query in this reference must have a legal key")
    scores = scores.masked_fill(~allowed, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return (weights @ v.float()).to(q.dtype)

def split_heads(x, heads):
    b, n, d = x.shape
    assert d % heads == 0
    return x.reshape(b, n, heads, d // heads).transpose(1, 2)

def merge_heads(x):
    b, h, n, dh = x.shape
    return x.transpose(1, 2).contiguous().view(b, n, h * dh)
```

为什么不直接 `view(B,H,N,dh)`？原始连续内存布局是 `[B,N,H,dh]`；需要交换 token/head 轴。Shape 合法不代表元素对应正确。`transpose` 后使用 `contiguous().view`，或用能处理当前布局的 reshape。

若调用 SDPA，可用 `attn_mask=allowed, is_causal=False` 并正确设置 dropout。若调用 MHA，需转换 True 语义及形状；不能把这里的四维 mask 直接复制过去。

上面是供阅读和云端自行运行的参考代码；本次不将标准库 Notebook 的结果当作它的运行证据。

## T8B：面试现场用性质定位错误

| 要验证的机制 | 操作 | 应观察到什么 | 不能证明什么 |
|---|---|---|---|
| 因果依赖 | 只扰动未来 token | 过去输出不变 | 一次检查不能覆盖所有路径 |
| Padding 隔离 | 替换 PAD 内容 | 有效输出不变 | 不替代 label/metric mask 检查 |
| Mask 真正生效 | 故意开放一条未来路径 | 选定非退化输入上出现变化 | 变化幅度不等于任务影响 |
| 多层安全 | 让 prefix 读 target 再堆两层 | 出现间接泄漏 | 单层正常不能说明深层正常 |
| Cache 正确性 | full vs incremental | 对应层/位置输出近似一致 | 不证明实际加速 |
| Cache 位置正确 | 故意用局部三角/重置位置 | 与全量结果不同 | 文本相同仍可能有logits错误 |

执行时控制 dropout、随机种子、模型状态和 dtype；检查输出 finite，使用有意义的非零输入。训练梯度要另做 backward/有限差分等检查，不能由 forward 等价性推断。

[TR01](notebooks/TR01_attention_masks_and_cache.ipynb)使用标准库模拟上述核心机制，便于逐行理解。它的两层 toy 只有 attention+residual，省略 learned projections、FFN 和 norm；它验证这些固定输入上的依赖关系，而非学习到的能力。

**三个常见错误的解释：**

1. `softmax(dim=-2)`：归一化的是 queries，改变“每个 query 从 keys 读取”的含义；输出 shape 可能依旧正确。
2. `loss = (loss * valid).mean()`：除以 padded 总数，样本长度分布改变有效梯度尺度；正确分母是所选监督单位的有效数量。
3. `is_causal=True` 配一个 cached query：若采用左上对齐非方阵语义，它可能只能读第一个 key，而非全部合法历史。

## T8C：迁移到 VLA 与机器人系统

语言模型的序列轴通常是文本顺序。机器人策略至少还有 observation 时间、动作 horizon、生成/去噪时间、控制执行时间。这些轴不能混为一谈。

**场景：** 在物理时刻 t，根据已获得的相机图像、语言和 state，生成未来 H 步动作。

- 若把动作离散化为自回归 tokens，动作 token 间按所选序列顺序 causal，并处理 label shift。
- 若用 diffusion/flow 联合生成动作块，每一步采样中所有 noisy action slots 已存在；块内双向读取可以合理。未来动作变量是待生成对象，不等于未来真实观测。
- 训练目标可以用未来 expert action 提供监督；但直接把干净真值当作部署时不存在的条件输入，会破坏任务一致性。Noisy input 含有来自真值的信息是去噪训练目标的一部分，应与相应采样过程一起解释。
- 若 condition prefix 会读不断变化的 noisy action，它的深层表示可能随每步更新，不能未经检查就复用 prefix cache。

把 mask 讲清楚的方法：列出每类 token 的来源、在何时可得、哪些是候选生成变量、哪些只用于 loss，再画读取表。具体模型还需核对实际代码，不能把教学结构直接当作 π0 或 FP2 的事实。

FP2 的压缩上下文 z 与 AR KV cache 缓存对象不同。压缩是表达瓶颈，KV cache 是特定依赖条件下的计算复用；两者都不能自动消除旧图像带来的 sensor age。继续阅读 [M8 应用](../08_attention_and_compression.md)及[策略 Transformer 主线](../../07_policy_transformer_and_causality.md)。

### 一道 15 分钟系统题

“策略输入 256 个视觉 tokens、32 个文本 tokens、1 个 state token，生成 16 个动作 slots。怎样设计 mask、训练和缓存？”

建议回答顺序：

1. 先问动作采用 AR 还是 flow/diffusion，明确部署可见条件和控制接口。
2. 写输入 `[B,289,d]` 和动作 `[B,16,d]`；若 concat，整段长 305。拆出有效性、位置和 modality 语义。
3. 为条件/动作两组写 block 读取表；说明条件是否允许读动作，以及对 cache 的影响。
4. 写对应目标和有效 loss 分母；解释缺失相机、padding、episode 边界。
5. 给出未来扰动、双层泄漏、cache 等价性检查。
6. 测实际预处理、推理、排队、执行延迟和闭环效果；分别报告 toy、模型与机器人证据。

**一分钟英文练习：** “A Transformer block combines content-dependent token mixing with token-wise nonlinear transformations. I define the allowed information flow before choosing a mask API. For autoregressive decoding, cached keys and values are reusable because earlier hidden states do not depend on newly appended tokens. I verify that property by comparing full and incremental computation under matching positions and model settings.”

## 自测与答案

1. 动作 horizon 长度 16 是否一定要 16 次模型 forward？**不一定**，取决于 AR tokenization、块生成和采样机制。
2. Cache 等价性通过可声称机器人实时性提高吗？**不能**，需单独测时延和闭环表现。
3. 面试被问“这个 mask 对不对”，第一步？**明确行列含义、True 语义和各 token 可得时间**，再看矩阵。
