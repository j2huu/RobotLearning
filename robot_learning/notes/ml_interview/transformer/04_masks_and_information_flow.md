# T4：Mask 专课——谁可以读取谁

[目录](README.md) · [上一课](03_positions_and_rope.md) · [下一课](05_training_objectives_and_generation.md)

本课必须能独立讲清：causal mask 控制信息依赖，padding mask 排除无效输入，loss mask 选择监督位置。三者解决不同问题，不能相互替代。本文矩阵行=query、列=key，1/True=允许。

## T4A：先画四个 token，再写代码

### Causal mask 为什么是下三角

输入 `[BOS, 我, 爱, 猫]`，每个位置预测下一个 token。位置 i 可读所有 j≤i：

```text
              keys
              BOS 我 爱 猫
query BOS      1  0  0  0
query 我       1  1  0  0
query 爱       1  1  1  0
query 猫       1  1  1  1
```

允许矩阵是下三角；禁止矩阵则是严格上三角。面试里说“上三角 mask”前应先交代 True 的意思。

为什么对角线可见？位置“我”的输出用于预测“爱”，输入“我”已经是已知历史，不是这个位置的答案。模型输入、loss label 如何错开，比盲目去掉对角线更重要。如果输入和目标不 shift，残差连接还能直接携带当前 token，光删 self-attention 对角线也无法阻止复制。

因果性这里指**计算依赖符合可用信息顺序**，不是说模型完成了因果推断。Token 顺序也不一定等于物理时间，VLA 动作块见 T8。

### Padding mask 控制输入可用性

同一个 batch 有 `[BOS,A,B,C]` 与 `[BOS,D,PAD,PAD]`。第二个样本的有效 keys 为 `[1,1,0,0]`。

$$
A_{b,i,j}=\mathbf1[j\le i]\land key\_valid_{b,j}.
$$

这会阻止其他位置读取 PAD，但不自动把 PAD query 的输出清零。无效 query 可以有任意有限输出，只要不会作为合法 key 被读取，且不会进入 loss/池化/指标。必须沿后续计算继续检查。

Padding embedding 为 0 不够：位置编码、bias、残差和后续层都可能让它变成非零；即使 V 是零，它仍可能占 softmax 分母，改变合法内容权重。

### Loss mask 不改变信息流

若只训练回答部分，可令 prompt 的 label 为 ignore，但 prompt 仍是回答的可读条件。反过来，某个答案 token 不计 loss，并不意味着其他 token 不能偷读它。

$$
\mathcal L=\frac{\sum_{b,i} m^{loss}_{b,i}\ell_{b,i}}{\sum_{b,i}m^{loss}_{b,i}}.
$$

Loss mask 是监督选择；attention mask 是输入依赖。只有一个有效 token 时应除以 1，不是 padded 总长度。

## T4B：Prefix、packing 与块依赖

### Prefix-LM：条件双向，生成部分因果

设条件 `P0 P1` 在生成前已完整可得；目标输入 `Y0 Y1 Y2` 是右移后的生成序列：

```text
          P0 P1 Y0 Y1 Y2
P0         1  1  0  0  0
P1         1  1  0  0  0
Y0         1  1  1  0  0
Y1         1  1  1  1  0
Y2         1  1  1  1  1
```

Prefix 可以双向，不代表整段序列可双向。关键是 P 不读 Y，否则未来答案可以先进入 prefix，再被其他目标位置读到。

### Packing：拼到一行不等于变成同一样本

把样本 A 的三 tokens 和 B 的两 tokens 拼接，独立样本训练可使用：

$$
A_{i,j}=\mathbf1[segment_i=segment_j]\land\mathbf1[position_j\le position_i].
$$

还要处理跨样本 next-token label：A 的最后位置不应错误预测 B 的第一个 token。Mask、位置重置和目标 shift 要一起检查。也存在有意跨文档拼接的训练策略；先明确任务是否要求隔离，不能仅看到 EOS 就假定信息被屏蔽。

### 一般 block mask：先问部署时能获得什么

条件 C、待生成变量 Z、仅训练时可见的真值 D：

| query \ key | C | Z | D |
|---|---:|---:|---:|
| C | 允许 | 禁止 | 禁止 |
| Z | 允许 | 取决于生成目标 | 禁止 |

这只是设计起点。若 Z 是 flow/diffusion 的 noisy action，块内双向可能合理；若 Z 是 next-token 的已右移目标，就要满足自回归顺序。不能给所有 Transformer 都套同一张 causal mask。

## T4C：API、数值和跨层路径

### 从统一 allowed 转换到接口

| 接口 | Bool True 含义 | 常见 shape |
|---|---|---|
| 本文 `allowed` | 允许 | `[B,1,Nq,Nk]`，广播到 heads |
| `F.scaled_dot_product_attention` 的 `attn_mask` | 允许 | 可广播到 `[B,H,Nq,Nk]` |
| `nn.MultiheadAttention` 的 `attn_mask` | 禁止 | `[Nq,Nk]` 或 `[B*H,Nq,Nk]` |
| `nn.MultiheadAttention` 的 `key_padding_mask` | 忽略该 key | `[B,Nk]` |

Additive mask：允许 0，禁止 $-\infty$，在 softmax **之前相加**。Bool 和 float mask 不是把 0/1 dtype 随意转换就等价。[MHA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)、[SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

示意构造（PyTorch，完整参考见 T8）：

```python
causal = k_pos[None, :] <= q_pos[:, None]   # [Nq,Nk]
allowed = causal[None, None] & key_valid[:, None, None, :]
# 注意：MHA 的 attn_mask 不能直接传这个四维张量；需要按接口转换。
```

### 全部屏蔽的一行为什么危险

`softmax([-inf,-inf])` 在数学上没有可归一化的概率；稳定实现中减最大值会遇到 `-inf - (-inf)`。不同 kernel 可能有特殊处理，但不能依赖未确认行为。

把 -∞ 换成统一 -1e9 也不是通用修复：全屏蔽时反而可能得到均匀权重，读取本该禁止的 values。FP16 下过大的负数还可能溢出。

对于**有效 query**，应保证至少一个合法 key，否则输入契约或 mask 有错。对于无效 query，可先给一条安全的 dummy 读取边，得到有限结果后把该 query 输出置零，同时屏蔽它作为后续 key 并排除 loss。必须在产生 NaN 之前处理；`NaN*0` 仍是 NaN。

### 看单层矩阵不足以排除泄漏

假设目标 C 不能直接读未来 D，但允许 C 读 P，而 P 可以读 D：

```text
第1层：D 的原始内容 → P 的第1层表示
第2层：P 的第1层表示 → C 的第2层表示
```

两层以后 C 仍依赖 D。逐 token FFN 和 LN 不会新增跨位置边，但跨 token pooling、错误归一化、卷积、预处理也可能产生路径。审查范围应覆盖全部层和输入构造。

**扰动检查：** 固定权重、位置、mask 和随机性，只改未来输入，比较过去输出；再改 padding 内容，比较有效输出。正确模型中差异应在数值容差内。用多种输入，并加入故意错误的 mask 作为正向对照；一次零差异可能只是退化参数让泄漏路径没有数值影响。

## 自测与答案

1. 输入 `[BOS,A,B]`，标签 `[A,B,EOS]`，第一行能看 A 吗？**不能**，那是它要预测的答案；只看 BOS。
2. Prompt 不算 loss，回答还能读 prompt 吗？**能，而且通常需要**。
3. 独立 packed sequences 有 causal mask 就够吗？**不够**，后一个样本仍能读前一个，还可能有跨边界 label。
4. Prefix 读 target，target 不读未来 target，安全？**不安全**，两层可间接泄漏。
5. Cached decode 的 causal mask 还是从局部 row=0 开始的下三角吗？**通常不是**；要用全局逻辑位置，下一课后在 T6 展开。
