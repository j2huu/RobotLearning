# 第 7 课：策略 Transformer——谁读取谁，为什么，以及如何验证

[上一课](06_multimodal_policy_representation.md) · [循序学习路线](../planning/vla_learning_path.md) · [下一课](08_cvae_and_diffusion_policy.md) · [Colab VLA01](../labs/notebooks/VLA01_token_action_trace.ipynb) · [两层泄漏反例](../labs/notebooks/Pi_mechanisms.ipynb)

第 6 课已得到 tokens。本课要回答：**一个 action 位置怎样读取图像、语言和 state；哪些路径必须禁止；这些限制如何影响缓存与训练正确性？**

继续使用 VLA01 教学配置：图文 prefix 38 个、state 1 个、noisy action 8 个，总计 47 个 tokens；batch 为 2，hidden width 为 32，attention heads 为 4。它是共享的两层小 Transformer，借用 π0 v1 的块依赖教学，不是 π0 的多 expert 实现。

## 怎么学这一课

| 小节 | 本次只解决什么 | 建议时间（估计） | 离开前留下什么 |
|---|---|---:|---|
| 7A | Q/K/V 到底怎样把信息从一个 token 送到另一个？ | 45–60 分钟 | head shapes 和一行手算 attention |
| 7B | 机器人策略为什么需要不同的 masks？ | 40–55 分钟 | P/S/A 表和 masked loss 分母 |
| 7C | 如何发现多层泄漏，什么时候能缓存？ | 35–50 分钟 | 两条扰动实验与缓存失效条件 |

前置自检：能说明 `[B,N,d]` 的每一轴、Linear 的输入输出和 softmax 的归一化意义即可。若 Q/K/V 只记得名字，照着 7A 手算；需要额外背景时查 [M8 attention](ml_interview/08_attention_and_compression.md)。暂时不用掌握完整 flow matching，先把 noisy action 理解为“当前正在生成的一组候选动作变量”。

## 7A：attention 是一次有条件的读取

### 7A.1 先从一个 query 开始

假设一个待生成动作位置需要决定夹爪往哪走。它可以读取场景中的不同视觉 patches，也可以读取“红杯”相关文字及机器人 state。Query 描述当前读取位置的匹配特征；Key 为可读取来源提供匹配特征；Value 携带被加权聚合的内容。它们都是 learned projections，不能预设每个 head 必然对应某个可解释概念。

对 hidden sequence $X\in\mathbb R^{B\times N\times d}$，简化写为：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

若 Q 来自 action tokens、K/V 来自另一组 context，就构成 cross-attention；若三者来自同一序列，是 self-attention。判断依据是张量来源，而不是模型是否包含多种模态。将图文和动作 concat 后对同一序列做 attention，可以是带 mask 的 self-attention。

### 7A.2 把 multi-head 的每个轴写出来

本例每个 head 的宽度 $d_h=32/4=8$，使用标准等宽 Q/K/V 教学：

```text
X                               [2,47,32]
Q / K / V 的整体投影             [2,47,32]
reshape 为4个 heads             [2,47,4,8]
交换 token/head 轴              [2,4,47,8]
K transpose 最后两个轴          [2,4,8,47]
Q @ K^T                         [2,4,47,47]
逐query对key做softmax            [2,4,47,47]
attention weights @ V           [2,4,47,8]
合并heads                       [2,47,32]
output projection               [2,47,32]
```

Scores 的倒数第二轴是 query，最后一轴是 key。**每一行在“可读取的 keys”上归一化**，不是跨 batch，也不是在 feature width 上归一化。

对一个 head：

$$
S_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}+M_{ij},
\qquad \alpha_{ij}=\frac{\exp(S_{ij})}{\sum_m\exp(S_{im})},
\qquad y_i=\sum_j\alpha_{ij}v_j.
$$

标准 additive mask 中，允许时 $M_{ij}=0$，禁止时 $M_{ij}=-\infty$。缩放因子在常见独立、近似单位方差的假设下，减轻 dot product 随维度增加而变大的问题，避免 softmax 过早变得极端。它不保证任意训练状态都恰好满足这些分布假设。

### 7A.3 做一次不用代码的手算

假设某个 query 对两个合法 keys 的**缩放后 logits**是 `[0, ln(3)]`，两个标量 values 是 `[2,10]`。

1. softmax 权重为 `[1/4,3/4]`。
2. 聚合结果为 $\frac14\times2+\frac34\times10=8$。
3. 若第二个 key 被禁止，logits 变成 `[0,-∞]`，权重为 `[1,0]`，结果为 2。

这解释了为什么 mask 应进入 softmax 之前：禁止来源不应占据归一化分母。仅把聚合后的输出某些值置零，并不能撤销已经混入的禁止信息。

**反向检查。** 将 value 从 10 改成 100，但不改这个 head 的 Q/K，attention 权重是否必然变化？

**答案提示。** 本次计算的权重不变，输出从 8 变成 75.5。Q/K 决定这一步的权重，V 决定被聚合的内容。后续层可能再受变化后的 hidden states 影响。

### 7A.4 Attention 之后为什么还有 residual 和 MLP

Attention 让不同位置交换信息；标准 Transformer 的 position-wise MLP 对每个位置做非线性特征变换，通常共享权重。Residual path 让原有表示与新变化共同保留。LayerNorm、residual 和 MLP 的前后次序取决于具体结构，不应仅凭一个 attention 公式宣称已经完整复现 Transformer。

VLA01 使用两层 `TransformerEncoderLayer`，每层 4 heads、FFN width 64、dropout 0。各模态经过同一组层；两层的参数彼此独立，并非跨层绑定权重。最后只取 8 个 action positions 的 hidden states `[2,8,32]`，经 `Linear(32,7)` 得到 `[2,8,7]`。该输出在 notebook 中是生成过程的 vector field；它不是“32 维 token 直接就是 32 自由度命令”。

**7A 完成标准：** 能写出 `[2,4,47,47]` 的四个轴，解释 softmax 沿哪个轴，手算一行 attention，并用来源判定 self-attention / cross-attention。

## 7B：先问部署时有什么，再设计 mask

### 7B.1 三块输入的依赖关系

令 P 为当前图文 prefix，S 为当前 state token，A 为整个 noisy action chunk。VLA01 使用以下 π0 v1 风格的 block 依赖；它不是所有 VLA 的通用 mask：

| Query / Key | P：图文条件 | S：当前 state | A：noisy action |
|---|---:|---:|---:|
| P | 允许 | 禁止 | 禁止 |
| S | 允许 | 允许 | 禁止 |
| A | 允许 | 允许 | 允许 |

**读表方法：先选行，说明谁在读；再选列，说明读谁。** P 的表示只由 P 决定；S 可以结合图文；A 可以结合全部条件及其他动作候选。

把各块临时缩成一个 token，additive mask 为：

$$
M=\begin{bmatrix}
0 & -\infty & -\infty\\
0 & 0 & -\infty\\
0 & 0 & 0
\end{bmatrix}.
$$

回到真实序列，左上块是 38×38，state 块是 1×1，action 块是 8×8；每个允许块内部双向读取。它是块三角结构，不能替换成对全部 47 个位置逐 token 的普通下三角 mask。

### 7B.2 为什么未来动作位置可以互读

“未来”有两种含义，先分清对象：

| 对象 | 什么时候存在？ | 是否能作为当前生成过程的一部分？ |
|---|---|---|
| 后续 8 个动作的 noisy variables | 当前采样过程已经持有 | 可以联合建模、互相读取 |
| demonstration 的 clean action chunk | 离线训练时可见 | 可构造加噪输入及监督目标；推理没有它 |
| 执行 0.5 秒后实际测得的力/图像 | 未来物理过程才发生 | 当前部署条件不能包含 |

Flow/diffusion 每次迭代更新一整段候选动作，所以 action positions 互读不等于读取真实未来。训练时用 clean action 构造 noisy variable 是训练目标的一部分；额外把 clean target 无条件直接拼入 context，会让推理缺少训练依赖的信息。

Teacher-forced autoregressive 模型按另一种分解预测动作 token，通常限制每个预测位置读取后续目标。第 12 课再与 FAST 对照。不能用“Transformer 都 causal”或“action chunk 都双向”概括全部架构。

### 7B.3 三种 mask 分别解决三个问题

| mask | 问题 | 本例 shape / 作用 |
|---|---|---|
| dependency mask | 架构允许谁读谁？ | `[47,47]`，同一规则用于 batch 各样本 |
| key validity mask | 这个输入位置真实存在吗？ | `[2,47]`，第二个样本的两个 text padding 禁止被读 |
| target / loss mask | 这个目标值有合法监督吗？ | `[2,8]`，第二个样本最后两步不计监督 |

两个样本的 action target 一共有效 $8+6=14$ 步，每步 7 个坐标，因此 valid scalar 数是 98。若误差平方逐坐标求和：

$$
\mathcal L=
\frac{\sum_{b,h,j}m_{b,h,j}(\hat u_{b,h,j}-u_{b,h,j})^2}
{\sum_{b,h,j}m_{b,h,j}},
\qquad m\in\{0,1\}^{2\times8\times7}.
$$

这里将 `[2,8]` target-valid mask 广播到 `[2,8,7]` 后再求分母。若分子逐坐标求和却只除 14，得到的是每步坐标误差和，其数值比每有效 scalar 的 MSE 大 7 倍；必须明确你希望哪种归约。

VLA01 只将无效 action target 排除出 loss，仍允许 noisy action slots 参与 attention。这是教学实现的选择，不能据此断言全部真实模型都应如此。要区分 action target padding、物理 action 维度 padding、输入缺失，它们不必共用一张 mask。

### 7B.4 实现前先确认 bool 语义

Notebook 的 `TransformerEncoder` 接收与内部 `MultiheadAttention` 对应的 bool dependency mask：`True` 表示禁止；`src_key_padding_mask=~key_valid` 中 `True` 也表示忽略该 key。直接使用 `scaled_dot_product_attention` 时，bool `attn_mask=True` 表示允许参与 attention，语义相反。调用接口变化时应查文档并用最小例子验证。[PyTorch MHA](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)、[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

还应检查有没有有效 query 的全部 keys 都被禁止。对全 `-∞` 行直接做数学 softmax 没有有效归一化；实现可能返回特殊值或产生 NaN，不能依靠最终 loss 乘零修复已污染的 hidden states。若出现 NaN，先检查 mask 方向、有效 key 数和数值，而不是先调整学习率。

**手算练习。** 若第二个样本最后两步 target 无效，为什么不一定应把它们的所有 attention 边都删掉？为什么 text padding 只做 loss mask 不够？

**答案提示。** target-valid 描述监督是否存在，候选生成变量是否参与计算由架构决定。Text padding 是条件输入的无效位置，即使它没有独立 loss，其他合法 token 仍可能读到它，因此需在信息读取处处理。

**7B 完成标准：** 先画 P/S/A 表再写代码；独立算出 98 个有效 scalar；能用一个例子区分 loss mask 和 input mask。

## 7C：用依赖关系解释 cache，用扰动发现泄漏

### 7C.1 Cache 节省什么，何时失效

一次 observation 固定时，生成模型会多次更新 noisy A 和 generation time。由于 P 不能读取 A，P 的每层表示可以保持独立于这些更新；若实现支持，就可能复用相应层的 prefix K/V。

推导这个结论要有条件：P 的原始输入不变、所有层都没有从变化来源进入 P 的路径、参数固定、位置与 mask 一致，且用于比较的随机性受控。单看最后一层不够。标准逐 token LayerNorm / MLP 不会在位置间引入新路径，但其他跨位置算子需要单独检查。

| 发生的变化 | 对 prefix cache 的判断 |
|---|---|
| 同一次 observation，仅 noisy A / 生成时间更新 | 在本例 block 依赖下，可满足 prefix 复用的依赖条件 |
| image / text tokens 改变 | 条件表示变了，需要更新对应缓存 |
| 仅单独的 S 更新且 P 不读 S | 理论上 P 仍可独立；S 及依赖它的部分需重新处理 |
| state 已经 tokenized 到 P 内 | state 变化会改变 P，不能套上一行结论 |
| 权重改变，例如一次 optimizer update | 旧 cache 不再对应新模型 |

VLA01 没有实现 KV cache：每次 forward 仍重新运行整个小 Transformer。它只检查“改变 A 是否改变 P”，用于理解 cache 的依赖前提。真实实现与收益需要查 [P5 源码路径](pi_series/05_openpi_engineering.md) 和实际时延；不能从理论独立性直接宣称加速多少。

### 7C.2 为什么检查单层 mask 还不够

设 D 是部署时不应读取的 future force，P 是 context，A 是当前 action。如果第一层允许 P 读取 D，第二层允许 A 读取 P，就有：

```text
future force D^(0) → context P^(1) → action A^(2)
```

即使 A 每层都禁止直接读取 D，信息仍能通过 P 传递。MLP / residual 不会自动消除已混入的信号。应检查整个多层依赖图，而不是只看 A→D 那个格子。

“隐藏了输入列名”也无效：若你把未来 force 预先编码进 context，mask 看到的是 context token，禁止原始 force 位置已经来不及。

### 7C.3 两个带预期结果的扰动实验

先固定权重、其他输入和随机种子，关闭 dropout 或使用确定性的比较设置。每次只改一种来源：

| 实验 | 改什么 | 观察什么 | 本实现的预期 |
|---|---|---|---|
| Prefix 独立性 | noisy actions 加 100 | 所有 prefix hidden 的最大绝对差 | 数值误差范围内接近 0 |
| Text padding 隔离 | 只改 invalid text IDs | 有效 action outputs 的差 | 数值误差范围内接近 0 |

VLA01 用两层网络、`eval()` 与 `1e-5` 阈值执行这两个 checks。差异接近 0 支持这两种受检扰动的隔离性质；它不证明全部可能泄漏都不存在，也不证明模型已学会任务。

若 prefix 发生变化，先检查 mask 行列是否颠倒、P 是否间接读了 S/A、是否存在全序列混合操作，以及 dropout/随机性。若改 padding 后输出变化，检查 padding mask 是否传入、bool 语义以及其他预处理是否读取了 padded values。

再打开 [Pi_mechanisms](../labs/notebooks/Pi_mechanisms.ipynb) 的多层泄漏反例，对比“直接边被禁止”与“所有间接路径被切断”。代码只在 Colab 运行；本次讲义细化没有产生新的运行结果。

**7C 完成标准：** 画出一条两层泄漏路径，写出扰动的输入、受保护输出和预期结果；清楚区分 cache 的数学前提、代码实现、实测收益三件事。

## 面试时怎样把知识串起来

**题目：为什么某些 VLA 的 action tokens 可以双向 attention，prefix 却不读 actions？**

先说生成对象：这些 action tokens 是当前已有的 noisy candidates，模型联合更新整个 chunk，因此互读不代表获取未来传感器。接着说条件依赖：prefix 只依赖固定图文，阻止它读取不断变化的 A，可使其表示在采样迭代间保持独立。最后补实现边界：这给 cache 提供条件，实际是否复用、何时失效和加速多少要查实现与测量。

若追问“有 mask 就不泄漏了吗”，给 D→P→A 的两层反例，再说明会做禁止来源扰动测试。若追问“padding 怎么办”，先区分 input-valid 与 target-valid，再算 14×7=98 的 MSE 分母。

90 秒英文练习：

> I distinguish generated future action variables from future observations that are unavailable at deployment. I then define which token groups may read each other and verify that restriction across multiple layers. A fixed prefix can support caching only when its inputs and dependencies remain unchanged, and the implementation must actually reuse the cached tensors.

记录一张 Q/K/V shape 图、一张 P/S/A 表和一段泄漏反例解释到 [学习进度](../planning/progress.md)，再进入第 8 课。无需背诵全部尺寸，但应能在给定尺寸后独立推导。

参考：[Attention 原论文](https://arxiv.org/abs/1706.03762)、[π0 v1 架构解析](pi_series/01_pi0_architecture.md)。本课矩阵与数值均为教学推演；与具体论文超参数分开记录。
