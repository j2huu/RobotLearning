# T5：训练目标、Label Shift 与生成

[目录](README.md) · [上一课：Mask](04_masks_and_information_flow.md) · [下一课：KV cache](06_kv_cache_and_decoding.md)

## T5A：为什么训练可以并行，生成却通常逐步

自回归模型分解：

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^T p(x_t\mid x_{<t}),\qquad
\mathcal L=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

训练时整个真实序列已知，能一次构造所有前缀，并用 causal mask 限制每个位置只看自己的历史。所有 token 位置可在一层中并行计算，但不同网络层仍顺序执行。推理时新的 token 尚未产生，标准自回归采样需要先确定前一步输出，再把它作为后一步输入。

用 `[BOS, 我, 爱, 猫, EOS]` 对齐：

| 位置 | 输入 token | 该位置可见输入 | 该位置 logits 的 label |
|---|---|---|---|
| 0 | BOS | BOS | 我 |
| 1 | 我 | BOS, 我 | 爱 |
| 2 | 爱 | BOS, 我, 爱 | 猫 |
| 3 | 猫 | BOS, 我, 爱, 猫 | EOS |

自己实现时可写 `inputs=ids[:,:-1]`、`targets=ids[:,1:]`。某些模型封装在内部 shift；外面再 shift 会错两格。不要只看 loss 数值，要逐行打印 input token、target token 和可见范围。

Teacher forcing 使用真实历史，而非模型刚生成的历史。它支持高效训练，但训练和推理的前缀分布可能不同，错误可以积累。不能因此说 teacher forcing 就是泄漏：只要真实历史在目标条件中合法，训练目标是成立的。

## T5B：从 logits 到 loss

hidden `[B,N,d]` 经词表投影得到 `[B,N,V]`，对 label 所属类别计算 CE：

$$
\ell_i=-z_{i,y_i}+\log\sum_{v=1}^{V}e^{z_{i,v}}.
$$

稳定计算减去最大 logit。框架的 cross entropy 通常接收原始 logits，不要先 softmax 再当作 logits 输入。

**手算：** 正确类别概率分别 0.5 和 0.25，另外两个位置为 padding。有效 token 平均 NLL 为 $(\log2+\log4)/2\approx1.0397$；困惑度 $\exp(NLL)\approx2.8284$。若错误除以总长度 4，会得到虚低 loss。不同 tokenizer、数据和掩码口径下的困惑度不能直接横比。

```python
# 自己实现 shift 的教学路线；不要与内部已 shift 的封装混用。
inputs = ids[:, :-1]
targets = ids[:, 1:]
target_valid = valid[:, 1:]
# logits = model(inputs, input_valid=valid[:, :-1])
# per_token = cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none')
# 对 valid targets 取均值；若有效个数为0，跳过或明确报错。
```

只训练回答时，`target_valid` 还需与回答区间相交；loss 区间应根据 **shift 后目标的位置** 定义。Loss mask 不应机械复制输入 mask。

### 三种目标不要混用

| 目标 | 输入条件 | 被预测对象 | 信息限制 |
|---|---|---|---|
| Causal LM | 目标前缀 | 下一个离散 token | 不能读待预测 token 或未来答案 |
| Masked LM | 被破坏/遮盖的完整上下文 | 被选位置原内容 | 可以双向，但必须按目标设计破坏输入 |
| Denoising/flow | 带噪候选、条件、噪声时间 | 噪声/速度/干净变量等 | 可联合读候选块，不能偷读部署时不可得真值 |

Attention 模块不决定 loss；同样 Transformer 可以承载不同生成目标。训练时允许读取哪些变量，应从条件概率和部署输入推导。

## T5C：生成策略与训练诊断

Temperature：$p_i\propto\exp(z_i/\tau)$，$\tau>0$。固定 logits 时小温度分布更尖，大温度更平；greedy 直接取最大值。Top-k 保留前 k 个候选，top-p 保留累计概率达到阈值的候选集合并重新归一化。它们改变采样策略，不能修复错误 shift 或 mask。

Beam search 保留多个候选前缀，按累计分数扩展；长度偏差和停止规则会影响结果。更高 likelihood 不保证更真实、更合任务，更不保证机器人动作安全。

训练时 dropout 等正则与推理不同。`eval()` 控制模块行为，`no_grad()` 控制梯度记录；二者不是替代关系。直接调用 SDPA 时要按训练状态显式设置 `dropout_p`，不能假设函数会读取外部模块的 eval 状态。[SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

| 症状 | 优先检查 | 最小诊断 |
|---|---|---|
| loss 异常接近零，生成很差 | input/label 未 shift、答案泄漏 | 打印对齐表；扰动未来答案 |
| 一直生成 PAD/EOS | label mask、数据比例、停止逻辑 | 看有效目标频率和每位置 loss |
| batch padding 改变预测 | key mask、position IDs、取 logits 的位置 | 单条与 padded batch 对比 |
| train 好而 rollout 差 | exposure bias、分布漂移、接口错误 | 对比真历史和模型历史 |
| loss NaN | 空 attention 行、无有效 loss、溢出 | 先检查有限性与 mask，再调学习率 |

## 自测与答案

1. Teacher forcing 为何不必逐 token forward？**历史都已知，mask 保证并行位置的合法依赖**。
2. 去掉 causal mask，训练 CE 下降更快算改进吗？**不能**，可能读到了答案；须匹配推理条件。
3. 将 prompt loss 设为 0 会阻断回答梯度进入 prompt 表示吗？**不会**，回答仍可通过 attention 对它反传；冻结/detach 是另一回事。
4. 只取 `logits[:,-1]` 总正确吗？**不一定**，右 padding 时最后列可能无效，需按模型输入布局取最后有效位置。
