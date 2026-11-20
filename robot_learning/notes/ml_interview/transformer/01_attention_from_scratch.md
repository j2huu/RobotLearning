# T1：从加权平均到 Multi-Head Attention

[目录](README.md) · [下一课：完整网络](02_blocks_norm_and_architectures.md)

本课问题：一个 token 如何根据自己的需求，从其他 tokens 中读取内容？先不讨论整套语言模型。

## T1A：从“所有内容平均”到“按问题读取”

给三个历史观测的标量特征 `[2, 4, 12]`，平均得到 6。平均池化把所有来源同等对待；但当前动作若关心最近接触，可能需要更偏向某一个来源。Attention 先计算匹配分数，再用归一化权重聚合 values。

对 query i：

$$
s_{ij}=q_i^Tk_j/\sqrt{d_h},\qquad
\alpha_{ij}=\frac{e^{s_{ij}}}{\sum_{m}e^{s_{im}}},\qquad
y_i=\sum_j\alpha_{ij}v_j.
$$

Q 是读取位置的特征，K 是用于匹配的特征，V 是被取走的内容。“问题、索引、内容”是类比，不表示每一维都有人工语义。Q/K/V 通常由不同线性层学习得到。

为什么 K 与 V 分开？匹配所需特征和输出所需特征可以不同。比如用“颜色”匹配目标，读取“空间位置”；若强制 K=V，会减少可学习的选择。模型也可以采用共享设计，但这不是一般定义。

Self-attention：Q/K/V 来自同一 hidden 序列。Cross-attention：Q 来自一个序列、K/V 来自另一个序列。将图像和文字 concat 后对整段做 attention，仍可属于 self-attention；是否多模态与是否 cross 是两回事。

**暂停思考：** 输入长度 100，query 长度 4，输出有多少个 token？答案是 4，每个 query 产生一个读取结果，而不是 100。

## T1B：手算、缩放和梯度

### 一个可以逐步验证的例子

取一行已经缩放过的 scores `[0, ln(3)]`，values `[2, 10]`。

1. 指数为 `[1, 3]`，权重 `[1/4, 3/4]`。
2. 输出 $y=0.25\times2+0.75\times10=8$。
3. 只把第二个 value 改成 100，权重不变，输出为 75.5。
4. 若禁止读取第二个 key，则 logits 为 `[0,-∞]`，输出为 2。

第 3 步刻意固定 Q/K，只改 V。在真正 self-attention 中改变输入 token 可能同时改变 Q/K/V，不能自动沿用“权重不变”的结论。

### 为什么除以根号维度

假设每个 $q_r,k_r$ 独立、零均值、单位方差，且不同分量乘积互不相关：

$$
\operatorname{Var}(q^Tk)=\operatorname{Var}\left(\sum_{r=1}^{d_h}q_rk_r\right)=d_h.
$$

除以 $\sqrt{d_h}$ 后方差约为 1。分数尺度太大时 softmax 容易接近 one-hot，使许多方向梯度变小。这是初始化尺度的解释，不是训练后 Q/K 必然独立或方差固定的保证。使用 $d_h$ 而不是模型总宽度 d，因为每个 head 独立计算内积。

### Attention 怎样反向传播

令单 query 的输出 y 为向量，$g=\partial\mathcal L/\partial y$，则：

$$
\frac{\partial\mathcal L}{\partial v_j}=\alpha_jg,\qquad
\frac{\partial\mathcal L}{\partial s_j}=\alpha_j g^T(v_j-y).
$$

第二个式子来自 softmax Jacobian。它说明分数梯度取决于“这个 value 相对当前混合结果的差异”。在 y=8 的标量例子里，若 $g=1$，两个 score 的梯度是 `[-1.5, 1.5]`，和为零；softmax 对所有 logits 同加一个常数不变。

继续链式法则：$\partial\mathcal L/\partial q=\sum_j(\partial\mathcal L/\partial s_j)k_j/\sqrt{d_h}$。固定禁止边不参与读取；正确屏蔽后，该边 score 不应传递梯度。所有 keys 被屏蔽时，这个归一化本身无定义，不能当作普通情况。

## T1C：多头、张量与参数

教学配置 $B=2,N=5,d=12,H=3,d_h=4$：

```text
X                                      [2,5,12]
Q=XWq, K=XWk, V=XWv                    [2,5,12]
reshape                                [2,5,3,4]
transpose token/head                   [2,3,5,4]
Q @ K.transpose(-1,-2)                  [2,3,5,5]
softmax(scores + mask, dim=-1)          [2,3,5,5]
weights @ V                            [2,3,5,4]
transpose + merge heads                [2,5,12]
output projection W_O                  [2,5,12]
```

多头不等于复制 H 份完整 d 宽 attention。标准等宽实现把总投影输出拆成 H 份，每份宽度 $d/H$，让不同子空间有各自的匹配权重。它提供表达机会，不保证自动分工或 head 数越多越好。固定 d 时增加 H 也会缩小每头容量。

忽略 bias，标准 MHA 的 Q/K/V/O 共 $4d^2$ 个参数；本例为 $4\times12^2=576$。Scores 元素数为 $BHN^2=150$。参数数目不随 N 增长，但 activation 和计算会增长。

输出投影把各 head 聚合结果混合回模型宽度。只 concat 不做投影是另一种设计，不要遗漏标准结构的 $W_O$。

**最小代码路线：** [T8 的参考实现](08_coding_debugging_and_robotics.md)；数字例子见 [TR01](notebooks/TR01_attention_masks_and_cache.ipynb)。

## 自测与答案

1. Softmax 在哪一轴？**Key 轴**；每个 query 对合法来源单独归一化。
2. Softmax 后再将禁止项乘零，等价吗？**通常不等价**，剩余权重之和小于 1；还要重新归一化且妥善处理空行。
3. 权重很大就证明这个 token 对预测因果重要吗？**不能**；还存在 V、输出投影、残差和其他层。需要控制变量或干预实验。
4. $N_q=3,N_k=7,H=2$ 时 scores 最后两轴？**[3,7]**；输出 query 轴为 3。

资料：[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)。本文小数字、梯度练习与教学配置为独立推导。
