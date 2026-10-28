# M5：优化、初始化、归一化与训练故障

[目录](README.md) · [上一课](04_tensors_and_backprop.md) · [下一课](06_evaluation_and_leakage.md) · [Colab ML03](../../labs/notebooks/ML03_autograd_optimization.ipynb)

训练 loss 不下降时，先提出可验证假设，不先换一个更大的模型。

## 1. SGD、momentum、Adam 和 AdamW

SGD 用当前 batch 梯度：

$$
\theta_{t+1}=\theta_t-\eta g_t.
$$

Momentum 聚合历史方向。Adam 分别维护梯度的一阶/二阶指数滑动平均，并做 bias correction：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

AdamW 将 weight decay 与自适应梯度更新解耦。SGD 下某种 L2 与 decay 写法可以等价，但不能把这个结论直接推广到 Adam。Bias/norm 参数是否 decay 是训练配方选择，不是普遍定理。

## 2. Learning rate 与 batch size

过大学习率可能震荡或发散，过小可能训练很慢。Warmup 在早期逐渐提高 LR，cosine decay 等 schedule 在后期降低 LR；它们不能修复标签泄漏或错误 loss。

比较不同 batch size 时需要控制训练数据量/optimizer steps/compute 的口径。梯度累积模拟的是更大有效 batch 的梯度聚合，但 BatchNorm 统计和 Dropout 随机性不一定与一次大 batch 相同。

## 3. 初始化与梯度传播

多层线性变换与激活可能让方差逐层放大/衰减。Xavier/He 初始化针对特定激活与独立性近似保持信号尺度；不是对任意网络和数据的保证。

Sigmoid/tanh 饱和会使局部梯度小；ReLU 负区间梯度为零，也可能有 dead units。Residual connections 提供较直接的梯度路径，但不使网络自动免于优化问题。

## 4. BatchNorm 与 LayerNorm

二者都先按选定轴算统计，再归一化与 affine transform；区别在统计轴及 train/eval 行为。

| 层 | 常见统计轴 | train/eval 关注点 |
|---|---|---|
| BatchNorm | batch 与可选空间轴，每个 channel | 通常训练更新 running stats，eval 使用它们 |
| LayerNorm | 每个样本/token 的指定 feature 轴 | 通常不使用跨 batch running stats |

BatchNorm 对 batch 大小与分布敏感；LayerNorm 不意味着不同 token 的信息永远隔离，attention 仍可混合 token。

Dropout 训练时随机置零并按保留率缩放；eval 时通常关闭。把模型留在 train mode 做部署评估可能引入随机输出。

## 5. 训练不稳定的排查表

| 观察 | 候选原因 | 最小检查 |
|---|---|---|
| loss NaN/Inf | log(0)、exp overflow、尺度过大、错误 mask | 打印首次非有限中间值，使用 logits loss |
| 参数没更新 | 没有 grad、未 step、冻结或 optimizer 漏参 | 检查 grad 与参数差 |
| train 好、validation 差 | 过拟合、分布差异、切分或预处理不一致 | 同一 Pipeline、分组切分、学习曲线 |
| train/validation 都差 | 欠拟合、优化失败、标签错配 | tiny-batch overfit、简化模型 |
| 指标突然异常高 | ID/未来信号泄漏、重复样本 | 去除可疑字段、按源 episode 隔离 |

Gradient clipping 限制梯度范数，不是概率模型的正则项；mixed precision 的 loss scaling 和数值范围又是另一层机制。本 notebook 不要求 AMP/GPU。

## 6. FP2 的 freeze 与 LoRA

阶段 2 只训练 compressor/decoder，阶段 3 只训练 force policy。要分别画 forward graph、backward graph 和 optimizer parameter list。模型名称里叫 adaptor，不能据此断言它就是 LoRA。

## 验收

解释 Adam bias correction；解释 AdamW 不等于随便加 L2；说清 BN/LN 的统计轴；给一个 train/eval 错误反例。Notebook 对 SGD/AdamW 使用同一初始权重和训练数据，比较目标是机制，不宣称某 optimizer 对所有任务最优。

参考：[PyTorch Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。
