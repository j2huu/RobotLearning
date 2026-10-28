# M10：迁移学习、LoRA 与 ML 面试系统题

[目录](README.md) · [上一课](09_generative_models.md) · [面试题卡](interview_drill.md) · [Colab ML03](../../labs/notebooks/ML03_autograd_optimization.ipynb)

## 1. Full fine-tuning、linear probe 与 sidecar

| 方式 | 更新什么 | 需要检查什么 |
|---|---|---|
| Full fine-tuning | backbone 与任务 head 的可训练部分 | 数据覆盖、遗忘、学习率和显存 |
| Linear probe | frozen feature 上的小 head | feature 有无标签信息，是否仅线性可分 |
| LoRA | 低秩增量参数 | target layers、rank、初始化与训练梯度 |
| Sidecar | backbone 外的下游模块 | 输入接口、缓存时效、职责与控制边界 |

FP2 的 compressor/force policy 是 downstream interface；RFM task adaptation 可以另用 LoRA，但两者不是同一模块。

## 2. LoRA 的 shape 与初始化

对数学形式 W∈R^{dout×din}：

$$
W'=W+\frac\alpha r BA,
\quad A\in\mathbb R^{r\times din},\quad B\in\mathbb R^{dout\times r}.
$$

可训练增量有 r(din+dout) 个参数，少于 dout×din 需要 r 足够小。减少训练参数不等于所有 forward FLOPs 同比例减少；backbone forward 仍可能需要执行。

常见初始化 A 随机、B=0，使开始时 BA=0。第一次 backward 中 A 的梯度可能为零而 B 非零，这是乘法结构的结果，不应直接误判 A 被冻结。ML03 用一个小 linear layer 检查这个梯度现象。

## 3. Forward graph 与 backward graph 分别画

输入参与 forward 不等于其来源模型被更新。冻结 feature extractor 以后，下游 head 仍能根据该 feature 学习；但不能反向改变 extractor 的表示。

检查三件事：parameter 的 requires_grad、forward 中的 detach/no_grad、optimizer 的 parameter list。再用实际 `.grad` 验证，不能只看函数名叫 frozen。

## 4. 灾难性遗忘与迁移评价

小数据 finetune 可能改变预训练能力。是否遗忘需要保留任务/能力的独立评价，不由单个 downstream loss 判断。数据混合、参数高效适配、蒸馏与冻结可以是候选，但各有容量/优化取舍。

Distillation 用 teacher 的输出或特征提供目标；不自动保证 student 的 closed-loop behavior 等同 teacher。Alignment 也需要说清对齐对象、空间和 loss。

## 5. 系统设计题：给一个 contact-rich policy 加 force

先问/声明：输入模态和频率、数据量、动作语义、低层接口、延迟预算、是否允许改 backbone、如何评价。然后给出可检验方案。

方案 A：force-conditioned action model，需要对新模态与原生表征共同训练；方案 B：frozen RFM + downstream regulation，需要 context/feedback 与 controller contract。不能只按模块名断言哪个必胜。

对两方案控制 task data、backbone、compute、controller 与测试场景。评价包括 task success、wrench error/peaks、contact loss、latency、失败类型和 OOD。

## 6. 一段合格的英文回答结构

```text
The problem is ...
The model observes ... and predicts ...
The training target comes from ...
At inference time, ... is available, while ... is not.
I would compare ... under the same ...
The main limitation is ...
My own contribution was ...; the team result was ...
```

准备 90 秒概念、5 分钟数据流、10 分钟方案三种版本。自己的参与、合成 notebook、论文结果分开陈述。

## 验收

手算 LoRA 参数量；说明 B=0 时两因子的首步梯度；设计一个公平的 force/no-force 消融；从数据、loss、inference、controller、evaluation 解释整个系统。

参考：[LoRA original paper](https://arxiv.org/abs/2106.09685)、[FP2 tensor walkthrough](../03c_fp2_tensor_walkthrough.md)。
