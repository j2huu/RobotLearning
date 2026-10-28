# 第 12 课：FAST、π0-FAST 与 π0.5 的训练配方

[上一课](11_pi0_policy_system.md) · [下一课：动作接口](13_action_and_control_interfaces.md) · 深读：[P3 FAST](pi_series/03_fast_action_representation.md)、[P4 π0.5](pi_series/04_pi05_generalization.md)

目标：说清表示、policy、训练配方与部署系统是不同对象，不能把几个名字混成“更快的 π0”。

按 [循序路线](../planning/vla_learning_path.md) 分三次：**12A读§1–2做codec数据流；12B读§3列AR训练/推理信息；12C读§4–7和P4填π0.5的数据/目标/部署表**。前置是第7课mask与第9课flow。最后做 [第12课面试卡](vla_interview_workbook.md)，用一个数值反例区分表示误差与policy预测误差。

## 1. 四个名字

| 名称 | 对象 | 动作机制 |
|---|---|---|
| π0 | 连续动作 VLA | flow action expert |
| FAST | action tokenizer | normalized chunk→DCT→量化→序列化→BPE |
| π0-FAST | 使用 FAST 的 AR policy | 预测离散 tokens，再解码动作 |
| π0.5 v1 | 异构数据 co-training 与分层推理系统 | 训练含离散/连续动作目标，最终低层使用 flow |

FAST 不是减少 Euler steps 的 sampler；π0.5 最终低层推理也不是先生成完整 FAST chunk 再逐个交给 flow 精修的串联流程。

## 2. Tokenizer 与 policy 分开评价

$$
A\xrightarrow{T}z_{1:L}\xrightarrow{T^{-1}}\tilde A.
$$

先测正确 tokens 的 codec round trip，再测 policy 预测错误，最后测真实执行。DCT 在保留全部精确系数时可逆，量化引入有损误差；不能把“压缩”直接等同于删掉所有高频信息。

接触阶段、夹爪切换与分维最大误差可能比全程平均 MSE 更有意义。Token 长度 L 不是物理 chunk 长度 H，二者的映射由 tokenizer 定义。

## 3. AR 训练与推理

$$
L_{CE}=-\sum_i m_i\log p_\theta(z_i\mid z_{<i},c).
$$

Teacher forcing 让训练并行计算各位置 logits；部署时生成前缀来自模型，通常顺序解码。检查 label shift、prompt loss mask、padding mask 和 indirect target leakage。

## 4. π0.5 的 co-training 不只是拼更多数据

P4 按本地 v1 区分 heterogeneous robot data、语义/子任务监督及不同训练阶段。每一类样本有自己的 observation、target、loss 与 sampling weight，不要求每条样本都有完整连续机器人 action。

语义任务改善表示，不自动提供精确 wrench control。跨 embodiment 的数据也不免除 native action conversion。

## 5. 多目标 attention 的隔离

如果模型同时处理 clean discrete action targets 与 noisy continuous actions，需要阻止连续分支读取答案。只屏蔽直接边不够，还需检查 prefix 的多层间接路径。

配套 [Pi 机制 notebook](../labs/notebooks/Pi_mechanisms.ipynb) 给出两层反例。它不是 π0.5 全模型训练，也没有验证所有 loss 的梯度路由。

## 6. 分层推理

高层 subtask 决策回答接下来做什么，低层 policy 回答给定当前 subtask 怎样产生动作。高层更新、低层 chunk inference、action execution 与 contact regulation 的频率需分别记录。

若高层选择错误目标，单纯增加低层力控无法可靠补救。若目标与运动大体正确但失去接触，第14课 FP2 式下游调节才是相应的职责。

## 7. 本课任务与验收

画出 tokenizer、AR policy 和 flow policy 三条不同路径；填写 π0.5 v1 的数据/阶段/目标表；对同一动作表示区分 codec error、policy error 与系统执行误差。

本课不增加伪 FAST tokenizer 实验：简单 DCT/rounding 不等于完整 FAST+BPE。当前 notebook 仅覆盖生成机制与 mask；完整 tokenizer 实现应固定官方代码与词表版本。

参考：[FAST 原论文](https://arxiv.org/abs/2501.09747)、[P3](pi_series/03_fast_action_representation.md)、[P4](pi_series/04_pi05_generalization.md)。
