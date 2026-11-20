# ML 面试题卡：40 题

Transformer 的更深入追问见独立 [40 道 Transformer 题卡](transformer/interview_drill.md)，配套 [8 课渐进学习路线](transformer/README.md)。本页原有 40 题编号保持不变。

[课程入口](README.md)。每题先做 60–90 秒口述，再回答一层追问。评分：0=概念错误；1=能背定义；2=有公式/shape/例子；3=能解释假设、反例和实验。记录分数与错误原因，不用阅读次数代替验收。

## 概率与线性模型（M1–M2）

1. **MLE 与 MAP 有何不同？** 要点：likelihood、prior、posterior mode；追问 Gaussian prior 对应什么 penalty，sum/mean 系数如何影响它？
2. **为什么 MSE 对应 Gaussian NLL？** 要点：固定方差、条件均值；追问预测 variance 后增加了什么项？
3. **多峰动作为什么会回归到不可行动作？** 要点：conditional mean 不等于 mode；追问增加 history 能否改变多峰性？
4. **Bias/variance 与模型输出 uncertainty 有何区别？** 要点：不同训练集的估计变化，不等同一个 Gaussian head 的方差。
5. **手写 linear regression gradient。** 要点：X=[N,d]、y=[N,1]、X.T@error/N；追问 intercept 与 penalty。
6. **推导 logistic regression gradient。** 要点：sigmoid+BCE 得 p-y；追问 stable logits loss。
7. **L1 与 L2 有何区别？** 要点：稀疏、收缩、尺度、相关特征；追问是否能据系数判断因果贡献？
8. **二分类阈值一定是 0.5 吗？** 要点：成本、prevalence、校准、validation selection；追问改阈值会不会改 AUC？

## 经典模型与评价（M3/M6）

9. **SVM 的 C 增大意味着什么？** 要点：violation penalty 增大；不要与正则 lambda 方向混淆。
10. **Random Forest 怎样降低 variance？** 要点：bagging、feature randomness、误差相关性；追问共同 bias。
11. **Bagging 与 Boosting 的区别？** 要点：聚合随机模型与逐步优化；追问一般 loss 下 residual 的含义。
12. **为什么 KNN/SVM 经常需要缩放？** 要点：距离/几何/penalty；追问树是否同样敏感？
13. **为什么 99% accuracy 可能没有价值？** 要点：class imbalance、confusion matrix、recall/precision。
14. **ROC-AUC、AP、F1 分别回答什么？** 要点：排序、PR、固定阈值；追问 AP 不等于所有 PR 积分实现。
15. **Scaler/PCA 不读 label，为什么还可能泄漏？** 要点：held-out 分布进入 fitted transform；追问 Pipeline 的 fit 范围。
16. **机器人数据应按 frame 还是 episode split？** 要点：独立单位、源 demo、object/scene/time；追问 confidence interval 的 bootstrap unit。

## 神经网络、优化与 coding（M4–M5）

17. **[B,1] prediction 减 [B] label 会怎样？** 要点：[B,B] broadcast，loss 可能仍下降。
18. **手写 stable softmax 与 CE gradient。** 要点：减最大值、log-sum-exp、p-onehot；追问 mean reduction。
19. **backward、step、zero_grad 分别做什么？** 要点：梯度计算/累积、参数更新、梯度清空。
20. **eval、no_grad、detach 分别改变什么？** 要点：层行为、图记录、图连接；追问 frozen backbone 的 downstream gradients。
21. **Adam 为什么有 bias correction？** 要点：初始零矩估计的偏差；追问 epsilon 与二阶矩的作用。
22. **AdamW 为什么不等同 Adam+L2？** 要点：decay 与 adaptive update 解耦；追问 SGD 下的特殊等价。
23. **BatchNorm 与 LayerNorm 统计哪些轴？** 要点：batch/channel 与 token features；追问 eval behavior 与小 batch。
24. **loss 不下降怎么查？** 要点：shape、labels、finite、grad、parameter diff、tiny-batch overfit，先小实验再加模型。

## 表示与 Transformer（M7–M8）

25. **PCA 最大保留方差是否等于最好分类表示？** 要点：低方差信号反例；追问 fit train only。
26. **K-means cluster IDs 可以直接算 accuracy 吗？** 要点：label permutation；追问 ARI 与使用标签的范围。
27. **Reconstruction 与 contrastive objective 的区别？** 要点：重建与 invariance/positive pairing；追问 shortcut、false negatives。
28. **Self-attention 与 cross-attention 怎么区分？** 要点：Q 与 K/V 来源，不是尺寸。
29. **Nq=1、Nk=N 时 attention 的 shape？** 要点：[B,h,1,N]→[B,1,d]；追问 softmax 的轴。
30. **FP2 compressor 的 query 是 force 吗？** 要点：learned parameter，C 是 K/V，force 在下游；追问 reconstruction decoder 何时丢弃。
31. **Mask 看起来正确还会泄漏吗？** 要点：多层间接路径、padding、API bool convention；追问 perturbation test。
32. **Cached context 与 KV cache 是同一件事吗？** 要点：缓存对象与时效不同；追问 cache 是否提供新物理观测。

## 生成、迁移与系统（M9–M10）

33. **CVAE 训练与推理可见信息有何不同？** 要点：posterior 读 action，prior sampling 不读未来 expert action。
34. **Diffusion 为什么训练时可随机采一个 timestep？** 要点：期望目标的 Monte Carlo estimate；追问 sampler 是否仍多步。
35. **推导线性 flow path 的 velocity。** 要点：a-epsilon；追问反转时间要怎样同时修改 dt 与 target？
36. **Denoising steps 增加是否提高机器人 feedback rate？** 要点：生成时间与物理时间不同。
37. **LoRA 参数量怎么算？** 要点：r(din+dout)、scaling；追问训练显存和 forward cost 是否同倍数降低？
38. **LoRA B=0 时首步 A 梯度为零是 bug 吗？** 要点：乘法 chain rule；追问后续 step。
39. **给 frozen RFM 加 force，你怎么选接口和评价？** 要点：observation/action/frame/time contract、matched ablation、成功/force/latency。
40. **如何说明自己的 FP2 项目贡献？** 要点：具体实现/分析/实验，团队结果、论文结论和 toy notebook 分开；用数据流支撑描述。

## 限时模拟

- 10 分钟基础：随机抽概率、模型、优化、评价各一题。
- 15 分钟 coding：NumPy logistic gradient 或 PyTorch masked loss，说明 shape 与 edge cases。
- 15 分钟系统：画 FP2 的 forward/backward graph，设计 force/no-force 与 stale-context 消融。
- 5 分钟英文：从问题、输入、目标、推理、实验、边界六点口述。

Notebook 输出只在本人运行后填写。不要预先写“全部通过”或把机制实验包装成机器人性能结果。
