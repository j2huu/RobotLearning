# M6：数据泄漏、评价指标与实验设计

[目录](README.md) · [上一课](05_optimization_and_training.md) · [下一课](07_representation_learning.md) · [ML02](../../labs/notebooks/ML02_models_and_evaluation.ipynb) · [Lab02](../../labs/notebooks/Lab02_episode_windows.ipynb)

## 1. Train、validation、test 的职责

Train 更新参数与预处理统计；validation 选模型、超参数、epoch、threshold；test 在方案固定后做最终评价。反复看 test 再改模型等价于把它用于模型选择。

Scaler、PCA、feature selection、imputation 都属于需要 fit 的变换，不能因为“没用标签”就 fit 全部数据。Pipeline 可以确保训练时 fit、评价时只 transform。

同一 source episode 的相邻窗口、增强版本、重采样数据应隔离。需要测试新 subject/object/scene 时，split unit 应匹配目标。时间序列的 split 还要避免未来样本泄漏。

## 2. Classification metrics

$$
Precision=\frac{TP}{TP+FP},\quad
Recall=\frac{TP}{TP+FN},\quad
F_1=\frac{2PR}{P+R}.
$$

若 99% 都是负类，全预测负类有 99% accuracy，但正类 recall 为零。报告 confusion matrix、正类比例、阈值与任务成本。

ROC 使用 TPR/FPR；PR 使用 precision/recall，基线与 prevalence 有关。极不平衡时不能只报 ROC-AUC。Average Precision 与梯形积分得到的 PR-AUC 不是所有情况下完全相同；本 notebook 明确用 `average_precision_score`。

Micro/macro/weighted averaging 回答不同问题。Macro 给每类相同权重，micro 聚合样本决策，weighted 按 support 加权；不要只说“F1”而省略口径。

## 3. Calibration 与 ranking

概率校准问“预测 0.8 的样本中是否约 80% 为正”，AUC 主要评价排序。排序不变的概率变换可能改变 calibration，却不改变 ROC-AUC。

$$
Brier=\frac1N\sum_i(p_i-y_i)^2.
$$

Reliability diagram 按预测概率分箱，比较平均预测概率与实际正类比例。ECE 依赖分箱与样本量，不应当作完整概率质量评价。温度缩放等校准方法也需要与最终 test 隔离的校准数据。

## 4. Regression、机器人与不确定性

MSE 对大误差更敏感，MAE 更接近绝对偏差；NLL 同时依赖分布位置、形状与尺度。不同单位、不同归一化下的 loss 不能直接比较。

机器人还要看任务成功、力峰值/误差、失去接触、延迟和失效类型。成功率的独立单位一般是 rollout，而不是其内部每一帧。

例如 8/10 成功不能仅报告“80%”而不写 n=10。Bootstrap 应按实验独立单位重采样；按 frame bootstrap 可能给出过窄区间。重复 seed、paired scenarios 与 CI 各控制不同不确定性。

## 5. 消融与分布变化

加入 force 后性能提高，需要控制 data、backbone、参数/compute、controller、训练预算与场景。必要时做 force shuffle、zero force、sensor delay 和 contact/free-space 分段。

输入 shuffle 是诊断，可能同时制造 OOD，不是对模态因果贡献的完整证明。报告 intervention 的具体方式和影响范围。

OOD 可以是新视觉、对象、物理参数、任务或 embodiment；一种泛化结果不自动覆盖其他类型。验证集更难时，validation loss 高于 train 不一定只是过拟合。

## 6. Notebook 的三项检查

ML02 在 train 上 fit 各候选，用 validation AP 选模型，再用 validation F1 选 threshold，最后只评价选定模型的 test。另造 group-ID 记忆反例，用 frame split 与 group split 对比，只说明泄漏风险，不声称固定数据下必有某个数值差。

Lab02 检查 causal timestamp、episode windows、padding mask 与 train-only statistics。没有真实图像或机器人控制。

## 验收

解释 AP 与 accuracy 的区别；说出 threshold selection 所用 partition；给出两种无标签 preprocessing 泄漏；设计一个按 rollout 隔离的 force 消融。

参考：[sklearn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)、[model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)。
