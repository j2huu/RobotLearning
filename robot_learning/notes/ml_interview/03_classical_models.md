# M3：经典 ML 模型与模型选择

[目录](README.md) · [上一课](02_linear_and_logistic.md) · [下一课](04_tensors_and_backprop.md) · [Colab ML02](../../labs/notebooks/ML02_models_and_evaluation.ipynb)

面试重点不是背模型名，而是知道假设、容量、优化、推理成本、预处理和评价协议。

## 1. 六类模型对比

| 模型 | 核心机制 | 关键超参数 | 典型风险 |
|---|---|---|---|
| Logistic regression | 线性 logits + likelihood | penalty/C、特征 | 非线性关系欠拟合 |
| KNN | 邻域投票或平均 | k、距离、特征缩放 | 高维距离退化、预测时依赖训练样本 |
| Linear/kernel SVM | margin 与 hinge loss | C、kernel、gamma | 缩放敏感；kernel 数据规模成本 |
| Decision tree | 递归分割特征空间 | depth、leaf size | 深树高方差、对数据扰动敏感 |
| Random Forest | 随机化树的集成 | trees、feature subsampling、leaf size | 树之间仍有关联；外推能力有限 |
| Gradient Boosting | 逐轮拟合当前模型的损失下降方向 | learning rate、trees、depth | 轮数过多过拟合；调参必须隔离 test |

## 2. SVM：margin 与 penalty

线性 soft-margin SVM 的一种 primal 形式为：

$$
\min_{w,b}\frac12\|w\|^2+C\sum_i\max(0,1-y_i(w^\top x_i+b)).
$$

y∈{-1,+1}。更大 C 增加 margin violation 的代价，与某些 `lambda * ||w||²` 写法方向相反。Kernel 通过内积形式表达特征映射，不等于免费消除高维计算成本。

SVM decision score 不是自动校准的概率。若需要概率，额外 calibration 也必须只用合适的训练/校准数据。

## 3. 树怎样选分裂

分类树常用 Gini impurity 或 entropy；分裂使按样本数加权的子节点 impurity 下降。

$$
Gini(p)=1-\sum_k p_k^2,
\qquad H(p)=-\sum_k p_k\log p_k.
$$

叶节点最小样本数、最大深度与 pruning 控制容量。训练 impurity 降到零只说明训练拟合，不说明泛化。其中 $p_k$是节点中第 $k$ 类样本的比例, 因此，$Gini$ 越低，节点越纯净。

## 4. Bagging 与 Boosting

Bagging 对重采样训练的模型聚合；Random Forest 还在分裂时随机抽取特征以降低相关性。若各模型误差高度相关，简单增加树数不能消除共同偏差。

Gradient Boosting 按序加入弱学习器以优化整体目标。Squared loss 下拟合 residual 有直接解释；一般损失下应说拟合 negative gradient，而不是所有情况都叫“拟合 y-p”。

XGBoost 是 gradient boosting 的具体系统与目标设计，不等于所有 boosting 的统称。本课程用 sklearn 内置 boosting 做小实验，不要求额外安装 XGBoost。

## 5. 公平比较需要同一数据协议

所有候选使用同一 train/validation/test split。Scaler 放在 Pipeline 内，只 fit train；SVM/logistic 可缩放，树模型通常不依赖相同缩放。

先在 validation 比较超参数和阈值，锁定候选后评价 test。Notebook 中不会用 test 选“最佳模型”，也不会对不同模型使用不同随机数据来宣布胜负。

时间与模型规模也属于比较维度。小 tabular 数据上一个合理的树模型胜过 MLP，并不证明其能替代图像/语言 foundation model。

## 6. 接回机器人

对于低维 contact-state classifier，logistic/tree baseline 可以帮助发现数据 shortcut。相比直接训练复杂 Transformer，简单 baseline 更容易暴露“靠 episode ID 就能预测标签”等问题。

但 action prediction、closed-loop control 与静态分类是不同任务。分类 accuracy 不能当成任务成功率。

## 验收与追问

1. C 变大时，SVM 的训练 violation 与正则相对权重怎样改变？
2. 为什么 RF 需要树之间的多样性？
3. Boosting 的 residual 解释何时成立？
4. 英文：How would you compare a neural model with a classical baseline fairly?

参考：[sklearn supervised learning](https://scikit-learn.org/stable/supervised_learning.html)。
