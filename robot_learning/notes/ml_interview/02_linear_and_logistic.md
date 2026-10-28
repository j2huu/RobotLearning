# M2：线性回归、逻辑回归与正则化

[目录](README.md) · [上一课](01_probability_and_objectives.md) · [下一课](03_classical_models.md) · [Colab ML01](../../labs/notebooks/ML01_linear_logistic.ipynb)

面试需要同时会解释模型假设、写梯度、处理 shape，并说清什么时候它只是 baseline。

## 1. 线性回归的 tensor 与梯度

设 X 为 [N,d]，w 为 [d,1]，y 为 [N,1]。预测 Xw+b 是 [N,1]。

$$
L=\frac1{2N}\|Xw+b-y\|^2,
\quad
\nabla_wL=\frac1N X^\top(Xw+b-y).
$$

注意 y=[N] 与 prediction=[N,1] 相减会 broadcast 成 [N,N]；程序能运行不代表目标正确。Notebook 用断言检查预测与 label 的 shape 相同。

满秩且条件数合理时可以求正规方程；实际数值计算优先使用 least-squares/SVD，而不是显式构造逆矩阵。

## 2. Ridge、Lasso 与特征缩放

$$
L_{ridge}=\frac1N\|Xw-y\|^2+\lambda\|w\|_2^2,
\quad
L_{lasso}=\frac1N\|Xw-y\|^2+\lambda\|w\|_1.
$$

Ridge 连续收缩参数，Lasso 的非光滑 L1 可以产生稀疏解。相关特征下参数解释要谨慎；预测稳定不等于系数因果可解释。

缩放会改变不同特征的 penalty 和优化几何。StandardScaler 只 fit train，validation/test 使用同一统计。改变单位可能改变正则化后的解。

截距通常不做相同 penalty。Notebook 中的手写 MSE 梯度与 sklearn Ridge 的数值 penalty 约定可能不同，比较前先核对 sum/mean。

## 3. Logistic regression 是分类概率模型

$$
z=x^\top w+b,\quad p=\sigma(z),
\quad \sigma(z)=\frac1{1+e^{-z}}.
$$

二分类 NLL：

$$
L=-\frac1N\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)].
$$

对 logits 的梯度是 $(p-y)/N$，因此：

$$
\nabla_wL=\frac1N X^\top(p-y).
$$

稳定的逐样本 loss 是 $\log(1+e^z)-yz$，实现可用 `logaddexp(0,z)-y*z` 或 `BCEWithLogitsLoss`。后者接收 raw logits，不要在输入前再 sigmoid。

## 4. Softmax、多分类与二分类接口

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},
\qquad \frac{\partial L}{\partial z_k}=p_k-y_k.
$$

PyTorch 多分类 CE 常用 logits=[B,C] 与整数 class IDs=[B]。one-hot 是推导形式，实际 API 可以接整数标签。二分类单 logit 的 shape、label dtype 要匹配 BCE 接口。

Logistic regression 的 decision boundary 在线性特征空间是超平面；增加非线性特征可以改变原输入空间中的边界。

## 5. 概率、阈值与校准不是一件事

改变 decision threshold 会改变 precision/recall，不需要重新训练模型。0.5 不是所有业务的最优阈值；类别先验、误报/漏报成本和校准情况都有关。

阈值在 validation 上选，test 上只评价固定规则。提高某一阈值下 F1 不等于概率变得更准确，也不等于 ROC-AUC 一定提高。

## 6. Colab 要观察什么

ML01 比较手写梯度与有限差分、不同 feature scale 的优化轨迹、polynomial degree 与 ridge penalty。先在 validation 上选复杂度，再只对选定模型查看 test。

练习：增加 label noise，重新观察 train/validation gap；保持数据和 seed 不变，仅改变 penalty；最后解释性能变化是否来自模型容量、优化速度或信息泄漏。

## 验收

1. 手写 linear/logistic loss 与梯度，逐项写 shape。
2. 说明为什么不能随便将 [B] 与 [B,1] 相减。
3. 解释 feature scaling 为什么对 SVM/L2 特别重要。
4. 英文：A linear model is a useful baseline because its assumptions and failure modes are easy to inspect.

参考：[CS229](https://cs229.stanford.edu/main_notes.pdf)、[sklearn linear models](https://scikit-learn.org/stable/modules/linear_model.html)。
