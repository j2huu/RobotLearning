# M7：PCA、K-means、自编码器与对比学习

[目录](README.md) · [上一课](06_evaluation_and_leakage.md) · [下一课](08_attention_and_compression.md) · [Colab ML04](../../labs/notebooks/ML04_pca_kmeans.ipynb)

问题：压缩一定保留任务信息吗？聚类的 cluster ID 等于语义标签吗？

## 1. PCA 的两个视角

对 train-centered $X∈R^{N×d}$，取 top-k 右奇异向量 $V_k$：

$$
Z=XV_k,\quad \hat X=ZV_k^\top,
\quad X=U\Sigma V^\top.
$$

PCA 在正交线性子空间约束下最大化保留方差，也最小化 squared reconstruction error。低方差方向仍可能含分类或力控关键信息；最大方差不是最大任务相关性。

Centering 不等于 standardization。不同单位下 PCA 结果会变；是否按 feature 标准化取决于数据语义。只能 fit train mean/std/components，再 transform held-out。

## 2. K-means 优化什么

$$
\min_{\mu_1,\ldots,\mu_K}\sum_i\min_k\|x_i-\mu_k\|^2.
$$

交替执行最近中心分配与中心更新。目标非凸，初始化会影响局部解；预先指定 K，受尺度、球状簇假设和异常点影响。

Cluster 0/1/2 没有固定语义。不能直接比较 cluster IDs 与 class labels 的逐元素 accuracy。Notebook 使用 adjusted Rand index；用标签评价聚类只在评价阶段，不把标签输入 K-means。

## 3. Autoencoder 与瓶颈

$$
z=f_\theta(x),\quad \hat x=g_\phi(z),\quad
L=\mathbb E\|x-\hat x\|^2.
$$

Linear AE 与 PCA 在特定损失、容量和最优条件下有子空间联系，但不能把任意深层 AE 等同 PCA。过强 decoder、旁路或错误目标可能让 latent 对下游无用。

FP2 把 RFM tokens 压缩到一个 token，再重建原 tokens。它重建的是 hidden features，而不是输入 RGB；单 token 仍含 d 个数，不是单个标量。

## 4. Contrastive learning

InfoNCE 的一个常见形式：

$$
L_i=-\log\frac{\exp(\operatorname{sim}(z_i,z_i^+)/\tau)}{\sum_j\exp(\operatorname{sim}(z_i,z_j)/\tau)}.
$$

正样本定义了希望保留的 invariance；负样本、temperature 与 batch composition 影响学习信号。不能仅靠“不同模态”就保证正样本语义一致，时间错位可能破坏配对。

机器人多模态中，force 与 vision 的正对必须有清楚的同步/可用性语义。模型也可能利用背景、传感器 bias 或 episode identity，形成 shortcut。

## 5. 学表示与学 policy 的区别

Reconstruction/contrastive loss 改善 latent，不自动产生 action 或 force-control 能力。要通过下游线性 probe、冻结表示评价、模态消融与任务指标检查。

Notebook 对比 raw/scaled PCA 的方差曲线与 K-means clustering；另加一个低方差标签方向的反例。后者说明“重建好不保证分类好”，不表示真实 FP2 一定丢失力觉信息。

## 验收

从 SVD 写 PCA projection/reconstruction；解释 PCA 为什么受单位影响；解释 cluster IDs 的置换；列出一种错误正对和一种 shortcut。

参考：[sklearn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)、[clustering](https://scikit-learn.org/stable/modules/clustering.html)。
