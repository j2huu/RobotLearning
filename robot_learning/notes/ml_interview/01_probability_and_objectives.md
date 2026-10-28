# M1：概率、统计与机器学习目标

[专题目录](README.md) · [下一课](02_linear_and_logistic.md) · 实验：[ML01](../../labs/notebooks/ML01_linear_logistic.ipynb)、[BC](../../labs/notebooks/Lab01_probabilistic_bc.ipynb)

先回答：模型在最小化什么？MSE 为什么对应 Gaussian？一个模型很自信，是否意味着它是正确的？

## 1. 随机变量、条件分布与 Bayes

随机变量描述不确定量；一次观测是它的 realization。机器人里 state、observation、action 是不同变量，不能因为 tensor 都有七列就混为一谈。

$$
p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}.
$$

prior 是观察 x 之前对 y 的分布，likelihood 是固定参数/假设下数据的概率，posterior 是条件化后的分布。Likelihood 是参数的函数，不要求对参数积分为 1。

独立不等于不相关；独立通常推出零协方差，逆命题一般不成立。条件独立也不等于边缘独立。相邻 episode frames 相关，不能把 frame 数直接当独立试验数。

## 2. 期望、方差与经验风险

$$
\mathbb E[Y]=\int y p(y)dy,
\qquad
\operatorname{Var}(Y)=\mathbb E[(Y-\mathbb E Y)^2].
$$

总体风险依赖未知的真实分布；训练时最小化有限样本上的经验风险：

$$
R(\theta)=\mathbb E_{(x,y)\sim p_*}[\ell(f_\theta(x),y)],
\quad
\hat R(\theta)=\frac1n\sum_i\ell(f_\theta(x_i),y_i).
$$

降低经验风险不自动降低新分布风险。训练数据偏差、有限样本与模型选择都会影响泛化。

## 3. MLE、MAP 和正则化

独立样本假设下，MLE 最大化似然的乘积，等价于最小化 NLL 的和：

$$
\theta_{\mathrm{MLE}}=\arg\min_\theta -\sum_i\log p_\theta(y_i\mid x_i).
$$

MAP 增加参数先验：

$$
\theta_{\mathrm{MAP}}=\arg\min_\theta
-\sum_i\log p_\theta(y_i\mid x_i)-\log p(\theta).
$$

Gaussian prior 对应 L2 penalty，Laplace prior 对应 L1 penalty，但系数还取决于 loss 的 sum/mean 约定和噪声尺度。MAP 给出 posterior mode；不等于进行完整 Bayesian posterior inference。

## 4. MSE、Gaussian NLL 与 CE

若 $y\mid x\sim\mathcal N(\mu_\theta(x),\sigma^2I)$ 且方差固定：

$$
-\log p_\theta(y\mid x)
=\frac{\|y-\mu_\theta(x)\|^2}{2\sigma^2}+\text{constant}.
$$

因此优化 mean 等价于优化 MSE。若模型还预测方差，就必须保留 log-variance 项；只放大方差并不能无限改善正确的 NLL。

对 one-hot 分类目标，categorical NLL 是 cross-entropy：

$$
\ell=-\log p_\theta(y\mid x)
=-z_y+\log\sum_j\exp(z_j).
$$

实现用 log-sum-exp 或框架的 CE，避免先算极端 softmax 再取 log。

## 5. 为什么回归到均值会失败

平方误差下的最优预测是 $\mathbb E[Y\mid X=x]$。若同一 observation 对应左右绕行两种动作 $-1,+1$，均值 0 可能既不属于左模式，也不属于右模式。

这个例子说明 observation 或分布表达可能不足，不能据此断言任何多峰问题都必须使用 diffusion。增加可靠 history、任务条件，或使用 mixture，都是可能选择。

## 6. Bias/variance 与不确定性

在平方损失、固定 x 和相应独立噪声假设下，预测误差可分成 bias²、variance 与不可约噪声。这里 variance 指对不同训练集重新训练时的预测变化，不是某个模型输出的 Gaussian variance。

Aleatoric uncertainty 来自给定观测下的随机性或缺失信息；epistemic uncertainty 与有限知识/模型有关。ensemble disagreement 可提供线索，但不能保证可靠检测全部 OOD。

置信区间描述估计过程的频率学覆盖性质；不是直接声称固定参数有 95% 概率落在已算出的区间内。少量成功 rollout 的成功率需要报告试验数和区间。

## 7. 接回 FP2

compressor 的 reconstruction loss 是自监督目标；force policy 的 frame/mask/wrench targets 是另一种监督。不同任务的 loss 数值大小不能直接比较“哪个模型更聪明”。

## 验收与追问

1. 从 Gaussian density 手推 MSE，并列出固定方差假设。
2. 写出 MLE 与 MAP 的差别，解释 L2 系数与样本 mean 的关系。
3. 区分输出方差、估计方差和 epistemic uncertainty。
4. 英文口述：Why can a low MSE policy fail on multimodal actions?

参考：[CS229 notes](https://cs229.stanford.edu/main_notes.pdf)、[PML](https://probml.github.io/pml-book/)。
