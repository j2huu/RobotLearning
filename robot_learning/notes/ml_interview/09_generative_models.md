# M9：从条件分布到 CVAE、Diffusion 与 Flow

[目录](README.md) · [上一课](08_attention_and_compression.md) · [下一课](10_transfer_and_system_design.md) · [Colab ML06](../../labs/notebooks/ML06_diffusion_flow.ipynb)

## 1. 先决定要表达什么分布

Deterministic MSE 输出条件均值；单 Gaussian 表达单峰位置/尺度；mixture 可以显式表达多个分量。多峰问题未必需要最复杂的生成模型，应先看 observation 是否缺失、数据是否覆盖模式。

生成模型的 training loss、采样质量、模式覆盖、推理耗时与控制稳定性是不同评价维度。不同目标的 loss 数值不能直接横比。

## 2. CVAE：训练有 posterior，推理没有未来 action

$$
\log p_\theta(a\mid o)\ge
\mathbb E_{q_\phi(z\mid o,a)}[\log p_\theta(a\mid o,z)]
-D_{KL}(q_\phi(z\mid o,a)\|p(z\mid o)).
$$

训练时 encoder 可读 expert action 形成 posterior；推理时未来 expert action 不存在，从 prior 取 z。Gaussian reparameterization 是 $z=\mu+\sigma\odot\epsilon$，让可微参数与随机噪声分开。

Posterior collapse 指 decoder 忽视 latent 等现象；不是 KL 数值小就能直接下结论。ACT 用 latent 与 chunking 的具体方式回看第 2 课/Pi 专题。

## 3. Diffusion：噪声预测是监督目标之一

用 data action a₀、扩散步 k 表示：

$$
a_k=\sqrt{\bar\alpha_k}a_0+\sqrt{1-\bar\alpha_k}\epsilon,
\qquad L=\mathbb E\|\epsilon_\theta(a_k,o,k)-\epsilon\|^2.
$$

训练随机抽 k，不需要每次展开完整去噪链；采样从噪声开始沿 reverse process 多步更新。预测 epsilon、x₀ 或 velocity 的参数化与采样器必须匹配，不能只改 target 不改 sampler。

Notebook 实现小型 DDPM，固定 schedule 与完整 reverse loop；不是完整 Diffusion Policy，也不声称重现 DDIM、图像 diffusion 或机器人执行。

## 4. Flow：路径决定 velocity target

本课选择噪声→数据的统一约定：

$$
x_\tau=(1-\tau)\epsilon+\tau a,
\quad u=a-\epsilon,
\quad L=\mathbb E\|v_\theta(x_\tau,o,\tau)-u\|^2.
$$

$$
x_{\tau+\Delta\tau}=x_\tau+\Delta\tau\,v_\theta(x_\tau,o,\tau).
$$

训练知道 endpoint a，用它构造 path/target；推理只给 noise、observation 与 tau，不能把真实 endpoint 当 oracle velocity 输入 sampler。Pi 机制 notebook 检查的是 endpoint 已知时的符号机制；ML06 另行训练真正的小型 conditional velocity network。

## 5. 两种时间与两种频率

tau/k 是生成过程时间；t 是机器人执行时间。增加去噪/ODE steps 不等于增加物理 feedback。长 action chunk 可以减少推理调用次数，但 open-loop 执行过久可能降低反应性。

FP2 原生 action generation 与下游 regulation 的分工不因为使用 diffusion/flow 而改变。1 kHz 插值输出也不能当作 learned policy 的新预测频率。

## 6. Notebook 的公平比较

ML06 对同一双峰合成条件分布，用相同 MLP 结构、训练 batch/steps 和 held-out observations 比较 DDPM 与 flow。它报告 mode compatibility、正负模式比例与 sampling wall time，不把这些叫 robot success。

模型结构相同不等于所有 compute 完全相等；DDPM schedule、flow 积分步数和 CPU runtime 会影响比较。先锁配置再查看 test，不凭训练 loss 宣布生成质量胜负。

## 验收

推导线性 path 的导数；区分 training target 与 inference inputs；说明 CVAE posterior/prior；解释 sampling steps、action horizon、execution horizon 与 feedback rate。

参考：[DDPM](https://arxiv.org/abs/2006.11239)、[Flow Matching](https://arxiv.org/abs/2210.02747)、[Pi 专题 P2](../pi_series/02_flow_matching.md)。
