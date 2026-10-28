# 第 8 课：从 CVAE 到 Diffusion Policy 的 action chunk

[上一课](07_policy_transformer_and_causality.md) · [下一课](09_flow_matching_for_actions.md) · [分步学习路线](../planning/vla_learning_path.md) · [Colab 生成实验](../labs/notebooks/ML06_diffusion_flow.ipynb)

第 6/7 课解释了观测怎样变成条件表示，以及哪些 token 能互相读取。本课接着回答：**条件已经给定，网络怎样生成一段可能有多种合理答案的动作？** 全课把条件写成 $c$，动作 chunk 写成 $A\in\mathbb R^{B\times H\times d_a}$。

本课分三次学习。每次建议约 45–75 分钟，包括阅读、手算和闭卷解释；这是学习时间估计，不是 notebook 运行时间。

| 单元 | 先解决的问题 | 当次留下的证据 |
|---|---|---|
| 8A：chunk 与 CVAE | 同一观测为什么可能对应多种动作？训练时的 latent 从哪里来？ | 训练/推理两张信息流图；一次 reparameterization 手算 |
| 8B：diffusion | 如何构造一次训练样本？为什么输出 noise 还能生成 action？ | 一张 batch 表；一次加噪与噪声预测 loss 手算 |
| 8C：采样与执行 | 生成循环怎样连接机器人时钟？怎样比较两类方法？ | 三个 horizon 的时间线；两分钟比较及一个反例 |

开始前只检查三件事：能解释 `[B,H,d_a]` 的三个轴；知道条件动作分布允许多峰；知道训练标签与部署输入不同。卡住哪项，就回看 [第 2 课](02_action_chunk_and_generative_policy.md) 或 [第 4 课](04_probabilistic_bc_and_training.md) 对应部分，不需要重读全部 ML 题卡。

## 8A：先从两条合理轨迹理解 chunk 与 CVAE

### 第一步：明确“预测一段”与“执行一段”

设机器人可以从障碍物左侧或右侧绕行。一个 chunk 同时包含接下来多个动作，它能表达“先向左、再向前”的关联。只对每个动作独立回归均值，可能给出穿过障碍物的中间路线。

为了手算，先用一个教学动作定义：每步只存二维平面位移，$d_a=2$；预测 3 步，$H=3$。一个训练样本是一个 $3\times2$ 矩阵，batch 中有两条样本时 shape 是 `[2,3,2]`。这里的 2D 位移仅为教学约定，不是 ACT 的原始机器人接口。

Chunk 的 $H$ 表示预测多少个动作位置。实际可以只执行前 $K\leq H$ 步，然后重新观测和预测。**在一个网络输出中联合预测未来动作，不等于部署时已经知道未来观测。**

### 第二步：给 decoder 一个表示动作变化的 latent

固定当前观测 $c$ 后，latent $z$ 允许 decoder 输出不同动作。训练中我们已经拿到专家执行的 $A$，所以可以用 posterior encoder 判断“这条示范对应怎样的 latent”。

通用 conditional VAE 的写法为：

$$
q_\phi(z\mid c,A),\qquad
z=\mu_\phi(c,A)+\sigma_\phi(c,A)\odot\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

$$
\hat A=g_\theta(c,z).
$$

通用公式里的 $c$ 是简写，具体 posterior 与 decoder 不必读取完全相同的条件。例如 ACT 的 posterior 使用动作序列和关节观测，decoder 结合视觉、关节观测与 latent。其原项目明确说明：测试时移除 posterior encoder，将 $z$ 设为 prior mean，即零。[ACT 原项目](https://tonyzhaozh.github.io/aloha/)

假设一个教学实现取 $B=2,H=3,d_a=2,d_z=4$，形状是：

```text
当前条件 c                 图像/state 的表示，具体结构依模型而定
专家动作 A                 [2,3,2]
posterior 输出 mu/logvar    [2,4] / [2,4]
latent noise epsilon       [2,4]
latent z                   [2,4]
decoder 输出 A_hat         [2,3,2]
```

`d_z=4` 是 latent width；`d_a=2` 是动作坐标数。两者不必相同。不能把每个 latent 坐标直接解释成某个关节角或某个力分量。代码若输出 `logvar`，需要用 $\sigma=\exp(0.5\,\mathrm{logvar})$ 得到标准差，再做 reparameterization。

### 第三步：区分“恢复示范”与“部署能取得的 latent”

训练目标同时要求动作重建良好、posterior 与可供推理使用的 prior 接近：

$$
L=\mathbb E_q[-\log p_\theta(A\mid c,z)]
+\beta D_{KL}\bigl(q_\phi(z\mid c,A)\|p(z\mid c)\bigr).
$$

重建项的具体形式取决于输出分布或实现。不能看到 CVAE 就默认一定用 MSE；ACT 作者的 policy 实现采用 L1 重建与加权 KL。[ACT policy 源码](https://github.com/tonyzhaozh/act/blob/main/policy.py)

若教学例子选择标准正态 prior，posterior 为对角 Gaussian，则一个样本的 KL 为：

$$
D_{KL}=\frac12\sum_{j=1}^{d_z}
\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).
$$

KL 约束让训练时利用的 latent 与部署可使用的 prior 有联系；它不保证模型一定有效利用 $z$，也不保证 $z$ 的各维有可解释语义。$\beta$、decoder 容量和数据都会影响结果。

```text
训练：专家 A + posterior 条件 → mu/logvar → z ───┐
      当前图像/state ──────────────────────────┴→ decoder → A_hat → loss

推理：prior 的指定用法 → z ─────────────────────┐
      当前图像/state ──────────────────────────┴→ decoder → action chunk
```

推理图中没有专家 $A$。通用 CVAE 可以从 prior 采样；ACT 的上述具体做法使用 $z=0$。后者不能推广成所有 CVAE 的必选规则。固定 $c,z$ 的一次确定性前向，也不会自动展示多种动作模式。

### 手算与自检

设 $\mu=(1,-1)$、$\log\sigma^2=(0,0)$、$\epsilon=(0.5,-0.5)$。

1. 写出 $\sigma$ 与 $z$，再计算相对于标准正态的 KL。
2. 若推理时把测试专家动作送进 posterior，指标为什么不能代表真实部署？
3. latent noise shape 能否写成 `[B,H,d_a]` 并声称这是同一个模型？

**检查答案：** $\sigma=(1,1)$，$z=(1.5,-1.5)$，KL 为 $1$。第二题引入了部署时不可得的标签；第三题会改变 latent 的建模方式，不能把 latent width 与 chunk shape 混用。

**完成标准：** 闭卷画出两张图，解释每条箭头在训练和部署时是否存在；能说明 prior、posterior、decoder 各负责什么。

## 8B：一次 diffusion 训练究竟输入和输出什么

### 第一步：先把干净 chunk 加噪

本节采用 DDPM 的 epsilon prediction 参数化。把专家动作写成 $A_0$，离散噪声等级写成 $k$；这里 $k$ 是生成过程中的索引，不是机器人将要执行的第 $k$ 步。

定义 $\alpha_k=1-\beta_k$，$\bar\alpha_k=\prod_{i=1}^k\alpha_i$，则可以直接构造任意噪声等级的输入：

$$
A_k=\sqrt{\bar\alpha_k}A_0
+\sqrt{1-\bar\alpha_k}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

当 $\bar\alpha_k$ 接近 1，输入主要保留动作信号；接近 0 时，输入主要是噪声。训练可直接采样 $k$ 并算出 $A_k$，无需先模拟前面所有加噪步。[DDPM 原论文，§2 与 Algorithms 1–2](https://arxiv.org/html/2006.11239v2)

### 第二步：监督模型预测加入的 noise

```text
condition c                当前图像/state/history 的编码
clean actions A_0          [B,H,d_a]
noise level k              [B]
alpha_bar[k]               [B] → reshape 为 [B,1,1]
Gaussian epsilon           [B,H,d_a]
noisy actions A_k          [B,H,d_a]
predicted epsilon          [B,H,d_a]
valid action mask          [B,H] 或 [B,H,d_a]
```

$$
L_{\epsilon}
=\mathbb E\left[
\frac{\sum_{b,h,j}m_{bhj}
\left(\epsilon_\theta(A_k,c,k)_{bhj}-\epsilon_{bhj}\right)^2}
{\sum_{b,h,j}m_{bhj}}
\right].
$$

这里 $m$ 已展开到 action scalar；若 mask 只记录有效时间步，就要在动作维广播并使用对应分母。第 9 课会专门手算这个分母。不同论文也可以预测 clean sample、score 或其他参数化，**不能把所有 diffusion 都定义为只预测 epsilon**。

一次训练迭代的过程是：取条件与专家 chunk → 随机噪声等级和噪声 → 加噪 → 网络预测 → 与已知噪声计算 loss → backward/update。训练时知道专家动作，是因为它充当监督信号；部署采样则从 fresh noise 开始。

### 第三步：用一个二维例子检查是否真的理解

令一个动作位置的 clean value 是 $(2,-1)$，噪声是 $(0,1)$，$\bar\alpha_k=0.64$。

先自己计算 $A_k$。再假设网络预测噪声 $(0.1,0.8)$，计算这两个 scalar 上的 mean squared error。

**检查答案：** $\sqrt{0.64}=0.8$，$\sqrt{0.36}=0.6$，所以 $A_k=(1.6,-0.2)$；噪声 MSE 为 $(0.1^2+(-0.2)^2)/2=0.025$。模型此时监督的是噪声 $(0,1)$，不是动作 $(2,-1)$。

如果噪声预测恰好正确，可以代数恢复该训练样本的 $A_0$：

$$
\hat A_0=\frac{A_k-\sqrt{1-\bar\alpha_k}\hat\epsilon}{\sqrt{\bar\alpha_k}}.
$$

这个恒等式帮助理解预测噪声为什么有用，但不证明真实网络能一步生成准确动作。高噪声下信号很弱，网络存在误差，采样器还要按其 schedule 更新。

### 第四步：把 backbone 与训练目标拆开理解

Temporal U-Net 与 time-series Transformer 都可以承担 denoiser 的计算。一个 Transformer 输出连续 epsilon 时，并不因此变成 autoregressive token policy。CVAE 描述 latent-variable 建模方式，ACT 是具体策略设计；比较它们时要分别问：输入是什么、监督量是什么、网络如何算、推理循环怎样走。

**完成标准：** 不看公式构造一份 diffusion training batch；解释 $k$ 的作用、为什么训练只抽一个噪声等级也能学习多个等级，以及输出 epsilon 的 shape 为什么与 action chunk 一致。

## 8C：从去噪循环走到 receding-horizon execution

### 第一步：采样使用预测值，不能调用训练标签

在本课的 DDPM 参数化下，一个反向更新可写为：

$$
\mu_\theta(A_k,c,k)=\frac1{\sqrt{\alpha_k}}
\left(A_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}
\epsilon_\theta(A_k,c,k)\right),
$$

$$
A_{k-1}=\mu_\theta(A_k,c,k)+\sigma_k\xi,
\qquad \xi\sim\mathcal N(0,I).
$$

令总噪声等级数为 $N_{\mathrm{diff}}$，采样从 $A_{N_{\mathrm{diff}}}\sim\mathcal N(0,I)$ 开始，按指定 schedule 往低噪声方向走；最后一步通常不再加随机噪声。$\sigma_k$ 的选取与 scheduler 有关，不能把这条更新原封不动用于所有 diffusion sampler。公式对应的 DDPM 推导见[原论文 §3.2](https://arxiv.org/html/2006.11239v2)。

模型反复读的是“当前 noisy chunk、当前条件、当前噪声等级”。它不会在每次去噪时获取一个新的真实未来观测。同一条件下图像编码有机会复用，但是否实现缓存、缓存哪些量，应继续检查具体代码。

### 第二步：把四种长度/时钟放到同一张纸上

| 量 | 含义 | 教学例子 |
|---|---|---|
| Observation horizon | 策略读多少历史观测 | 当前帧与上一帧，共 2 帧；仍需声明帧间隔 |
| Prediction horizon $H$ | 一次生成多少动作位置 | 50 个，动作间隔 0.02 s |
| Execution horizon $K$ | 更新策略前先执行多少个 | 10 个，对应约 0.2 s |
| Sampling steps | 生成这一个 chunk 调几次 denoiser | 例如 16 次，与 $H/K$ 无直接等号 |

上述 $H=50$ 在“每个动作负责 0.02 s”的约定下覆盖约 1 s；若计第一个与最后一个采样点之间的时间差，则为 $(H-1)\times0.02=0.98$ s。面试中说明时间戳约定，比只背“长度除频率”更准确。

Diffusion Policy 原始设计结合视觉条件和 receding-horizon control：预测一段动作，再滚动更新计划。[DP 原项目](https://diffusion-policy.cs.columbia.edu/)

若推理耗时 0.3 s，既定执行窗口只有 0.2 s，下一段可能不能及时到达。把已有 reference 插值成 1 kHz 只增加输出采样点；新图像和新策略判断仍受原采样及计算时序约束。要实际测量端到端延迟，并考虑减少采样步、调整执行窗口或异步调度等方案的取舍。

ACT 的 temporal ensembling 在同一实际时刻聚合不同预测窗口对该时刻的动作；它与随意平均两条不同绕行路线不是同一个操作。实现时仍要核对时间对齐、动作表示、聚合权重和模式一致性。

### 第三步：把 Colab 当作机制实验

[ML06 notebook](../labs/notebooks/ML06_diffusion_flow.ipynb) 的“DDPM：预测 epsilon”与“两个 inference sampler”对应本课 8B/8C。它的代码被设计为训练一维 conditional DDPM 与 flow，没有 CVAE/ACT 实现；8A 采用手算和作者代码阅读完成。本课程材料尚无此次 Colab 实际运行证据。

阅读时先标出 `endpoint`、`noise`、`state`、`time`、`predicted_noise`；然后查看 `sample_diffusion` 的参数，确认没有专家 endpoint。它没有视觉 encoder、真实 chunk dynamics 或机器人环境。[VLA01](../labs/notebooks/VLA01_token_action_trace.ipynb) 可以补充多模态 shape/mask 检查，但两份教学 notebook 合起来仍不等于完整 Diffusion Policy 复现。

实际在 Colab 执行后，记录 seed、训练预算、schedule、采样步数、输出模式分布与计时；没有运行就保持“待运行”。不能直接横比 diffusion noise loss 与 flow velocity loss 的大小来选模型。

### 面试表达：先讲信息流，再讲取舍

**问题：ACT/CVAE 和 diffusion policy 都能预测 chunk，区别在哪里？**

可用的短答结构：两者都以当前观测为条件预测动作序列。CVAE 通过 latent 表达示范变化，训练额外用 posterior 读取专家动作，部署改用 prior 的指定用法；ACT 原始推理采用 $z=0$。Diffusion 对带噪 chunk 学习去噪函数，部署从 noise 开始多步更新。比较时还需要固定动作语义、数据、执行窗口与计算预算，再看采样质量和闭环效果。

继续回答三个追问：

1. **“Transformer 是否意味着逐 token 生成？”** 不一定；看输出参数化和 attention/采样方式。
2. **“预测 50 步就有 1 s 完全开环吗？”** 取决于执行 $K$ 步后是否重规划，以及低层反馈；$H$ 不能独自决定闭环程度。
3. **“采样模式覆盖好，机器人就成功吗？”** 还要经过动作解码、控制、延迟和接触，最终需要相同条件下的 rollout 评价。

**本课通过标准：** 能画 CVAE 两套信息流、手构造 diffusion batch、区分 backbone/objective、计算预测与执行时间、解释实验支持与不支持的结论。然后进入 [第 9 课](09_flow_matching_for_actions.md)，把同样的问题换成 flow matching 来回答。
