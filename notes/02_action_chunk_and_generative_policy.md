# 第 2 课：Action Chunk 与生成式动作策略

## 本课目标

学完后，你应当能独立回答：

1. action chunk 为什么能降低 imitation learning 的有效 horizon？
2. prediction horizon、execution horizon 和 observation horizon 有什么区别？
3. ACT、Diffusion Policy、π0 和 π0-FAST 分别怎样表示和生成动作？
4. diffusion timestep / flow timestep 为什么不是 robot timestep？
5. 为什么 action chunk 越长并不一定越好？

参考原文：[ACT](https://arxiv.org/abs/2304.13705)、[Diffusion Policy](https://arxiv.org/abs/2303.04137)、[π0](https://arxiv.org/abs/2410.24164)、[FAST](https://arxiv.org/abs/2501.09747)。

---

## 1. 统一问题：预测一个条件动作分布

给定最近的观测、语言指令和机器人状态，我们希望学习：

$$
p_\theta(A_t\mid O_t,\ell),
$$

其中

$$
O_t=(o_{t-T_o+1},\ldots,o_t),\qquad
A_t=(a_t,a_{t+1},\ldots,a_{t+T_p-1}).
$$

- $T_o$：observation horizon，模型能看到多少个历史观测。
- $T_p$：prediction horizon，一次预测多少个未来动作。
- $T_a$：execution horizon，一次真正执行多少个动作后重新观察和规划。

通常 $T_a\le T_p$。模型可以预测较长的连续计划，只执行前面一小段，然后用新观测重新规划。这就是 receding-horizon execution。

### 一个具体例子

机器人控制频率为 20 Hz：

```text
To = 2 observations
Tp = 16 predicted actions = 0.8 s future
Ta = 4 executed actions  = 0.2 s before replanning
```

此时：

- 低层控制器仍以 20 Hz 接收动作；
- policy 每 0.2 s 重新规划一次，即 5 Hz；
- 每次预测覆盖未来 0.8 s；
- 后 12 个动作提供时间一致性和未来结构，但本轮不会全部执行。

所以“动作输出为 20 Hz”不等于“大模型以 20 Hz 做完整推理”。

---

## 2. Action chunk 为什么有用

假设一段任务的 **episode horizon** 为 $T_{\mathrm{episode}}=1000$ 个控制步；这里的 1000 是完成整项任务所需的总步数，不是模型一次预测的 prediction horizon。单步 BC 每次只预测并执行一个 $a_t$，因此从 policy 层面看约有 1000 次连续决策。

现在假设 action prediction horizon 为 $T_p=50$，并且每次把预测出的 50 步全部可靠执行，即 execution horizon $T_a=k=50$。policy 每 50 个控制步才重新决策一次，所以粗略的有效 **policy-decision horizon** 为：

$$
T_{\text{effective}}
\approx
\left\lceil\frac{T_{\mathrm{episode}}}{T_a}\right\rceil
=\frac{1000}{50}=20.
$$

这 20 表示完成 episode 需要约 20 次 chunk-level policy decisions；每个 decision 内仍包含 50 个真实控制动作。若模型预测 $T_p=50$ 步但只执行 $T_a=10$ 步便重新规划，则有效决策 horizon 是 $1000/10=100$，不是 $1000/50$。

这不是说 chunk 内不会犯错，而是模型一次输出了局部连贯的运动结构，减少了相邻步骤独立预测造成的抖动和误差累积。

Action chunk 还让模型直接学习时间相关性。例如插孔动作不是 50 个互不相关的向量，而是：

```text
靠近 → 减速 → 接触 → 沿表面搜索 → 对准 → 插入
```

### 为什么不能无限增大 chunk

当 $T_a=T_p$ 且 chunk 很长时，执行接近 open loop。执行期间出现物体移动、视觉误差、接触卡滞，policy 都不能及时修正。

因此 chunk length 存在权衡：

$$
\text{long chunk}
\Rightarrow
\begin{cases}
\text{更强的时间一致性}\cr
\text{更短的有效任务 horizon}\cr
\text{更弱的反馈响应}\cr
\text{更难的长序列建模}
\end{cases}
$$

ACT 的消融正好体现这一点：$k=1$ 过于短视；接近完整 episode 的 chunk 又缺少反应性，中间值表现更好。

---

## 3. ACT：一次回归一段动作

ACT 的核心并不只是“用了 Transformer”，而是三件事：

1. action chunking；
2. CVAE 对人类示范中的非平稳、多模态风格建模；
3. temporal ensemble 平滑来自不同时间查询的重叠预测。

### 3.1 训练

训练时，CVAE encoder 读取当前关节状态 $\mathbf q_t$ 和从时刻 $t$ 开始的专家 action chunk $\mathbf A_t=(\mathbf a_t,\ldots,\mathbf a_{t+k-1})$，并输出 latent style $\mathbf z$ 的条件概率分布：

$$
q_\phi(\mathbf z\mid \mathbf q_t,\mathbf A_t)
=
\mathcal N\!\left(
\boldsymbol\mu_\phi(\mathbf q_t,\mathbf A_t),
\operatorname{diag}\!\left(\boldsymbol\sigma_\phi^2(\mathbf q_t,\mathbf A_t)\right)
\right).
$$

这里 $\mathbf q_t$ 中的 $q$ 是机器人关节状态，而 $q_\phi(\cdot)$ 中的 $q$ 是变分推断里对 posterior distribution 的传统命名，两者没有关系。$\phi$ 表示 CVAE encoder 的可学习参数。Encoder 针对每个训练样本输出均值向量 $\boldsymbol\mu_\phi$ 和标准差向量 $\boldsymbol\sigma_\phi$，它们共同描述 latent $\mathbf z$ 的对角高斯分布。训练时通过 reparameterization trick 采样：

$$
\boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\mathbf I),
\qquad
\mathbf z
=
\boldsymbol\mu_\phi
+
\boldsymbol\sigma_\phi\odot\boldsymbol\epsilon,
$$

其中 $\mathbf I$ 是单位矩阵，表示标准高斯各 latent 维度相互独立且方差为 1；$\odot$ 表示逐元素乘法。

policy decoder 根据图像、当前关节状态和 $z$ 预测动作：

$$
\hat A_t=f_\theta(I_t,q_t,z).
$$

简化后的目标为：

$$
\mathcal L_{\text{ACT}}
=\|A_t-\hat A_t\|_1
+\beta D_{\mathrm{KL}}\big(q_\phi(z\mid q_t,A_t)\,\|\,\mathcal N(0,I)\big).
$$

第一项要求重建专家动作；第二项把 latent 分布约束到测试时可以采样的先验附近。CVAE encoder 只在训练时使用。

### 3.2 推理

测试时没有专家 chunk，使用先验 latent，decoder 一次输出连续 action chunk。Temporal ensemble 可以在每个时刻重新查询，使同一未来动作获得来自多个历史 query 的预测，再做指数加权平均：

$$
\bar a_t=\frac{\sum_i w_i\hat a_t^{(i)}}{\sum_iw_i}.
$$

它减少相邻 query 的建模噪声，但也可能把策略需要的突然修正平滑掉，因此不是所有方法都适合使用。

### 3.3 ACT 的概率表达能力

ACT 的 CVAE 理论上能表达多模态动作，但实践中常固定或弱化 latent，以获得稳定、可重复的控制。不要简单地把“CVAE”理解成必然能完整解决 multimodality；还要看 latent 是否被实际使用、KL 权重和测试时怎样选 $z$。

---

## 4. Diffusion Policy：从噪声迭代得到 action chunk

Diffusion Policy 不直接回归 $A_t$，而是学习如何从被加噪的动作中预测噪声。

### 4.1 前向加噪

这里同时存在两种时间：

- $t$：真实机器人时间；
- $k$：diffusion noise level，与真实时间无关。

对数据中的干净 chunk $A_t^0$，采样噪声 $\epsilon\sim\mathcal N(0,I)$：

$$
A_t^k=\sqrt{\bar\alpha_k}A_t^0+\sqrt{1-\bar\alpha_k}\epsilon.
$$

当 $k$ 较大，$A_t^k$ 更像噪声；当 $k=0$，它是干净动作。

### 4.2 训练目标

网络读取 noisy action、noise level 和 observation，预测加入的噪声：

$$
\mathcal L_{\text{diff}}
=\mathbb E_{A_t^0,O_t,k,\epsilon}
\left[\left\|\epsilon-\epsilon_\theta(A_t^k,k,O_t)\right\|_2^2\right].
$$

训练时只需随机选择一个 $k$ 做一次网络预测，不需要完整运行全部去噪链。

### 4.3 推理

推理从高斯噪声开始：

$$
A_t^K\sim\mathcal N(0,I),
$$

然后迭代：

$$
A_t^K\rightarrow A_t^{K-1}\rightarrow\cdots\rightarrow A_t^0.
$$

最终 $A_t^0$ 是条件于当前观测的动作序列。它能在动作空间中保留多个高概率区域，而不是被迫输出它们的均值。

### 4.4 Receding horizon 是闭环的来源

Diffusion 本身并不自动等于 closed loop。闭环来自：预测 $T_p$ 步，只执行 $T_a$ 步，再读取新观测重新采样。

若 diffusion inference 耗时为 $L$ 秒，则 $T_a/f_{ctrl}$ 必须给下一次推理留出足够时间，否则会出现动作断档或不得不进一步拉长 open-loop execution。

---

## 5. π0：把 diffusion-style 生成换成 flow matching

π0 同样生成连续 action chunk，但网络预测的是 velocity field，而不是每个离散 noise level 上的噪声。

论文令：

$$
A_t=[a_t,\ldots,a_{t+H-1}],\qquad H=50.
$$

观测条件为多个 RGB 图像、语言和本体状态。采样：

$$
\epsilon\sim\mathcal N(0,I),\qquad \tau\in[0,1],
$$

构造线性路径：

$$
A_t^\tau=\tau A_t+(1-\tau)\epsilon.
$$

这条路径的目标速度是常量：

$$
u=A_t-\epsilon.
$$

因此训练目标为：

$$
\mathcal L_{\text{FM}}
=\mathbb E\left[
\left\|v_\theta(A_t^\tau,\tau,O_t)-(A_t-\epsilon)\right\|_2^2
\right].
$$

推理时从 $A_t^0\sim\mathcal N(0,I)$ 出发，用 Euler 法积分：

$$
A_t^{\tau+\delta}
=A_t^\tau+\delta v_\theta(A_t^\tau,\tau,O_t).
$$

论文实现用 10 个 flow steps 生成整个 chunk。注意这里的 $\tau$ 是生成过程时间，$t$ 才是机器人时间。

### Diffusion 与 flow matching 的直觉区别

```text
Diffusion:
  学习不同噪声等级下如何去掉噪声
  noise K → K-1 → ... → clean action

Flow matching:
  学习连续速度场，把样本从噪声分布运输到动作分布
  τ = 0 --------------------→ τ = 1
```

两者都能表达连续、多峰、高维 action chunk，也都需要多次网络调用完成采样。它们的差别在训练 target、概率路径和数值求解方式，不在于“一个连续、另一个离散”。

---

## 6. π0-FAST：不做连续生成，改为压缩后自回归

逐维、逐时刻量化高频 action chunk 会产生很长的 token 序列，而且相邻时刻高度相关。例如一秒 50 Hz、32 维的 chunk，在朴素表示下对应 1600 个标量符号。

FAST 利用动作轨迹的低频结构：

```text
continuous action chunk
  → normalize
  → per-dimension discrete cosine transform (DCT)
  → quantize frequency coefficients
  → serialize coefficients
  → BPE compress repeated coefficient patterns
  → short action-token sequence
```

低频 DCT 系数描述轨迹的整体形状，高频系数描述快速变化。平滑机器人动作通常只需较少有效系数，因此比逐时刻 binning 更容易压缩。

π0-FAST 随后使用标准自回归目标：

$$
\mathcal L_{\text{AR}}
=-\sum_{i=1}^{L}\log p_\theta(z_i\mid z_{<i},O_t),
$$

再由 FAST decoder 把 token 还原成连续 chunk。

所以：

$$
\boxed{\pi_0:\text{continuous flow matching}}
\qquad
\boxed{\pi_0\text{-FAST}:\text{discrete autoregressive tokens}}
$$

FAST 是 action tokenizer；π0-FAST 是使用它训练出的 VLA policy。二者不能混为一个概念。

---

## 7. 四种方法放在一张表里

| 方法 | 模型输出空间 | 训练 target | 推理 | 多峰性 | 主要实时性代价 |
|---|---|---|---|---|---|
| ACT | 连续 action chunk | L1 reconstruction + KL | 一次 decoder 前向 | 由 CVAE latent 承担 | 大模型单次前向 |
| Diffusion Policy | 连续 action chunk | noise prediction | 多步去噪 | 强 | 多个 denoising steps |
| π0 | 连续 action chunk | vector-field velocity | ODE 数值积分 | 强 | 多个 flow steps + VLM |
| π0-FAST | 离散压缩 token | next-token CE | 自回归 token 解码 | 由 token distribution 承担 | 序列长度 × AR 前向 |

模型参数量、实现优化和硬件会改变实际速度，因此不能只根据“多步”或“自回归”判断谁一定更快。

---

## 8. 与当前力觉论文的连接

这一课解释了本地资料反复出现 slow-fast 结构的原因：

- [ImplicitRDP](../2512.10946v1.pdf)：生成式视觉策略负责较慢的 chunk planning，force 负责更快的局部反应。
- [FAVLA](../2602.23648v1.pdf)：slow VLM 不能以 force sensor 的频率重算，因此让 fast action expert 利用 force/state。
- [PhaForce](../2603.08342v1.pdf)：仅有 fast residual 还不够，还要根据接触阶段决定何时、在哪个子空间修正。

它们共同针对同一个结构性矛盾：

$$
\text{large generative policy is slow}
\quad\text{but}\quad
\text{contact dynamics are fast}.
$$

解决方案通常不是强迫整个 VLM 高频运行，而是让语义规划、chunk generation、fast correction 和低层 controller 分工。

---

## 9. 计算练习

### 练习 A：三个 horizon

控制频率 50 Hz，$T_p=50$，$T_a=10$，$T_o=2$。

请计算：

1. 每次预测覆盖多少秒？
2. 每隔多少秒重新规划？
3. policy replanning frequency 是多少 Hz？
4. 如果一次推理耗时 260 ms，不做异步推理会发生什么？

### 练习 B：多峰动作

机械臂要绕过杯子到达目标，数据中一半从左绕，一半从右绕。说明以下策略可能输出什么：

1. deterministic MSE BC；
2. Diffusion Policy；
3. autoregressive token policy。

### 练习 C：诊断系统

某 VLA 每次生成 1 秒动作，全部执行完才重新读取相机；底层位置控制为 100 Hz。它能否称为 100 Hz closed-loop VLA？请分别写出 model frequency、command frequency 和 feedback-to-policy frequency。

---

## 10. 自测答案与判分点

先独立完成再展开核对。

<details>
<summary>练习 A</summary>

1. $50/50=1$ 秒。
2. $10/50=0.2$ 秒。
3. $1/0.2=5$ Hz。
4. 同步推理比 200 ms 的执行窗口更慢，会造成动作断档、降低实际更新频率，或迫使系统增加 $T_a$；可用异步推理、缩短采样、warm start 或 fast residual controller 缓解。

</details>

<details>
<summary>练习 B</summary>

1. 可能平均左右两条轨迹，直接朝杯子移动。
2. 从噪声采样后可落入“左绕”或“右绕”的概率模态。
3. 可以给左右两种 token 序列分配概率并采样其中一种；效果取决于 tokenizer 是否保留必要的轨迹差异。

</details>

<details>
<summary>练习 C</summary>

不能称为 100 Hz closed-loop VLA。低层 controller/command tracking 是 100 Hz；VLA model 和 feedback-to-policy 都约 1 Hz。模型在一秒 chunk 内不使用新相机反馈，因此 policy 层是长时间 open loop。

</details>

---

## 本课通过标准

合上笔记后，能够画出下面四条路径，并在每条路径上标注训练 target 和推理循环：

```text
ACT:       observation + z → continuous chunk
Diffusion: observation + noisy chunk + k → predicted noise → repeated denoising
π0:        VLM condition + noisy chunk + τ → velocity → ODE integration
π0-FAST:   VLM condition → token1 → token2 → ... → FAST decode → chunk
```

然后任选一篇本地 force-aware policy，指出它在 chunk 内如何获得新反馈。做到这一步，才算真正连接上下一阶段的 π0 架构学习。
