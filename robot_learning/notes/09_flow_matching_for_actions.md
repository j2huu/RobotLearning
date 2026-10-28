# 第 9 课：Flow Matching 的连续动作 head

[上一课](08_cvae_and_diffusion_policy.md) · [下一课](10_from_vlm_to_vla.md) · [分步学习路线](../planning/vla_learning_path.md) · [Colab ML06](../labs/notebooks/ML06_diffusion_flow.ipynb) · [完整推导 P2](pi_series/02_flow_matching.md)

第 8 课已经建立“当前条件 → 生成一段动作 → 执行其中一部分”的框架。本课学习另一种生成方法：训练网络估计 action 空间中的向量场，再从噪声积分得到动作样本。始终分清 **物理时间 $t$、生成时间 $\tau$、chunk 内动作位置 $h$**。

| 单元 | 问题 | 当次留下的证据 |
|---|---|---|
| 9A：路径与监督量 | 从两个端点怎样得到带噪输入和 target？ | 手推导数；计算一对二维端点 |
| 9B：tensor、head 与 loss | 路径如何接入 VLA，padding 如何影响 loss？ | 完整 shape 表；有效 scalar 分母手算 |
| 9C：采样、单位与诊断 | 没有专家终点时怎样生成动作？ | Euler 手算；符号反例；采样到执行流程 |

每单元建议约 45–75 分钟，是学习预算估计。前置自检：知道第 8 课训练输入/标签/推理输入的差异，能说明 `[B,H,d_a]` 与 token width 的差别。导数或条件均值暂时不熟时，按需查 [ML 生成模型](ml_interview/09_generative_models.md) 与 [P2](pi_series/02_flow_matching.md)。

## 9A：先固定端点，再推公式

### 第一步：我们在什么空间里移动

假设 $H=8,d_a=7$，一个 chunk 含 56 个动作 scalar。Flow 的生成过程在这个 56 维动作表示空间中改变样本；batch 增加独立样本的数量。采样中间态可能没有合理的机器人动作意义，因此不会把每个 noisy 中间态逐个送给机器人。

把训练专家 chunk 记为 $A$，独立 Gaussian noise 记为 $E$。本课统一采用 **noise → data**：

$$
A\in\mathbb R^{B\times H\times d_a},\qquad
E\sim\mathcal N(0,I),\qquad
X_\tau=(1-\tau)E+\tau A.
$$

先做端点检查：$X_0=E$，$X_1=A$。若这两句还不能脱口而出，暂时不要背 loss 或 sampler。

### 第二步：target 由求导产生

对生成时间求导，并把这一次采样的两个端点视为常量：

$$
U=\frac{dX_\tau}{d\tau}=A-E.
$$

$A,E,X_\tau,U$ 的 shape 都相同。这里“速度”是动作表示对生成时间的变化率，不是机器人对物理时间的速度。

手算例子：一个动作位置取 $A=(2,-1)$、$E=(-2,3)$、$\tau=0.25$。

**检查答案：** $X_{0.25}=(-1,2)$，$U=(4,-4)$。换成 $\tau=0.75$，$X_{0.75}=(1,0)$，这条特定直线路径的 target 仍为 $(4,-4)$。

### 第三步：训练知道端点，网络为什么不需要读端点

训练采样出 $(A,E,\tau)$ 后，用它们构造 $X_\tau$ 与监督 $U$。网络接口是：

$$
v_\theta(X_\tau,c,\tau),
$$

其中 $c$ 是当前可用的图像、语言、状态或历史条件。监督端点 $A$ 没有作为一条额外的 condition 输入。这个设计允许部署时继续使用同一接口，只不过起点换成 fresh noise。

平方损失下，理想预测量可以理解为：

$$
v^*(x,c,\tau)=\mathbb E[A-E\mid X_\tau=x,c,\tau].
$$

不同训练端点可能经过相近位置，因此网络学到给定当前状态的平均方向。这里的条件包含 $x$ 和 $\tau$，不是只给观测 $c$ 的单步动作均值；不同噪声起点经过场的演化仍可能产生不同模态。Flow matching 的一般框架见[原论文](https://arxiv.org/abs/2210.02747)，本课只使用便于手算的线性路径。

### 小练习：找出非法推理输入

下面两种代码接口中，哪个可以用于真实推理？

```text
sample(condition, fresh_noise, trained_model)
sample(condition, expert_action, trained_model)
```

**检查答案：** 第一种；第二种依赖测试时未知的专家标签。已知 $A$ 后直接使用 $A-E$ 积分，只能做路径/符号的教学检查。

**完成标准：** 用任意标量例子推导端点、target 和时间方向，并说明网络输入为什么不包含专家终点。

## 9B：把路径接到 VLA 的 tensor 和 loss

### 第一步：一个完整 training batch

延续 VLA01 的教学尺寸 $B=2,H=8,d_a=7,d=32$。下面所有 action 坐标都假设已用 train split 统计量归一化。

```text
clean action A             [2,8,7]
independent noise E        [2,8,7]
generation time tau        [2,1,1]
path input X_tau           [2,8,7]
target U=A-E               [2,8,7]

image/text prefix          [2,38,32]   教学例子：32 image + 6 text slots
state token                [2,1,32]
Linear(7→32)(X_tau)        [2,8,32]
time embedding             [2,1,32]   广播到8个action位置
action hidden              [2,8,32]
Linear(32→7)(hidden)       [2,8,7]     预测向量场
```

这里 $\tau$ 的 `[2,1,1]` 在 horizon 与 action 维广播：每个样本的整段 chunk 使用同一个生成时刻。网络也需要 action-position 信息来区分 8 个未来位置。**生成时刻相同，并不意味着 8 个动作在同一物理时刻执行。**

`7→32→7` 中的 32 是 hidden width，不能解释成动作维数变为 32DoF。具体生产模型的原生动作、padded 动作宽度和 hidden width 必须分别追踪。VLA01 使用共享小 Transformer 检查机制，并非 π0 的多 expert 实现。

### 第二步：先算有效 scalar，再写 mean

本课采用“batch 内所有有效 action scalar 等权”的 reduction：

$$
L_{\mathrm{batch}}=
\frac{\sum_{b,h,j}m_{bhj}
\left(v_\theta(X_\tau,c,\tau)_{bhj}-U_{bhj}\right)^2}
{\sum_{b,h,j}m_{bhj}}.
$$

训练再对 batch 采样求期望。若希望每条示范等权，也可以先做每条样本的有效均值再平均；那是另一种样本权重，不能无意混用。

VLA01 第一条样本 8 个动作有效，第二条只有前 6 个有效；每个动作含 7 个有效维度。

```text
action_valid        [2,8]
unsqueeze(-1)       [2,8,1]
expand_as(target)   [2,8,7]
有效scalar总数      (8+6)×7 = 98
```

若每个有效 scalar 的 squared error 都是 1，正确 loss 为 $98/98=1$。

- 只除以有效时间步 14，结果变成 7，改变了 loss 的尺度。
- 先把无效值乘零再对全部 $2\times8\times7=112$ 个元素 `.mean()`，结果为 $98/112=0.875$，padding 比例会影响尺度。
- 先按每条样本的有效 scalar 均值再平均，在误差不均匀时还会改变两条样本的相对权重。

这些操作不一定都“非法”，但必须与宣称的优化目标一致。对于含无效数值的目标，先清理或选择有效条目；`NaN * 0` 不会自动变成有效的零。

全 batch 没有有效目标时应跳过/拒绝该 batch 或明确处理，不能让分母为零。Loss mask 也不是第 7 课的 attention mask：前者决定监督哪些位置，后者决定网络读取哪些信息。

### 第三步：追踪一次 gradient，而不是猜测训练成功

对 [VLA01](../labs/notebooks/VLA01_token_action_trace.ipynb) 的“Masked loss 与一次 backward”，先在纸上写出上述 98，再查看 `valid_scalar` 与分母。它只做一次 backward 检查 gradient path，没有 optimizer update；看见有限梯度不能说明这个随机模型已经会做任务。

对 [ML06](../labs/notebooks/ML06_diffusion_flow.ipynb) 的“Flow：noise→data”，追踪 `endpoint → state/target → model → loss → optimizer.step`。该 notebook 被设计为学习一个一维合成分布；当前没有本轮 Colab 运行记录。

**完成标准：** 闭卷填完 shape 表，解释时间广播、hidden width 与 action width，并在更换 $H/d_a$/valid mask 后重新算出 loss 分母。

## 9C：推理积分、物理单位与故障定位

### 第一步：采样只有条件、当前状态和网络

采样 ODE 与显式 Euler 更新为：

$$
\frac{dX_\tau}{d\tau}=v_\theta(X_\tau,c,\tau),
\qquad X_0\sim\mathcal N(0,I),
$$

$$
X_{j+1}=X_j+\Delta\tau\,v_\theta(X_j,c,\tau_j).
$$

假设用 $J$ 步从 0 积分到 1，均匀步长为 $\Delta\tau=1/J$。$J$ 是采样迭代数，$H$ 是未来动作数；它们是两个不同的循环轴。

```text
training:
  取 observation condition c、expert action A
  采 noise E 和 tau
  X = (1-tau)*E + tau*A
  U = A-E
  predict v(X,c,tau) → masked loss → backward/update

sampling:
  取当前 condition c
  X = fresh noise
  对 j=0,...,J-1:
    tau = j/J
    X = X + v(X,c,tau)/J
  返回 X，随后按 action contract 解码
```

每个训练 batch 通常抽取噪声等级来回归向量场，不要求先进行完整采样链再计算终点误差。训练循环与采样循环应能分别画出来。

### 第二步：先做可验证手算，再看真实网络

**仅检验符号的已知端点例子：** $E=1,A=3$，所以常速度 $U=2$。取 $J=4$，得到：

$$
1\rightarrow1.5\rightarrow2\rightarrow2.5\rightarrow3.
$$

该例中常向量场已知，Euler 恰好到达终点；它不包含学习能力证据。

**模拟网络输出的两步例子：** 起点 $X_0=1$，$\Delta\tau=0.5$。假设模型第一步输出 1.6，则 $X_1=1.8$；在新状态与新时间，模型第二步输出 1.2，则 $X_2=2.4$。这里网络可能在每一步改变方向/大小，推理接口没有真实目标供它直接计算差值。

只把已知端点例子的 target 改为 $-2$，却继续正向积分，结果会到 $-1$。这是快速定位正负号混用的方法。

### 第三步：另一种时间约定怎样保持一致

若改用 $s=1-\tau$，则路径可以写为：

$$
X_s=(1-s)A+sE,\qquad U_s=E-A.
$$

此时噪声在 $s=1$，数据在 $s=0$，采样需要 $\Delta s<0$。target 与积分方向一起变才自洽，$(-0.25)\times(-2)=0.5$ 仍沿正确方向前进。

[P2](pi_series/02_flow_matching.md) 对本地 π0 v1 公式、按字面存在的符号/协方差问题，以及固定 revision 的 openpi 实现分别作了核对。主课先掌握一套 noise→data 约定，再查版本差异。教学中均匀采 $\tau$ 也不代表所有论文都采用均匀时间分布。

### 第四步：输出 field 不是控制器命令

若模型空间的动作已归一化，则向量场单位是“normalized action / 生成时间”。即使最终 action 的某一维代表 m/s，flow field 也不能直接解释成末端 twist；它表示对这个速度命令数值的生成时间导数。

例如一个最终 normalized joint-position sample 为 2，train 均值是 0.5 rad、标准差是 0.1 rad，则反归一化值为：

$$
a=2\times0.1+0.5=0.7\ \mathrm{rad}.
$$

这仍只是目标关节位置；后续控制器如何跟踪、是否采用 delta 表示、怎样对齐时间，都要由 action contract 决定。

```text
sample [B,H,d_padded]
  → 按指定adapter取有效action维度
  → 对对应native维度使用train统计量反归一化
  → 明确position/delta/velocity/gripper与frame
  → chunk时间对齐、选择当前motion reference
  → robot controller
```

上面是逻辑职责顺序；实际库可能合并 transforms。Padding removal、统计量和 action conversion 必须在同一维度映射下匹配。10 个 Euler steps 不等于 10 个未来动作，也不等于每秒 10 次物理反馈。

### 第五步：做公平的实验解释

[ML06](../labs/notebooks/ML06_diffusion_flow.ipynb) 的“两个 inference sampler”与“held-out 采样指标”可用于观察一维 conditional field 的采样。[Pi_mechanisms](../labs/notebooks/Pi_mechanisms.ipynb) 主要做已知端点的公式/符号检查，两者证据不同。

在 Colab 做改变步数的实验时，固定模型权重和测试条件，在 validation 上选定设置，再用保留测试集报告。记录采样时间、模式兼容性及失败；不要把 CPU toy timing 当作 VLA 部署延迟。

增加步数可能改善数值近似，同时增加计算；不能保证实际任务效果单调变好，更不能修复错误观测、单位或模型。Flow velocity MSE 不是 action NLL，也不直接等于闭环成功率。使用 flow 目标训练示范动作仍属于 imitation learning，不会自动变成 RL。

### 面试表达：用一次完整数据流回答

**问题：Flow matching 在 VLA 里究竟学什么？**

可以这样组织短答：给定当前多模态条件，我把专家动作 chunk 与 Gaussian noise 构造一条路径；在 noise→data 的线性约定下，监督量是 $A-E$。网络输入中间态、条件与生成时间，输出同 shape 的向量场。部署时没有专家动作，从新噪声出发积分这个场，最后反归一化并按机器人动作接口执行。生成时间与机器人时间是两套时钟，采样步数与执行频率需要分别记录。

三个继续追问：

1. **“既然训练 target 恒定，网络为什么不输出常数？”** 恒定是对某一对端点而言；不同样本/条件有不同 target，网络对当前 $X,c,\tau$ 回归条件场。
2. **“FM 与 diffusion 的 loss 谁低谁好？”** 预测量和尺度不同，应先检查各自目标，再以相同条件下的样本/rollout、时间和接触指标比较。
3. **“换成反向时间只改 target 符号行吗？”** 不行；还要核对端点、时间输入与积分步长符号。

**本课通过标准：** 不查笔记推导路径和 target，算对 98 个有效 scalar，手算一个非恒定模型输出的 Euler 更新，完整说明 sample 到 controller 的职责边界。下一步在 [第 10 课](10_from_vlm_to_vla.md) 比较模型时，优先追这些相同接口，而不是只记模型名字。
