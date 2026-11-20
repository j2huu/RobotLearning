# VLA 面试练习册：从讲清机制到用证据回答追问

[循序学习路线](../planning/vla_learning_path.md) · [第 6–15 课目录](README.md) · [Colab 入口](../labs/notebooks/README.md) · [个人进度](../planning/progress.md)

这份练习册跟着第 6–15 课使用。每学完一课，只完成对应的一张场景卡；第一遍可以开卷，第二遍关掉笔记解释，第三遍更换数字或故障条件。目标是把已经读过的 ML 知识用于解释一条机器人策略的数据流。

下面的题目是课程设计的练习，不声称来自公司的真实面试。数值和小 tensor 均为教学配置；模型特定结论回到对应版本的讲义与原文。已有 notebook 只做过静态检查，个人运行证据要在自己实际完成 Colab 运行后填写。

## 使用顺序与统一评分

每张卡花 15–25 分钟，作为该课最后一个学习单元的一部分：

1. 用 2 分钟画输入、模块、输出与时间轴；先不要看参考骨架。
2. 用 60–90 秒解释核心问题，录音或写下自己的回答。
3. 用 5–8 分钟做推导或故障题，写明假设、shape 与单位。
4. 用 3–5 分钟回答追问；提出能区分两个原因的实验。
5. 对照要点评分，记录一条错因和下一次具体修正。

**四维评分，每项 0–3 分，总分 12 分。** 同一量表用于学习路线的阶段检查。

| 维度 | 0 分 | 1 分 | 2 分 | 3 分 |
|---|---|---|---|---|
| 机制准确 | 核心因果关系错误 | 会说名词，连接关系不清 | 能独立讲清输入、计算和输出 | 能解释为什么这样设计，并给出反例或替代设计 |
| Tensor / 推导 | 轴、运算或公式错误 | 依赖答案才能补出步骤 | 能独立推对给定例子，说明每个轴 | 换尺寸、mask 或时间约定后仍能推导并检查 |
| 训练、推理与系统边界 | 使用部署不可见信息，或混淆单位/时钟 | 知道存在区别，但说不完整 | 能分别列出可见信息、更新参数、物理接口 | 能追踪跨模块依赖及错误如何影响执行 |
| 证据 / 诊断 | 虚构结果，或结论明显超出证据 | 只说“做消融、看 loss” | 提出可执行对照，说明指标与结论范围 | 能分析混杂、失败反例和仍无法排除的解释 |

个人复习时，**总分至少 9 分且每项至少 2 分**，可以进入下一阶段；这只是学习标准，不是招聘通过保证。低分项决定回看哪一步，不要求整课重读。运行尚未完成时，可以用严谨的实验设计得分，但必须标注“计划”，不能将其写成实验结果。

## 第 6 课场景卡：同一个 token 宽度，不同的物理含义

配套：[多模态表示](06_multimodal_policy_representation.md)，用于 6C。

**60–90 秒问题：**“两台相机、语言指令和机器人 state 是怎样进入同一个策略网络的？”

回答骨架：先定义部署时可见的观测及其时间；图像切成 patches，经编码得到 tokens；文字 ID 查 embedding；state 按约定归一化后投影。各支路映射到兼容的 feature width，再通过 attention 等机制交互。Camera、position、type 与有效性信息帮助区分来源。最终 action head 还要把 hidden representation 映射成具有明确语义的动作。

**现场推演：**教学输入为 RGB `[B=2, cameras=2, 3, 32, 32]`，不重叠 patch 为 `8×8`，宽度 `d=32`；文字有 6 个 slots，state 有 1 个 token。请写 image tokens、图文 prefix 和加入 state 后的长度。随后移除一台相机。

检查答案：每台相机有 `(32/8)^2=16` 个 patches；image tokens 为 `[2,32,32]`，图文 prefix 为 `[2,38,32]`，加入 state 为 `[2,39,32]`。只保留一台相机时分别为 `[2,16,32]`、`[2,22,32]`、`[2,23,32]`。这里的 32 有时是 token 数，有时是 feature width，必须根据轴解释；它不是机器人 32 个自由度。

**深入追问与推理要点：**

- 两台相机 concat 后，模型就知道外参了吗？来源标识提供区分，但不保证显式标定或几何理解；要说明训练数据与额外几何输入的作用。
- 缺失 force 全填零为什么可能出问题？真实零力与未测量成为同一数值；需要有效性约定，并检查训练是否覆盖缺失模式。
- 指令变化导致 hidden 变化，是否证明语言有用？还需检查动作是否按指令改变、任务是否完成，控制图像和初态；仅变化不证明变化正确。

**连接个人项目：**拿自己真正处理过的一种 observation 写“原始字段 → 预处理 → token → 有效 mask”四列。若仅分析了 VLA01，就描述为合成数据的接口分析；运行后记录预测 shape 与实际输出的差异。

## 第 7 课场景卡：张量算得通，仍可能读到了答案

配套：[Transformer 与因果性](07_policy_transformer_and_causality.md)，用于 7C。

**60–90 秒问题：**“为什么 action chunk 可以双向 attention，却又必须防止未来信息泄漏？”

回答骨架：先区分共同生成的 noisy action variables、训练 target 和真实未来传感器。联合生成时所有 noisy action positions 都是当前输入变量，可以互相读取；未来测量不是部署条件。Mask 要约束每层的信息路径，同时处理 padding；训练 target 只能沿目标所需的合法路径使用。缓存要检查被缓存表示是否独立于每步变化的变量。

**现场推演：**`B=2, heads=4, N=47, d=32`，每个 head 的 query/key 宽度为多少？Attention score shape 是什么？某个 query 对三个 keys 的 logits 为 `[0, log(3), 5]`，第三个 key 禁止读取，正确权重是什么？

检查答案：每个 head 宽度为 8，scores 为 `[2,4,47,47]`，softmax 沿最后的 key 轴。把第三个 logit 设为负无穷，权重为 `[1/4,3/4,0]`。若只把禁止项设成 0，softmax 后仍有非零权重。具体 bool mask 的 True/False 含义必须按所用 API 核对。

**深入追问与推理要点：**

- 连续分支 A 不直接读 clean target D，为何仍可能泄漏？若第一层 prefix P 读 D，第二层 A 读 P，路径为 `D → P → A`；要检查多层依赖。
- 怎么验证？固定合法输入，只扰动禁止来源；在 `eval` 等可比设置下比较受保护输出。给测试增加两层，避免只验证单层直接边。
- Prefix 不受 noisy action 改变，就已经加速了吗？这只支持缓存所需的信息依赖条件；还需要真实 cache 实现、正确失效策略和耗时测量。

**连接个人项目：**保留一个由自己构造的坏 mask、它产生的受保护输出变化及修复后的变化。VLA01 的 prefix independence 检查没有实现真实 KV cache，面试时说明这一范围。

## 第 8 课场景卡：训练有专家动作，部署时动作从哪里来

配套：[CVAE 与 Diffusion Policy](08_cvae_and_diffusion_policy.md)，用于 8C。

**60–90 秒问题：**“CVAE 与 diffusion 都能生成动作，它们训练和部署时分别看见什么？”

回答骨架：CVAE 训练的 posterior 可以读取专家 chunk，用 latent 帮助重建，并用 KL 约束它与 prior 的关系；部署没有专家 chunk，必须按 prior 的约定提供 latent。原始 ACT 的测试方式是使用零 latent，不能推广为所有 CVAE 的规定。Diffusion 训练把专家动作加噪，以条件、噪声级别和 noisy actions 预测噪声等监督量；部署从噪声开始迭代采样。两者输出的 chunk 都还需要执行窗口与反馈调度。

**现场推演：**采用 epsilon prediction，标量 clean action 为 2，采样噪声为 -1，$\bar\alpha_k=0.64$。写 noisy action 与训练 label。

$$
A_k=\sqrt{0.64}\times 2+\sqrt{1-0.64}\times(-1)=1.0.
$$

检查答案：该样本的噪声监督值为 -1，而不是 2 或 1.0。网络读取 noisy action 1.0、条件与 step；clean action 用于构造训练样本，但不能额外作为部署输入。

**深入追问与推理要点：**

- 训练只抽一个 step，为什么推理还要多步？随机抽样估计跨噪声水平的训练目标；生成时要从起始噪声沿采样过程得到动作，二者计算任务不同。
- `H=20, K=5, action rate=20 Hz`，可供下一次规划的执行窗口通常有多长？按每条命令占一个采样周期，约 0.25 s；若一次推理需 0.32 s，简单同步方案短缺 0.07 s。异步方案仍需说明参考覆盖与 deadline。
- Transformer 与 diffusion 是二选一吗？前者可以是 denoiser 的网络结构，后者定义训练与采样过程。

**连接个人项目：**画自己能解释的训练/推理双图，划掉部署不可见的箭头。ML06 是一维合成生成实验，不能说它验证了真实 Diffusion Policy 操作成功率。

## 第 9 课场景卡：Flow 的 velocity 属于哪一个时间

配套：[Flow Matching](09_flow_matching_for_actions.md)，用于 9C。

**60–90 秒问题：**“Flow matching 输出一个 velocity，为什么不能直接把它当末端速度？”

回答骨架：先规定 noise→data 的生成时间 $\tau$。线性路径连接噪声 E 与训练动作 A，目标为路径对 $\tau$ 的导数 A−E。网络输出与整个 action chunk 同 shape 的向量场，用于更新生成状态。经过采样后才得到 normalized actions，随后反归一化并依据 action contract 执行。生成时间、chunk 的物理时间索引和机器人控制周期是三个不同概念。

**现场推演一：**`A=2, E=-2, tau=0.25`，求路径位置与目标；推理时当前 `X=-1`、模型实际预测 `v=0.8`、步长 `0.1`，下一步是多少？

检查答案：训练位置为 $X_{0.25}=0.75(-2)+0.25(2)=-1$，目标为 4。推理更新为 $-1+0.1\times0.8=-0.92$。不能在推理中用训练终点算出的 4 替代模型预测，因为未知目标动作正是待生成的对象。

**现场推演二：**`[B,H,d_a]=[2,8,7]`；两个样本分别有 8、6 个有效动作，所有有效 scalar 的误差为 2。按有效 scalar 求平均的 MSE 为多少？

检查答案：有效 scalar 数为 `(8+6)×7=98`，平方误差总和为 `98×4=392`，均值为 4。若分母只除有效 waypoint 数 14，则得到 28，改变了 loss 的 reduction 尺度。无效维度也存在时，要进一步应用分维 mask。

**深入追问与推理要点：**反转路径方向时，target、积分起终点和步长符号须一起变；增加 ODE steps 会增加计算量，但不能自动增加新视觉反馈；flow MSE 的监督对象是带时间条件的向量场，不等于直接回归 action 的条件均值。

**连接个人项目：**展示一条自己的 endpoint/sign 检查与一个故意写错的 sampler。已知终点的手算仅验证代数；模型采样效果需独立运行和评价。

## 第 10 课场景卡：比较模型时先固定比较的对象

配套：[从 VLM 到 VLA](10_from_vlm_to_vla.md)，用于 10C。

**60–90 秒问题：**“把一个 VLM 改成 VLA，需要补哪些环节？”

回答骨架：视觉语言知识是初始化和条件能力的一部分；机器人还需要 state/action contract、动作表示与 head、动作监督、适配数据和执行接口。比较原始 OpenVLA、Octo、π0 时，分别说明条件、初始化、动作表示、目标与采样过程，记录所指版本。离散 token action decoding 与连续 flow decoding 是不同输出路径，名称中都有 Transformer 并不能替代这些接口信息。

**诊断题：**模型甲的 action CE 为 2.1，模型乙的 flow MSE 为 0.3，能据此选乙吗？两者最终都输出 `[1,8,7]`，能直接接同一 controller 吗？

检查答案：两种 loss 的随机变量、尺度与 reduction 不同，数字不能直接排序。同 shape 也不保证同语义；要检查这 7 维是关节位置还是 Cartesian motion 加夹爪、absolute/delta、单位、frame、时间戳及归一化。只有完成对应转换，才可在一致的任务协议下比较。

**深入追问与推理要点：**

- “用了 pretrained VLM”是否表示它一直冻结？初始化与优化参数集合是两个问题，查训练配置和梯度。
- 怎样比较两种动作 head？先固定条件与数据，再报告训练预算、采样预算和下游指标；无法匹配的计算成本写出来。
- 换指令输出不变一定是 language encoder 坏了？也可能任务无需区分、数据存在捷径或下游 head 忽略条件，需要能区分这些解释的测试。

**连接个人项目：**制作一张基于具体版本的 model interface 表，标注哪项来自论文、哪项来自源码、哪项仍未知。只读源码时就说“完成接口 trace”，不写成模型训练复现。

## 第 11 课场景卡：把 π0 从架构图讲到一个 batch

配套：[π0 策略系统](11_pi0_policy_system.md)，用于 11C。

**60–90 秒问题：**“请从一个训练样本解释 π0 到底怎样学习动作，再说部署时删去什么。”

回答骨架：当前 image/language/state 提供条件；专家 chunk 经规范化构造 noisy actions 与 flow target；图文 tokens 与专门的 action expert 按依赖 mask 交互；action positions 输出向量场，计算有效目标的损失。部署去掉专家 chunk 和目标构造，从噪声迭代得到动作，再恢复 native action 接口。Action expert 的专门参数和逐层交互不能简化成任意 VLM 最后的一个 MLP。

**现场推演：**假设 `B=2,H=10,d_native=7,d_model_action=32,d_hidden=64`，并假设 loss 仅监督 7 个有效 native 维度，全部 waypoint 有效。写 noisy input、action embeddings、velocity output 与有效 scalar 数。

检查答案：依次为 `[2,10,32]`、`[2,10,64]`、`[2,10,32]`，有效 scalar 数为 `2×10×7=140`。三个宽度分别属于机器人接口、模型对齐的动作空间与 hidden representation。这是一道教学题，真实模型的宽度和 padding/loss 规则必须核对指定配置。

**深入追问与推理要点：**

- Prefix 不读 action，action loss 就不能更新 prefix 网络吗？不成立；action 读取 prefix 的 K/V 时，梯度可能沿这条计算路径反传。冻结、detach/stop-gradient 才是额外的梯度限制。
- 固定 observation 的多步采样与重新获取 observation 有什么区别？前者改生成变量，后者改策略条件并可能使缓存失效。
- 生成 10 个 actions、做 10 次 Euler 更新、10 Hz policy 分别是什么？分别是 horizon、数值采样步数、物理查询频率，不能互相推导。

**连接个人项目：**从 [P5](pi_series/05_openpi_engineering.md) 的固定版本阅读路径中选择一个真实函数，说明输入来自哪里、输出给谁以及训练/推理差别。VLA01 的小型 shared Transformer 不包含完整 π0 expert 权重结构。

## 第 12 课场景卡：先定位 tokenizer 错误，再讨论 policy

配套：[FAST 与 π0.5](12_fast_and_pi05.md)，用于 12C。

**60–90 秒问题：**“FAST、π0-FAST、π0.5 之间是什么关系？”

回答骨架：FAST 定义动作的编码/解码表示；π0-FAST 是预测这类离散序列的 policy；π0.5 v1 的训练配方包含异构数据与离散/连续目标，并使用高层子任务与低层动作的分层推理。最终低层 flow 路径不应画成“先生成完整 FAST action，再由 flow 逐项精修”。应先说明自己讨论的是表示、训练阶段还是实际推理路径。

**诊断题：**真实动作 `[0,1]` 经正确 tokenizer round trip 得 `[0,0.9]`；policy 预测 tokens 解码后得 `[0,0.7]`。计算相对真实动作的两项 MAE；怎样区分误差来源？

检查答案：round-trip MAE 为 0.05，policy 解码结果相对真值的 MAE 为 0.15。第一项说明表示本身已经有误差；第二项还包含预测错误。要直接检查正确 tokens、预测 tokens 及其解码结果，不能一般性地把总 MAE 减去 codec MAE 当成独立可加的 policy 误差分量。

**深入追问与推理要点：**

- DCT 本身一定丢信息吗？完整精确系数的变换可逆；量化等处理引入不可逆误差。具体 codec 还需固定序列化和词表。
- Token 序列长 12 就有 12 个物理动作吗？没有这个必然关系；物理 horizon 与符号长度由编码过程联系。
- Teacher forcing 为什么训练可并行、推理通常逐个生成？训练位置看到已知正确前缀，部署前缀来自此前预测。Label shift 与 causal mask 必须匹配。
- 离散与连续分支共享训练时要检查什么？Clean discrete targets 不能直接或经中间 prefix 泄漏给不应读取答案的连续分支。

**连接个人项目：**画三条不同的数据流：codec round trip、AR prediction、flow sampling。若自己只写了 DCT 加 rounding，就明确它是 codec 子步骤实验；完整 FAST 还涉及官方编码方案与词表。

## 第 13 课场景卡：动作平滑，为什么接触仍然失稳

配套：[动作与控制接口](13_action_and_control_interfaces.md)，用于 13C。

**60–90 秒问题：**“系统输出 1 kHz 命令，是否说明它每毫秒都能根据新观测做一次力控决策？”

回答骨架：先把policy查询、waypoint、force-policy、传感器到达、插值与实际控制律分别计时。历史用户说明的1kHz插值与论文内部1kHz控制律分别记录；当前所选peg代码只有planner/fast目标50Hz等应用层证据，不能从其中一项推断其余各项。详见 [源码时序](03d_fp2_source_architecture_and_training.md)。

**现场推演：**同一表达坐标轴下，参考点从旧点移到新点，位移 `r=[0.1,0,0] m`，力 `f=[0,10,0] N`，旧点力矩为零。新点力矩是多少？

$$
m_{new}=m_{old}-r\times f=[0,0,-1]\ \mathrm{N\cdot m}.
$$

检查要点：先声明 r 的方向为旧点指向新点；力矩随参考点改变。只旋转力向量、遗漏参考点偏移，可能让数值看似合理但物理意义错误。

**深入追问与推理要点：**

- 同样的 delta pose 为什么得到不同轨迹？检查 delta 相对于当前 state、上条 command 还是 chunk 起点，及旋转的乘法约定。
- 输出已经平滑，为何新 chunk 仍跟错？新 chunk 可能因推理延迟而过时；平滑性与时效性是两个指标。
- 怎样区分 force sensor 延迟和反馈增益问题？先固定 reference 与输出时钟，分别改变 delay 与 gain，记录实际可用测量的 age 和响应，不同时改两项。

**连接个人项目：**使用 [VLA02](../labs/notebooks/VLA02_multirate_force_execution.ipynb) 时，记录输出频率相同条件下的反馈差异。它是规则式 spring 接触实验；若尚未运行，当前证据只是一份实验设计。

## 第 14 课场景卡：FP2 的 compressor 到底压缩谁

配套：[力觉适配](14_force_aware_adaptation.md)、[论文tensor详解](03c_fp2_tensor_walkthrough.md)、[3d源码精读](03d_fp2_source_architecture_and_training.md)，用于14B/14C。

**60–90 秒问题：**“RFM 没有 force input，FP2 怎么利用力觉？Compressor 是不是由力产生 query？”

回答骨架：RFM 输出原始动作 chunk 与 context。Compressor 用 learned query 从 context 的 K/V 聚合压缩表示，并缓存供后续使用；这个步骤不读取实时 wrench。Fast force policy 结合压缩 context、近期 wrench 与 proprioception，输出 interaction frame、六维选择 mask 和期望 wrench。Hybrid controller 结合这些参数、处理后的 RFM motion reference 与实际反馈，形成执行命令；RGB 不直接送进该 controller。

**现场推演：**教学 context 为 `[B=2,N=64,D=128]`，先投影到 `d=32`；4 个 heads、单个 learned query。写 Q/K/V、attention weights 与压缩输出 shape。

检查答案：分 head 后 `Q=[2,4,1,8]`，`K,V=[2,4,64,8]`；weights 为 `[2,4,1,64]`，对 64 个 context positions 做 softmax；合并 heads 后为 `[2,1,32]`。Q 的来源是可学习参数，K/V 的来源是 context。题中宽度仅为教学值，不能据此声称 FP2 的真实隐藏层配置。

**深入追问与推理要点：**

- 输出是一个力控参数吗？应分别解释 frame、mask、desired wrench，且区分 desired 与 measured wrench。六维 mask/wrench 的语义明确，frame 的 raw encoding 需核对来源。
- Force loss 会更新 compressor 吗？按本课程所述 FP2 阶段，compressor 重建训练后冻结，decoder 丢弃；后续 force learning 不反传重塑它。
- 为什么重建 context 好仍可能力控不好？重建目标与接触任务目标不同，压缩可能丢掉下游关键信息；还要评价 force-policy 与 controller 的闭环效果。
- Force regulation 能修正选错物体吗？通常没有相应高层职责；目标/粗运动明显不对时，快速接触调节不能保证补救。

**2026-11-09源码追问：** 限定peg/RLT1024/no-wrist/no-action配置，再回答：

- `[B,N,2048]`为什么只取image positions仍能有语言条件？在VLM上下文化后再切片，读取依赖由上游mask决定。
- 快策略是否也用cross-attention融合？本例用FiLM→两层GRU→FiLM，再将history/context两路gating；compressor的cross-attention不能代替这个答案。
- raw frame/force/mask各几维？9/6/6；frame反变换成7D pose，mask阈值后还可能被设备层清除旋转轴。
- target stop-gradient足够吗？不够，还要看输入路径和optimizer filter；alpha=0只训RLT，alpha>0有额外VLA action-loss更新分支。
- 同一demo比较能把收益都归给压缩器吗？不能，需区分任务微调、额外force监督、实时反馈、controller和readout结构。

**连接个人项目：**写清自己分析或实现过的是 context compression、force-policy 数据、控制接口还是完整闭环。ML05 的合成 compressor 与 VLA02 的规则 controller 分别展示机制，把两份 notebook 拼在一起仍不能声明复现了真实 FP2。

## 第 15 课场景卡：改进看起来明显，怎样判断证据够不够

配套：[VLA 评估](15_vla_evaluation_and_generalization.md)，用于 15C。

**60–90 秒问题：**“加 force 模块后 success 从 80% 到 90%，你会怎样判断它是否有效？”

回答骨架：先核对成功定义、trial 数、测试初态与独立单位；检查 backbone、数据、训练预算与 controller 是否一起改变。再按目标评价任务成功、接触误差/峰值、延迟及失败类型。使用配对初态或明确的重采样协议，报告不确定性与任务分项；单个 aggregate 数字不足以归因，也不证明所有 OOD 条件都改善。

**诊断题：**方法 A 成功 16/20 次，B 成功 18/20 次；两者是否使用同一组初态尚不清楚。可以确认什么，还缺什么？

检查答案：可确认记录中的样本成功率分别为 80% 与 90%，绝对差 10 个百分点，即两次成功的差别。仅凭这些汇总不能判断稳定性或因果归属。需要 trial 协议、任务/场景组成、是否配对以及各 trial 结果；配对比较还需要知道哪些初态上 A/B 结果不一致。相关帧不能当成独立机器人试验来扩大样本量。

**深入追问与推理要点：**

- 只改善新背景，能说物理泛化了吗？应将视觉背景变化与摩擦、刚度、质量变化分别定义测试。
- Force shuffle 后变差，是否证明模型以正确方式用 force？它支持依赖性线索，但可能引入分布外输入；还需原生缺失模式、时间延迟和任务合理的反事实对照。
- Offline loss 下降，成功率反而下降怎么查？依次检查接口、分布偏移、执行时延和关键接触阶段；提出一项最小干预区分原因，而不是立即扩大模型。

**连接个人项目：**选一个真实发生的失败，写“观测现象 → 两个候选原因 → 区分实验 → 结果 → 剩余局限”。没有结果时停在实验设计，保留待运行状态。

## 25 分钟模拟面试

先完成一个阶段的场景卡再模拟；不必等第 15 课全部读完。每次选择一个已学主题和一个相邻主题，避免一次抽取所有知识。

| 时间 | 内容 | 检查重点 |
|---|---|---|
| 0–3 分钟 | 讲一个自己完成的分析或实验 | 问题、个人动作、可查证据、局限 |
| 3–7 分钟 | 一张场景卡的短答及追问 | 输入输出、因果关系、具体设计 |
| 7–13 分钟 | 换一组数字做 tensor / 公式题 | 每个轴、广播、mask、符号与单位 |
| 13–19 分钟 | 给一个故障，提出两种原因和区分实验 | 控制变量、可测指标、预期差异 |
| 19–22 分钟 | 英文复述同一个机制 | 事实与中文一致，句子清楚 |
| 22–25 分钟 | 按统一量表评分，写两项修正 | 一个机制问题、一个证据或表达问题 |

英文可轮流练：`Walk me through one training batch.`、`What changes at inference time?`、`Which experiment would distinguish these two explanations?`。先能用中文准确推导，再练英文；不以术语数量或语速替代内容。

## 复习节奏与卡住时的处理

采用一个可调整的节奏：当天关卷复述；隔 2–3 天换数字复做；约一周后把两课串起来。时间间隔只是学习安排，不作为已验证的最优学习配方。已经稳定掌握的题减少频次，错误集中在哪个维度，就回该维度对应的小步骤。

| 卡住的表现 | 回去做什么 | 再次检查 |
|---|---|---|
| 只记住模型名 | 只画输入、输出与模块间箭头 | 去掉模型名字仍能说明职责 |
| Tensor 频繁错 | 给所有轴命名，再用极小数字手算 | 改 batch 或 horizon 后仍正确 |
| 训练与推理混淆 | 并排列可见输入，划掉部署不可见的信息 | 能指出一个信息泄漏反例 |
| 一谈效果就只说 loss | 写任务指标、控制变量和失败分类 | 给出能推翻自己解释的结果 |
| 面试说得太散 | 按问题→机制→例子→边界四句重写 | 90 秒内说完且可接受追问 |

不确定某个精确配置时，可说：“我能确定这个接口的职责；该版本的 hidden width / history length 我还没有核对，会查配置和 tensor trace，不用教学数字代替。”接着讲已确认的机制及核对方法。问题本身有歧义时，先声明假设，例如“这里我把 action 定义为 TCP delta pose，expressed in base frame”，再推导。

## 可复制的个人证据记录

```text
日期 / 课号与小单元：
本次问题及我采用的假设：
我的 60–90 秒回答：
手推或 tensor 题结果：
四维分数：机制 __ / tensor __ / 边界 __ / 证据 __ （总分 __ /12）
一个最关键的错误及其原因：
纠正后能解释的反例：

证据类型：论文阅读 / 源码 trace / 手算 / 实验设计 / 本人已运行
依据的文件、模型版本或源码 revision：
Colab notebook / 本人运行日期 / runtime / seed / 修改项：
观测结果或输出链接：（未运行则写“待运行”）
该结果支持什么：
该结果不能支持什么：
个人完成部分 / 使用的已有代码 / 团队结果：

下一次复习日期或顺序：
下一次具体改动或追问：
```

最终面试材料应能从一句结论追到一张图、一个计算步骤或一条运行记录。先把第 6/7 课的输入与信息流讲清，再加入第 8/9 课的生成过程，随后接上模型、控制与评估，逐步形成一条完整且有证据的数据流叙述。
