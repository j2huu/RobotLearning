# Pi / VLA 面试题卡：短答、追问与评分

配套 [P1–P6](README.md)。这是课程练习题，不声称来自任何公司的真实面试。先限时回答，再看参考要点；每题举一个自己的实验或机器人例子。

## 1. 架构与生成模型

### Q1：用 90 秒解释 π0

**短答要点**：以预训练 VLM 提供图文条件，用按 token 类型划分权重的 action expert，通过 flow matching 生成连续 action chunk；配合跨机器人预训练与任务 post-training。执行还需要动作转换、调度与低层 controller。

**追问**：action expert 在哪几层与 VLM 交互？为什么不用每个 action scalar 一个词表 token？

**失分点**：只说“VLM + MLP”；把 VLM 初始化说成永久冻结；把 50 Hz 当成 VLM 查询率。出处：[P1](01_pi0_architecture.md)。

### Q2：Action token 一定是离散 ID 吗？

**短答要点**：不是。π0 的 noisy action vector 经 projection 后占一个序列位置，属于连续 embedding；FAST action token 才对应离散符号。需要分别说明 sequence axis 和 vocabulary。

**追问**：50 个动作、32 维，π0 expert 输入有 50 个还是 1600 个 action positions？答案是 50 个，每个位置由 32 维向量投影。

### Q3：为什么 chunk 双向 attention 不一定作弊？

**短答要点**：所有 noisy future action positions 是共同生成的变量，可以互相建模；不可用的是部署时未获得的未来观测或 clean target。标签进入 loss，不等于能进入 conditioning。

**追问**：未来 measured force 与未来 predicted action 的区别是什么？出处：[P1](01_pi0_architecture.md)、[第 3 课](../03_slow_fast_contact_policy.md)。

### Q4：现场推导 flow target

**题目**：$x_s=(1-s)A+s\epsilon$，写 target、起点、终点和 Euler 更新。

**参考**：$u=\epsilon-A$；从 $s=1$ 的噪声到 $s=0$ 的数据；$\Delta s=-1/K$，$x\leftarrow x+\Delta s\,v_\theta$。

**追问**：若反转时间，哪些符号必须同时改变？为什么训练不需要完整运行 K 步？出处：[P2](02_flow_matching.md)。

### Q5：Velocity 是关节速度吗？

**短答要点**：这里是 action tensor 相对生成时间的变化率；物理 action 本身才可能是 joint position/velocity 等。生成时间、chunk 时间位置、控制器时钟分别定义。

**追问**：把第二个 ODE step 直接发给机器人有何问题？

### Q6：增加 flow steps 一定更好吗？

**短答要点**：可能减少数值误差，但增加时延，不能修复模型和接口错误；闭环中更慢的响应可能抵消收益。要比较相同数据/条件下的采样质量、耗时和 rollout。

**追问**：如何设计 5/10/20 steps 对照，防止把动作噪声差异误当成算法收益？固定 observation/noise 做离线配对，再用一致 reset protocol 评估闭环。

### Q7：Diffusion、flow 与 AR 怎样比较？

**短答要点**：依次比较概率路径、监督量、输出表示、采样/解码循环、条件接口和 latency。不能只说“flow 连续，diffusion 离散”或“AR 训练不能并行”。

**追问**：相同步数是否代表相同 FLOPs/耗时？出处：[P2](02_flow_matching.md)、[P3](03_fast_action_representation.md)。

### Q8：FAST 是什么？

**短答要点**：动作 tokenizer，把 chunk 经时间维 DCT、量化、序列化与压缩编码成离散序列。π0-FAST 是使用它的 policy。压缩比、codec error 和 AR 解码耗时都要测。

**追问**：DCT 和量化哪一步造成不可逆误差？训练 teacher forcing 与推理有哪些差异？出处：[P3](03_fast_action_representation.md)。

## 2. π0.5、泛化与实验

### Q9：π0.5 相比 π0 的关键变化

**短答要点**：v1 使用 MM/ME/CE/HL/WD 异构离散预训练，再加入 flow expert、VI 与相关数据 post-training；同一模型支持 subtask 文本和低层动作；state tokenization 与 time conditioning 也变化。

**追问**：为什么不能只归结为“更大模型”或“更多 web data”？出处：[P4](04_pi05_generalization.md)。

### Q10：把 π0.5 的分解式写出来

**参考**：$p(A,\hat\ell\mid o,\ell)=p(\hat\ell\mid o,\ell)p(A\mid o,\hat\ell)$。最后一项不再直接条件于总任务，是该模型设计的假设。

**追问**：高层预测错，低层很准确，会发生什么？same model 是否意味着 same frequency？

### Q11：Web data 对 π0.5 的作用有多大？

**短答要点**：分协议回答。v1 Fig. 10 的 aggregate progress 中 no-WD 差异不显著；Fig. 11 的 OOD objects 和 Fig. 13 的高层任务更体现 WD 的收益。

**追问**：为什么“没有显著差异”不等于“证明没有作用”？样本量、统计不确定性、任务组成与指标可能不同。

### Q12：HL 与 VI 有何区别？

**短答要点**：HL 对机器人观测标注 subtask；VI 是人通过语言命令引导现有低层 policy 的实时高层示范。它们提供的状态分布和监督接口不同。

**追问**：VI 为什么不能简单当作 action-level DAgger？需要说明查询的标签是高层命令，以及真实收集/聚合流程。

### Q13：Attention 隔离是否等于 gradient 隔离？

**短答要点**：不等于。action query 读 prefix 的 K/V 时，action loss 可以反传到产生 prefix 的参数。stop-gradient 是额外的 backward 路径选择。

**追问**：画出 C→P 的 forward 读取与 loss→P 参数的 backward 路径，再说明 KI 与冻结 backbone 的差别。出处：[P4](04_pi05_generalization.md)、[P6](06_vla_foundations_and_capstone.md)。

### Q14：为什么 FAST targets 不能被 flow branch 读取？

**短答要点**：FAST targets 编码了 clean actions，会形成部署不存在的答案捷径。还需阻断通过 prefix 的间接路径。

**追问**：如果直接 mask 看起来正确，怎样写 perturbation test 查两层泄漏？出处：[机制实验](../../labs/pi_series_mechanics.py)。

### Q15：论文里的 80% 到底是什么意思？

**短答要点**：先查 rubric，可能是 task progress、对象比例或完整 episode success。还要说 trial 数、任务/场景、初始化、是否见过训练数据。

**追问**：Dishes in Sink 得 6/8 分，能说 75% 完整成功率吗？不能，那是这一 trial 的 progress。

### Q16：怎样证明是场景多样性带来的改善？

**短答要点**：在固定样本/训练预算下改变 scene coverage，按 scene 隔离测试，并报告对象、任务、embodiment 等共同变化。重复多次评估，说明无法完全排除的混杂。

**追问**：一家庭 10,000 帧与 100 家庭各 100 帧，哪些信息不同？出处：[P4](04_pi05_generalization.md)。

## 3. 数据、训练与系统排错

### Q17：新机器人接入 VLA，第一步是什么？

**短答要点**：写 observation/action contract，确认 camera、state/action 维度与语义、units、frame、timestamp、controller、episode 和 split；再接 transforms 与模型配置。

**追问**：两个 `[50,7]` 的 actions 为什么可能完全不兼容？出处：[P5](05_openpi_engineering.md)。

### Q18：低 loss、真机反向动，怎么查？

**短答要点**：首先检查 joint order、单位、action sign、absolute/delta、frame 和 denormalization；固定样本做 round trip，再检查标签时序和训练。

**追问**：归一化后数值都在合理范围，能排除接口错误吗？不能。

### Q19：LoRA 为什么省参数，为什么不保证省同等显存？

**参考**：$\Delta W=(\alpha/r)BA$，可训练参数为 $r(d_{in}+d_{out})$。activation、其他可训练层、optimizer state 和实现开销仍影响显存。

**追问**：2048×2048、rank 8 的参数量是多少？32,768，对比 4,194,304。出处：[P6](06_vla_foundations_and_capstone.md)。

### Q20：DDP、FSDP、gradient accumulation 各解决什么？

**短答要点**：DDP 同步多个模型副本梯度；FSDP 分片模型相关状态；累积用多个 microbatch 形成 effective batch。它们处理不同成本。

**追问**：显存不够时，减 microbatch、checkpointing、LoRA 和 FSDP 各对哪部分有影响？先估算并 profiling。

### Q21：20 Hz 的控制为什么不等于 20 Hz 的视觉闭环？

**短答要点**：command rate、controller feedback rate、感知刷新、模型查询和传输延迟分别计时。chunk 内没有新视觉反馈时，底层可以闭环跟踪旧参考，policy 仍在开环执行。

**追问**：50 Hz，执行 10 个动作，推理 260 ms，预算够吗？200 ms 窗口不足；需处理 deadline/队列，不可仅声称异步后问题消失。

### Q22：K/V cache 什么时候失效？

**短答要点**：被缓存 hidden states 依赖的输入或模型权重变了，就需重算；跨 ODE step 与跨 observation 的缓存不同。pi05 的 state 在 prefix 中也影响 cache。

**追问**：同一图像、同一 prompt，只变 noisy action，为什么 prefix 可以有效？出处：[P1](01_pi0_architecture.md)、[P5](05_openpi_engineering.md)。

### Q23：怎么证明策略使用了语言或 force？

**短答要点**：固定其他条件交换指令/目标；对 force 做因果遮蔽、shuffle 和 contact-phase 分段评估；报告 rollout 与控制指标。attention 权重可辅助分析，不能单独证明因果利用。

**追问**：直接置零 force 为什么可能产生 OOD 输入？如何构造更合理的对照？

### Q24：MSE、NLL、CE、flow loss 能直接排序吗？

**短答要点**：不同目标、尺度、单位和 reduction 的数值不能直接横比。比较各自拟合诊断和一致 protocol 的下游指标。

**追问**：flow 训练也是 MSE，为什么不等于回归 action mean？因为目标与条件是时间相关的向量场。出处：[P2](02_flow_matching.md)。

## 4. Robot Learning 与研究延伸

### Q25：BC、DAgger、offline RL 的区别

**短答要点**：BC 拟合固定示范；DAgger 在学习者到达状态获得专家标签并聚合；offline RL 使用固定 reward/transition 数据优化回报，同时面对 action support 风险。

**追问**：为什么生成式动作模型仍会有 covariate shift？出处：[P6](06_vla_foundations_and_capstone.md)。

### Q26：VLA 与 world model 的区别

**短答要点**：policy 在条件下生成动作；forward/world model 预测环境或 latent 后果，可用于规划。未来 action chunk 不是未来世界状态。

**追问**：MPC 里谁评估候选动作，为什么执行前缀后要重新规划？

### Q27：FP2 怎样接在 π0.5 下游？

**短答要点**：保持 foundation motion generation，压缩缓存 context；结合实时 wrench/proprio history 预测 interaction frame、selection mask、desired wrench，交给 hybrid controller。fast module 的职责是 interaction regulation。

**追问**：哪些错误靠它修不了？高层目标错误、严重几何偏差或不可行粗轨迹。出处：[第 3b 课](../03b_fp2_adaptor_twist_wrench.md)。

### Q28：Twist/wrench 变换为什么不能相同？

**参考**：$\xi'=Ad_T\xi$、$W'=Ad_T^{-\top}W$，保持 $W'^\top\xi'=W^\top\xi$ 的功率不变。要说明分量顺序、frame 和参考点。

**追问**：F/T sensor origin 到 TCP 有偏移，为什么 moment 会变化？

### Q29：Jacobian 在 action interface 中有什么用？

**短答要点**：$\xi=J(q)\dot q$ 把关节速度映射到末端 twist；逆向求解需考虑奇异性、冗余、约束和正则化。

**追问**：策略输出 Cartesian delta 时，谁负责 IK、限位与低层跟踪？出处：[P6](06_vla_foundations_and_capstone.md)。

### Q30：讲一个自己完成的 VLA 项目

**结构**：问题与失败触发 → 自己实现的部分 → baseline/控制变量 → 结果与证据 → 失败/局限 → 下一步。所有数字能对应运行记录。

**追问**：如果去掉你的方法，性能怎样？数据泄漏怎么排除？哪部分来自现有仓库或团队？

**失分点**：把源码阅读说成训练复现，把 toy 指标说成真机成功率，把团队结果说成个人实现。

## 5. 评分与模拟面试

每题四项各 0–2 分：概念准确、机制/推导、取舍/边界、代码或实验依据。**至少 6/8，且概念准确不能为 0** 才视为通过。卡住的题回到对应课做具体练习，不只重背短答。

一次 45 分钟模拟：5 分钟项目介绍、10 分钟 π0/π0.5、10 分钟现场推导/shape/mask、10 分钟系统故障设计、10 分钟实验和英文追问。

英文练习选四题：Q1 架构、Q4 flow、Q9 配方、Q30 项目。先中文说完整，再用英文讲相同事实；语速和术语数量不替代准确性。
