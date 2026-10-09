# Robot Learning 后续课程：第 4–19 课

制定日期：2026-10-09。学习投入：每周 8–10 小时。沿用已经学习的第 1–3 课，不重新从头开始。

本文件是接下来的执行计划；[VLA 与 π 系列路线](VLA_PI_STUDY_ROADMAP.md) 保留为主题参考。课程按知识依赖排序，以本地论文做案例，以小实验检查理解，以英文口述连接实习和面试。

目录约定：课程、计划、进度、资源索引与实验都放在 `notes/` 下；根目录 README 提供入口。第 4 课保留原编号，当前从 [第 5 课](05_training_data_and_observation.md) 接着准备，后续沿用第 6–19 课。

参考用户提供的两个仓库后，补入 dataset/benchmark/simulation、异构 policy pretraining 和动作 tokenization 的对照；映射见 [资源地图](RESOURCE_MAP.md)。

## 1. 当前起点与目标

已经讲解过：BC、多峰动作、action chunk、ACT/CVAE、DP/flow/FAST 的基本差别、RDP/ImplicitRDP、Force Policy、FP2、twist–wrench 和混合力位控制。这些内容已接触，不等于已通过独立推导或实验验收。

当前 22 篇本地论文集中于力觉、顺应控制、多模态与 force-aware VLA。下一阶段重点补齐支撑它们的概率建模、神经网络优化、序列建模、RL、模型学习、数据工程和实验评估。

结束时应能独立完成：

1. 从条件动作分布推导 BC 的 MSE、Gaussian NLL、CE、CVAE、diffusion 和 flow 目标，说明各自假设。
2. 画出 VLA 从 image/language/state 到 action chunk 的计算图，并说明 attention mask 与 inference schedule。
3. 判断一个力觉方法是在改善表示、生成 motion、调节 wrench，还是依靠低层 controller 实现闭环。
4. 建立数据 schema，处理单位、坐标系、时间戳、chunk、归一化与 episode split。
5. 区分 BC、DAgger、online/offline RL 与 model-based planning 的信息来源和优化目标。
6. 用成功率、接触力、延迟、失效模式和泛化测试评价模型，而不是只看 training loss。
7. 用英文讲清 FP2 的职责边界，并以准确措辞区分自己完成的工作、团队贡献与论文结果。

8 周用于建立上述主干和实验习惯；RL、world model 与完整 foundation-model 训练还需要后续专题深挖。

## 2. 整体依赖

```text
条件概率 / 风险最小化 / 优化
             ↓
数据与 POMDP ──→ BC / DAgger ──────────────→ RL / offline RL
             ↓                                      ↓
表示学习 / 多模态 ──→ Transformer              dynamics / MPC
             ↓                                      ↓
CVAE / diffusion / flow ──→ VLM → VLA → π0 / FAST / π0.5
                                      ↓
动作接口 / 力觉融合 / fine-tuning / FP2
                                      ↓
数据 → loss → inference → controller → rollout → evaluation
```

力控专题贯穿整条路线：每引入一个 ML 概念，都回到 RDP、ImplicitRDP 或 FP2 检查它如何影响实际机器人执行。

## 3. 每周时间安排

| 活动 | 时间 | 固定产出 |
|---|---:|---|
| 两次课程 | 3 小时 | 一次重理论/推导，一次重论文与系统 |
| 代码实验 | 3–4 小时 | 可运行脚本、配置、指标、观察与解释 |
| 定向阅读 | 1–2 小时 | 一篇主读；第二篇只查对照章节 |
| 英文复盘与面试 | 1 小时 | 90 秒定义、5 分钟架构说明、一次追问 |

建议把两次课程间隔 2–3 天。时间不足时先完成本周主实验和验收，把选读后移。某一项验收未通过时，在下次课前用 20–30 分钟补做，不重复整篇论文。

## 4. 八周课程表

| 周 | 两次课程 | ML 背景与关键问题 | 主读/对照 | 本周代码产出 |
|---|---|---|---|---|
| 1 | **4：概率视角的 BC**；**5：训练、数据与观测** | conditional distribution、MLE/NLL、MSE、梯度/Adam、正则化；POMDP、history、episode split、时间对齐。为什么低 loss 不保证 rollout 成功？ | [第 4 课讲义](04_probabilistic_bc_and_training.md)；[CR-DAgger](../NeurIPS-2025-compliant-residual-dagger-improving-real-world-contact-rich-manipulation-with-human-corrections-Paper-Conference.pdf) 的问题定义 | **Lab 01**：MSE / Gaussian / mixture policy 对比；**Lab 02 起步**：数据字段、split、normalization |
| 2 | **6：多模态表示学习**；**7：Transformer 与因果性** | supervised/self-supervised、reconstruction、contrastive、bottleneck、shortcut；Q/K/V、cross-attention、causal mask、cache。怎样确认 policy 使用了 force？ | [Making Sense of Vision and Touch](../1810.10191v2.pdf)、[FACTR](../2502.17432v2.pdf)；回看 [ImplicitRDP](../2512.10946v1.pdf) 与 FP2 compressor | **Lab 03**：future-force perturbation 验证 mask；modality ablation/force shuffle |
| 3 | **8：CVAE 与 Diffusion Policy**；**9：Flow Matching** | latent、reparameterization、KL/ELBO；forward noise、noise prediction、DDIM；path、velocity target、ODE。生成时间与机器人时间如何区分？ | [ACT 项目](https://tonyzhaozh.github.io/aloha/)、[DP 项目](https://diffusion-policy.cs.columbia.edu/)、[Flow Matching](https://arxiv.org/abs/2210.02747)；回看 RDP tokenizer | **Lab 04**：在同一个小型多峰数据集上实现 diffusion 与 flow；对比采样步数、分布和耗时 |
| 4 | **10：从 VLM 到 VLA**；**11：π0 完整架构** | image tokens、language tokens、projector、预训练/迁移、cross-embodiment；prefix/suffix、action expert、state/time embedding。语义怎样条件化连续动作？ | [Open X](https://robotics-transformer-x.github.io/) 作数据背景；[Octo](https://octo-models.github.io/) / [OpenVLA](https://openvla.github.io/) 作架构对照；[π0](https://arxiv.org/abs/2410.24164) 主读 | **Lab 05 起步**：画 tensor/attention 图；阅读固定版本 openpi，追一次 observation → loss |
| 5 | **12：FAST 与 π0.5**；**13：动作与控制接口** | action tokenization、CE/autoregression、co-training、subtasks；joint/Cartesian、absolute/delta、rotation、force/TCP frame、multi-rate schedule | [FAST](https://arxiv.org/abs/2501.09747)、[π0.5](https://arxiv.org/abs/2504.16054) 的指定章节；[AVP](../2409.14440v2.pdf) 与 Force Policy 作接口对照 | **Lab 02 完成**：chunk、mask、归一化 round trip、frame transform；**Lab 06 起步**：slow-fast 调度实验 |
| 6 | **14：力觉融合与 foundation adaptation**；**15：泛化与实验设计** | early/late fusion、distillation、curriculum、LoRA vs sidecar、冻结策略；OOD、matched ablation、多个 seed、force/time 指标 | [FP2](../ICRA27_2601_MS.pdf) 主读；[CRAFT](../2602.12532v1.pdf) / [FAVLA](../2602.23648v1.pdf) 二选一主对照；FD-VLA 与 CompliantVLA 查表 | **Lab 06 完成**：cached context + fast force feedback + transition；形成五组消融与失败分析 |
| 7 | **16：RL 基础与 Actor-Critic**；**17：Offline RL、DAgger 与 Residual Learning** | return、V/Q、Bellman、TD、policy gradient、actor/critic；offline support、extrapolation、expert correction、residual objective。示范与 reward 各提供什么？ | [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) 的 IL/RL/offline 章节；[原始 DAgger](https://proceedings.mlr.press/v15/ross11a.html)；[CR-DAgger](../NeurIPS-2025-compliant-residual-dagger-improving-real-world-contact-rich-manipulation-with-human-corrections-Paper-Conference.pdf) 主读 | **Lab 07**：小环境 Bellman/TD；同环境 BC vs DAgger；offline action support 小反例 |
| 8 | **18：Dynamics、MPC 与 World Models**；**19：代码闭环与面试整合** | learned dynamics、prediction rollout、model error、receding horizon；world model 与 action policy；data → loss → inference → execution | CS285 model-based 章节；[openpi 官方代码](https://github.com/Physical-Intelligence/openpi)；[FP2](../ICRA27_2601_MS.pdf) 综合复盘 | **Lab 08**：小型 dynamics + MPC；**Lab 05 完成**：openpi 数据流报告；FP2 capstone 与英文模拟面试 |

第 10 课中的 RT-1/RT-2 只用于解释历史和设计选择，不要求当周精读整个演化链。第 12 课只精读 FAST 的 tokenizer 和 π0.5 的 co-training/hierarchy 相关部分。第 16–18 课的代码均用小环境，SAC 全量复现与大型 world-model 训练列为进阶任务。

第 10 课加入 HPT 的 heterogeneous inputs/shared trunk 对照；第 11/13 课加入 RDT 的 diffusion head/unified action space 对照；第 8/12 课加入 VQ-BeT 的 learned quantization 对照。第 15 课加入 robosuite/SimplerEnv 的 observation/action/controller contract、sim-to-real 与 domain-randomization 评估。均为指定章节对照，保持原课号与每周时间投入。

## 5. 每课验收

| 课 | 不看笔记需要做到 |
|---|---|
| 4 | 推导 MSE 对应固定方差 Gaussian NLL；解释 conditional mean 为何可能不是专家动作；说清 μ/σ/I 与条件 |
| 5 | 定义 state/observation/history；解释 frame split 泄漏；正确处理 timestamps、训练集统计和 padding mask |
| 6 | 提出证明 force 有用的消融；区分 reconstruction、contrastive 与 policy objective；解释力觉 shortcut |
| 7 | 算出 attention 中每个 tensor 的 shape；画 causal mask；说明 cache 的有效期与 stale context |
| 8 | 写出 CVAE reconstruction+KL 和 diffusion noise-prediction loss；说明训练/推理的信息差异 |
| 9 | 推导线性路径 velocity target；手算一次 Euler step；说明积分步数与反馈频率是两个问题 |
| 10 | 比较 Octo/OpenVLA 的条件、动作输出和 loss；解释 VLM knowledge 与 physical skill 的区别 |
| 11 | 从 image/state/action noise 一路解释到 π0 action chunk；写出 loss 中所有量的来源 |
| 12 | 区分 FAST tokenizer、π0-FAST 和 π0.5；解释 token CE 与 flow loss，以及 co-training 如何服务泛化 |
| 13 | 给一条 action 定义单位、frame、绝对/增量、执行频率；正确变换 wrench 与 twist；设计 chunk transition |
| 14 | 为同一任务比较 FP2 sidecar 与 force-conditioned action expert；说明训练数据、频率、职责和局限 |
| 15 | 设计控制数据量/计算量的公平实验；区分新对象、场景、物理参数、任务和 embodiment 的泛化 |
| 16 | 从 reward 推导 return、Bellman target 和 actor objective；解释 critic 的作用与 IL/RL 的差别 |
| 17 | 区分 DAgger 的 expert label 与 RL 的 reward；解释 offline OOD action 和 residual 的监督来源 |
| 18 | 区分 forward dynamics、inverse model、world model 与 policy；解释长 rollout 误差和 MPC 的反馈 |
| 19 | 独立提交数据流图、一个可重复实验和 10 分钟英文 FP2/VLA 说明，并回答消融与失败追问 |

## 6. 渐进式代码项目

项目围绕同一条主线扩展，而不是每课启动一个大型仓库：

```text
Lab 01: conditional action distribution
Lab 02: episode dataset → observation history/action chunks
Lab 03: sequence/multimodal conditioning and masks
Lab 04: generative action heads
Lab 05: mapping these concepts to openpi
Lab 06: asynchronous execution and force feedback
Lab 07: interactive data / return / offline support
Lab 08: learned dynamics + planning
```

详细实现规格见 [实验说明](labs/README.md)。[概率 BC 实验](labs/01_probabilistic_bc.py) 和 [episode 数据窗口实验](labs/02_episode_windows.py) 已可运行。Lab 02 的 frame/action conversion 与其余实验按课程顺序实现。

所有实验先记录 seed、数据生成方式、split 和指标。先使小 batch/小环境工作，再扩展容量。toy result 只支持对应机制的结论，不声称复现论文的机器人成功率。

本机已检查：Python 3.13、NumPy、PyTorch 可用，CUDA 可用。大型模型的可运行性仍需确认显存、依赖、checkpoint 和具体配置。每个实验都有不依赖完整 foundation checkpoint 的基础验收；全模型 inference/fine-tuning 作为资源允许时的扩展。

## 7. 本地论文的使用方式

| 知识线 | 核心案例 | 按需选读 |
|---|---|---|
| 表示与模态利用 | Making Sense of Vision and Touch、FACTR | ManipForce、Multi-Modal Policy Consensus、Learning When to See and When to Feel |
| 生成与高频反应 | 已学 RDP、ImplicitRDP | PhaForce |
| 数据与控制接口 | Force Policy、AVP、FP2 | ForceMimic、Adaptive Compliance Policy、EquiContact |
| 力觉 VLA 与迁移 | FP2；CRAFT/FAVLA 选一 | CompliantVLA-adaptor、FD-VLA、ForceVLA2、AT-VLA、VLF-DiT |
| 交互数据与纠错 | CR-DAgger | 对照原始 DAgger |

所有本地文件对应关系见 [论文索引](PAPER_INDEX.md)。匿名/审稿论文用于讨论当前稿件，不视为已正式接收，也不据此推断 release 或公开代码状态。

## 8. 需要补充的主干资料

以下材料已核对原始论文或官方项目入口；目前列为外部阅读，尚未下载到本地论文库。

| 资料 | 用途 | 阅读深度 |
|---|---|---|
| [Probabilistic Machine Learning](https://probml.github.io/pml-book/) | conditional probability、MLE、优化、latent-variable models | 查相应主题，不要求通读 |
| [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) | IL、RL、offline RL、model-based learning | 只跟第 16–18 课对应讲次 |
| [ACT](https://tonyzhaozh.github.io/aloha/) | CVAE、chunking、temporal ensemble | 第 8 课指定章节 |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | continuous action generation 与 receding-horizon deployment | 第 8 课主干精读 |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | 路径、velocity field、ODE | 第 9 课数学部分 |
| [Open X](https://robotics-transformer-x.github.io/) / [Octo](https://octo-models.github.io/) / [OpenVLA](https://openvla.github.io/) | 数据接口、generalist policy、VLM→action | 第 10 课比较阅读 |
| [π0](https://arxiv.org/abs/2410.24164) / [FAST](https://arxiv.org/abs/2501.09747) / [π0.5](https://arxiv.org/abs/2504.16054) | VLA 主干 | 第 11–12 课指定章节 |
| [DAgger](https://proceedings.mlr.press/v15/ross11a.html) | on-policy state distribution 与 expert relabeling | 第 17 课方法部分 |
| [openpi](https://github.com/Physical-Intelligence/openpi) | 公式到真实数据/代码接口 | 第 4 周起固定版本追踪 |

## 9. 面试与进度记录

使用 [学习记录](LEARNING_PROGRESS.md) 保存每课产出与未解决问题。主讲以中文解释并保留英文术语；每周英文口述围绕三个层级：

1. 90 秒：定义概念与核心问题。
2. 5 分钟：解释输入、输出、loss、inference 和 controller。
3. 追问：给出证据、消融、失败边界和自己的贡献。

最终 FP2 capstone 包括：完整框图、2/50/1000 Hz 时序、compressor 训练、twist–wrench/frame/mask、五组消融和两个失败案例。自己的实习贡献以实际参与记录为依据，避免把团队整体结果描述为个人独立实现。

继续：[第 5 课：训练、数据与观测](05_training_data_and_observation.md)。第 4 课的概率推导和 Lab 01 作为前置内容保留。
