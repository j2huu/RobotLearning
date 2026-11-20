# Robot Learning 学习项目

这个目录用于系统学习 Vision-Language-Action（VLA）、Physical Intelligence 的 π 系列，以及接触丰富操作中的力觉与顺应控制。

论文 PDF 统一放在项目根目录的 [paper/](../paper/README.md)，本目录存放课程、实验、计划与资源索引。

**当前学习位置（2026-10-21）：ML题型已阅读，从第5课继续 [第6课6A：像素到token](notes/06_multimodal_policy_representation.md)。** 按 [循序学习路线](planning/vla_learning_path.md) 将第6–15课拆成30个小单元，每次完成一个机制、一个手算或反例、一次口述。第6–9课提供分步详解，后续课程按路线中的A/B/C任务学习。

[ML专题M1–M10](notes/ml_interview/README.md) 和 [40道题卡](notes/ml_interview/interview_drill.md) 用于补具体知识缺口；当前面试训练使用与正课对应的 [VLA练习册](notes/vla_interview_workbook.md)。代码使用 [11份自包含Colab notebooks](labs/notebooks/README.md)，不需要本地训练环境。

近期 RL 面试复习使用与 ML 并列的 [Reinforcement Learning 专题 R1–R11](notes/Reinforcement%20Learning/README.md)：从 [R1：MDP 与 return](notes/Reinforcement%20Learning/01_mdp_and_return.md) 开始，逐步进入 Bellman/TD、PPO、SAC 与 Offline RL。配 33 道题卡、公式表和专题内的 RL01 notebook，讲义准备与本人掌握/运行分别记录。

机器人基础同步使用 [Robotics Fundamentals B0–B6](notes/Robotics%20Fundamentals/README.md)，与 ML/RL 并列。先用两连杆理解关节、任务与动作空间，再学坐标系、FK/IK 和 Jacobian，接着用单摆认识动力学，最后补 PD、重力补偿与接触控制。每课有手算和自测，另有可选 notebook。

运控专题使用 [Motion Planning 与 Trajectory Generation MP1–MP8](notes/Motion%20Planning/README.md)，以两关节避障串起 A*、RRT、路径平滑、轨迹限幅和执行交接。

[Pi系列专题P1–P6](notes/pi_series/README.md) 在主线第11/12课作为深读材料，不需同时从头启动另一条课程。[第4–19课总体计划](planning/course_plan.md) 保留课程覆盖范围；当前每天学什么以循序路线为准。

学习目的进一步明确为 VLA / Robot Learning Engineer 的知识补齐、面试表达与项目能力，见 [岗位能力地图](resources/vla_job_competency_map.md)。配套 [实验](labs/README.md)、[学习记录](planning/progress.md) 和 [资源地图](resources/resource_map.md) 沿用原目录。

2026-11-09：[FP2源码精读](notes/03d_fp2_source_architecture_and_training.md) 已同步Ai2最新架构笔记，以peg/RLT1024配置讲清context读取、compressor冻结、FiLM/GRU/gating、三头与标签；配合 [论文tensor推导](notes/03c_fp2_tensor_walkthrough.md) 和第14课阅读。论文、源码配置与教学实验分别标注。

RL现结合个人项目学习：[DM Walker项目课](notes/Reinforcement%20Learning/07d_dm_walker_project.md) 从24维observation、6维motor到actor/critic、GAE、PPO更新，配 [RL02 Colab](notes/Reinforcement%20Learning/notebooks/RL02_walker_ppo_tensor_trace.ipynb)。原RL各课添加项目对照，SAC等作为后续比较，不写成个人已完成实现。

## 第5课之后：VLA 正课第6–15课

| 课 | 讲义 | 本课重点 / 实验 |
|---|---|---|
| 6 | [多模态表示](notes/06_multimodal_policy_representation.md) | RGB、language、state 的编码与融合；[VLA01](labs/notebooks/VLA01_token_action_trace.ipynb) |
| 7 | [策略 Transformer 与因果性](notes/07_policy_transformer_and_causality.md) | block masks、padding、cache、间接泄漏 |
| 8 | [CVAE 与 Diffusion Policy](notes/08_cvae_and_diffusion_policy.md) | condition→chunk，训练与采样；复用 ML06 |
| 9 | [Flow Matching 动作 head](notes/09_flow_matching_for_actions.md) | tensor、时间约定、ODE sampling、action postprocess |
| 10 | [从 VLM 到 VLA](notes/10_from_vlm_to_vla.md) | OpenVLA / Octo / π0，预训练与动作接口 |
| 11 | [π0 完整系统](notes/11_pi0_policy_system.md) | action expert、数据流、源码 trace；深读 P1/P2/P5 |
| 12 | [FAST 与 π0.5](notes/12_fast_and_pi05.md) | tokenizer、AR、异构训练与分层推理；深读 P3/P4 |
| 13 | [动作与控制接口](notes/13_action_and_control_interfaces.md) | normalization、frame、时序、controller；[VLA02](labs/notebooks/VLA02_multirate_force_execution.ipynb) |
| 14 | [力觉与适配](notes/14_force_aware_adaptation.md) | force-conditioned action / residual / FP2 regulation |
| 15 | [泛化与评估](notes/15_vla_evaluation_and_generalization.md) | 消融、失败归因、闭环与接触指标 |

学习顺序分五段：**6–7表示与信息流 → 8–9动作生成 → 10–12真实模型 → 13–14执行与力觉 → 15评估与证据**。每段完成一张图、一次推导/诊断和一次口述，具体检查答案见 [学习路线](planning/vla_learning_path.md)。第16–19课继续按原计划准备。Notebook的静态检查与本人Colab运行分开记录，目前尚无Colab运行证据。

## Pi 系列深读专题

| 专题 | 内容 |
|---|---|
| [P1：π0 架构](notes/pi_series/01_pi0_architecture.md) | VLM、action expert、tensor shapes、attention 与 cache |
| [P2：Flow Matching](notes/pi_series/02_flow_matching.md) | 路径推导、时间方向、Euler sampling、v1 公式核对 |
| [P3：FAST](notes/pi_series/03_fast_action_representation.md) | 动作表示、teacher forcing、CE 与推理成本 |
| [P4：π0.5](notes/pi_series/04_pi05_generalization.md) | 完整 v1 配方、六类数据、hierarchy、消融证据 |
| [P5：openpi 工程](notes/pi_series/05_openpi_engineering.md) | 固定 revision、transforms、数据流、延迟与故障定位 |
| [P6：VLA 基础与项目](notes/pi_series/06_vla_foundations_and_capstone.md) | 优化/LoRA、机器人几何、IL/RL、评估与 capstone |
| [30 道面试题](notes/pi_series/interview_drill.md) | 短答、推导、系统设计、英文口述与评分 |

建议投入约 24–32 小时。配套 [Colab 机制实验](labs/notebooks/Pi_mechanisms.ipynb) 不依赖 GPU 或 checkpoint，原 [脚本](labs/pi_series_mechanics.py) 保留；完整学习顺序、版本边界与验收见 [专题入口](notes/pi_series/README.md)。材料准备好不表示学习者已完成验收。

## 目录职责

| 目录 | 放什么 | 入口 |
|---|---|---|
| `notes/` | 连续编号的课程讲义、专题笔记与面试题 | [讲义目录](notes/README.md) |
| `labs/` | 可运行代码、实验规格与小型结果 | [实验目录](labs/README.md) |
| `planning/` | 课序、时间安排、验收与个人学习记录 | [课程计划](planning/course_plan.md)、[进度](planning/progress.md) |
| `resources/` | 本地论文索引、VLA 路线与外部资源映射 | [论文索引](resources/paper_index.md)、[资源地图](resources/resource_map.md) |

每课沿用“读讲义 → 推导/答题 → 做 Lab → 解释结果 → 记录进度”的顺序。课号与 Lab 编号独立；一个 Lab 可以跨多课扩展。

## 当前课程与实验对应

| 课 | 讲义 | 配套实验 | 当前材料 |
|---|---|---|---|
| 4 | [概率 BC](notes/04_probabilistic_bc_and_training.md) | [Colab Lab01](labs/notebooks/Lab01_probabilistic_bc.ipynb) | 旧脚本的 [运行记录](labs/results/lab01_run_notes.md)、[结果](labs/results/lab01_seed7.json) 不代表新 notebook 已运行 |
| 5 | [训练、数据与观测](notes/05_training_data_and_observation.md) | [Colab Lab02](labs/notebooks/Lab02_episode_windows.ipynb) | frame/action conversion 后续扩展；云端运行待记录 |
| M1–M10 | [ML 面试专题](notes/ml_interview/README.md) | [6 个新 ML notebooks](labs/notebooks/README.md) | 讲义/代码已准备，Colab 实际运行与本人验收待完成 |
| 6–15 | 本页主线讲义已准备 | VLA01/VLA02 + 已有生成/机制notebooks | 讲义、静态代码与本人云端运行分开记录 |
| 16–19 | 按 [课序](planning/course_plan.md) 继续准备 | RL/offline RL/dynamics/MPC 与综合项目 | 尚未完成的讲义/代码不生成占位文件 |

## 当前从这里开始

1. 打开 [学习路线](planning/vla_learning_path.md)，先看完整链和统一教学符号。
2. 完成 [第6课6A](notes/06_multimodal_policy_representation.md)，手算patch数量与projection shape；6B/6C再加入language、state和融合。
3. 按 [Colab分段阅读表](planning/vla_learning_path.md) 使用现有notebook，不要求第一次就理解全部cell。
4. 用 [VLA面试练习册](notes/vla_interview_workbook.md) 做对应场景卡，并将未通过的推导或解释记到 [进度](planning/progress.md)。

## 第1–5课与历史材料索引

1. 阅读 [学习路线](resources/vla_pi_roadmap.md)，先看整体依赖和阶段目标。
2. 阅读 [论文索引](resources/paper_index.md)，了解当前 24 篇本地论文分别解决什么问题。
3. 开始 [第 1 课：从机器人策略到 VLA](notes/01_vla_policy_foundations.md)。
4. 继续 [第 2 课：Action Chunk 与生成式动作策略](notes/02_action_chunk_and_generative_policy.md)。
5. 进入 [第 3 课：Slow-Fast 接触策略](notes/03_slow_fast_contact_policy.md)，对比 RDP、ImplicitRDP 与 Force Policy。
6. 深入阅读 [FP2 adaptor 与 twist–wrench 机制](notes/03b_fp2_adaptor_twist_wrench.md)。
7. 使用 [第 3 课面试题卡](notes/03_interview_drill.md) 做口述训练。
8. 阅读 [第 4 课：概率 BC](notes/04_probabilistic_bc_and_training.md)，完成 Lab 01。
9. 继续 [第 5 课：训练、数据与观测](notes/05_training_data_and_observation.md)，完成 Lab 02 的数据窗口阶段。
10. 继续 [第6课](notes/06_multimodal_policy_representation.md)，依次进入第6–15课；第11/12课衔接 [Pi深读专题](notes/pi_series/README.md)。
11. [ML专题](notes/ml_interview/README.md) 保留作推导与面试参考；代码继续在Colab运行，阅读完成不自动代替实验记录。
12. 每读一篇新论文，复制 [论文笔记模板](notes/reading_template.md)，并在论文索引中增加一行。

## 学习进度

| 课程 | 状态 | 核心验收 |
|---|---|---|
| 第 1 课：从机器人策略到 VLA | 已讲解，待独立验收 | 能解释 BC、多峰性、分布偏移与 VLA 接口 |
| 第 2 课：Action Chunk 与生成式动作策略 | 已讲解，待独立验收 | 能比较 ACT、Diffusion Policy、π0、π0-FAST |
| 第 3 课：Slow-Fast 接触策略 | 已讲解，待独立验收 | 能比较 RDP、ImplicitRDP、Force Policy 及其与 VLA 的关系 |
| 第 4 课：概率 BC | 讲义与代码已准备 | 推导 MSE/NLL，解释条件均值与多峰性 |
| 第 5 课：训练、数据与观测 | 讲义与数据窗口代码已准备 | episode split、因果对齐、normalization、mask 与训练诊断 |
| Pi 独立专题：P1–P6 | 讲义、题卡与机制实验已准备，待学习者验收 | 从论文、公式、数据流解释到执行；完成一个可复查项目 |
| 第6–15课 | 讲义、任务与配套notebook入口已准备 | 从多模态输入到action生成、controller和闭环评价 |
| 第16–19课 | 已安排，后续准备 | RL、offline RL、模型学习与综合项目 |
| ML 面试 M1–M10 | 用户反馈题型已阅读完（2026-10-21） | 实验运行与独立验收另行记录 |

## 两条学习线

```text
VLA 主干：BC → ACT / Diffusion Policy → RT / Open X → Octo / OpenVLA
                                            ↓
                                      π0 → FAST → π0.5 → openpi

研究专题：视觉-触觉表示 → 力觉模仿学习 → 顺应/混合力位控制
                                            ↓
                         force-aware VLA → fast/slow closed-loop VLA
```

主干回答“通用机器人策略怎样从多模态输入生成动作”；专题回答“遇到接触、遮挡、卡滞和高频扰动时，策略怎样稳定地感知并控制”。两条线在 π 系列的 action expert、action chunk 和闭环执行问题处汇合。
