# 课程笔记

本目录存放课程讲义、相关专题和面试题。学习计划在 [planning/](../planning/course_plan.md)，代码实验在同级 [labs/](../labs/README.md)，论文与外部材料在 [resources/](../resources/paper_index.md)。

**当前按 [30个小单元的循序路线](../planning/vla_learning_path.md) 学习，从第6课6A开始。** 第6–9课已展开为直觉、tensor、公式、手算答案与追问；10–15课的A/B/C任务见路线。每课对应一张 [VLA面试场景卡](vla_interview_workbook.md)，训练解释、推导和失败诊断。

| 课 | 内容 | 配套 |
|---|---|---|
| 1 | [机器人策略与 VLA](01_vla_policy_foundations.md) | BC、distribution shift、action interface |
| 2 | [Action chunk 与生成式策略](02_action_chunk_and_generative_policy.md) | ACT、DP、flow、FAST |
| 3 | [Slow-fast 接触策略](03_slow_fast_contact_policy.md) | [面试题卡](03_interview_drill.md) |
| 3b | [FP2 adaptor 与 twist–wrench](03b_fp2_adaptor_twist_wrench.md) | frame、mask、hybrid force-position control |
| 3c | [FP2 tensor 与数据流](03c_fp2_tensor_walkthrough.md) | compressor、三个控制参数、controller tensor 示例 |
| 4 | [概率视角的 BC](04_probabilistic_bc_and_training.md) | [Colab Lab01](../labs/notebooks/Lab01_probabilistic_bc.ipynb) |
| 5 | [训练、数据与观测](05_training_data_and_observation.md) | [Colab Lab02](../labs/notebooks/Lab02_episode_windows.ipynb) |
| 6 | [VLA 多模态表示](06_multimodal_policy_representation.md) | RGB/text/state→tokens；[VLA01](../labs/notebooks/VLA01_token_action_trace.ipynb) |
| 7 | [策略 Transformer 与因果性](07_policy_transformer_and_causality.md) | P/S/A mask、padding、cache 与泄漏 |
| 8 | [CVAE 与 Diffusion Policy](08_cvae_and_diffusion_policy.md) | chunk 生成、训练/推理与执行窗口 |
| 9 | [Flow Matching 动作 head](09_flow_matching_for_actions.md) | velocity tensor、采样、反归一化 |
| 10 | [从 VLM 到 VLA](10_from_vlm_to_vla.md) | OpenVLA/Octo/π0 原始架构对照 |
| 11 | [π0 策略系统](11_pi0_policy_system.md) | 主线任务 + P1/P2 深读 |
| 12 | [FAST 与 π0.5](12_fast_and_pi05.md) | tokenizer、AR、co-training、hierarchy |
| 13 | [动作与控制接口](13_action_and_control_interfaces.md) | units/frame/time；[VLA02](../labs/notebooks/VLA02_multirate_force_execution.ipynb) |
| 14 | [力觉与 foundation adaptation](14_force_aware_adaptation.md) | force fusion、冻结、FP2 sidecar |
| 15 | [VLA 泛化与评估](15_vla_evaluation_and_generalization.md) | rollout/force/latency、消融与失败诊断 |

2026-10-21：当前学习入口从第5课接 [第6课6A](06_multimodal_policy_representation.md)。第6–15课讲义已准备；先逐单元理解再做阶段复盘。ML专题按需查缺，第11/12课复用Pi深读材料，学习者完成情况见 [进度](../planning/progress.md)。

## ML 面试补强专题（2026-10-19）

[M1–M10 课程与面试优先顺序](ml_interview/README.md)，包括概率/统计、线性与经典模型、tensor/backprop、优化、数据评价、PCA/聚类、attention、生成模型和 LoRA。配套 [40 题](ml_interview/interview_drill.md) 与 [Colab notebooks](../labs/notebooks/README.md)。使用独立 M 编号，主线 6–19 的安排保留。

## Reinforcement Learning 面试复习专题（2026-10-25）

[R1–R11 课程与复习顺序](Reinforcement%20Learning/README.md) 与 ML 专题并列，依次复习 MDP/return、Bellman、MC/TD、DQN、Policy Gradient、Actor-Critic/GAE、PPO、SAC、探索/模型、Offline RL 和机器人系统。配 [33 道面试题](Reinforcement%20Learning/interview_drill.md)、[公式表](Reinforcement%20Learning/cheatsheet.md) 和专题内 [RL01 Colab notebook](Reinforcement%20Learning/notebooks/RL01_tabular_foundations.ipynb)。先完成 [R1A/R1B](Reinforcement%20Learning/01_mdp_and_return.md)，再按单元推进；采用独立 R 编号，承接主线第 16/17 课相关内容。

## Darwin Robotics 二面专项

[Darwin 二面：ML、Coding 与 Systems](interview/darwin_round2_ml_coding_systems.md) 面向 90 分钟 Machine Learning Challenge + Systems Questions，包含 P0/P1/P2 优先级、数学与训练调试、12 道 coding 参考题、软件工程与机器人系统、英文回答和两套限时模拟。题型是备考设计，不是公司题库。

## Pi 系列独立专题

[专题入口与顺序](pi_series/README.md)，编号 P1–P6，不占用主线第 6–19 课：

| 专题 | 讲义 |
|---|---|
| P1 | [π0 完整架构](pi_series/01_pi0_architecture.md) |
| P2 | [Flow Matching 推导与时间约定](pi_series/02_flow_matching.md) |
| P3 | [FAST 与动作表示](pi_series/03_fast_action_representation.md) |
| P4 | [π0.5 配方与泛化证据](pi_series/04_pi05_generalization.md) |
| P5 | [openpi 工程数据流](pi_series/05_openpi_engineering.md) |
| P6 | [VLA 基础补齐与求职项目](pi_series/06_vla_foundations_and_capstone.md) |
| 题卡 | [30 道面试练习](pi_series/interview_drill.md) |

第6–15课已接入主线；第16–19课（RL、offline RL/DAgger、dynamics/MPC、综合项目）继续按 [课序](../planning/course_plan.md) 准备。论文阅读记录使用 [模板](reading_template.md)。

主线文件沿用 `课号_主题.md`；短补充使用 `03b_...` 一类后缀，独立多课专题放入如 `pi_series/` 的目录。课后把推导、实验解释和英文口述情况记到 [学习进度](../planning/progress.md)。
