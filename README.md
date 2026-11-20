# Robot Learning / VLA 论文与课程

本目录由 Git 管理论文库与 Robot Learning 学习项目。学习入口是 [robot_learning/README](robot_learning/README.md)。

**当前学习入口：[VLA循序学习路线](robot_learning/planning/vla_learning_path.md)。** 第5课之后的第6–15课拆为30个小单元；从 [第6课6A：像素到token](robot_learning/notes/06_multimodal_policy_representation.md) 开始，按直觉、手算、tensor、代码和口述逐步推进。对应 [面试练习册](robot_learning/notes/vla_interview_workbook.md)，每课练解释、推导和诊断。

文档使用相对链接，避免依赖某一台机器的绝对路径。代码实验默认使用 [Colab notebooks](robot_learning/labs/notebooks/README.md)，不要求本地安装或运行训练环境。

深读材料：[Pi 系列独立专题](robot_learning/notes/pi_series/README.md)。从 π0 / π0.5 原文讲到公式、openpi 数据流、部署诊断和面试项目，使用 P1–P6 标识。求职目标和课程对应见 [VLA / Robot Learning 岗位能力地图](robot_learning/resources/vla_job_competency_map.md)。

[ML 面试补强 M1–M10](robot_learning/notes/ml_interview/README.md) 与 [40道题卡](robot_learning/notes/ml_interview/interview_drill.md) 保留为按需参考。既有实验见 [11份Colab notebook](robot_learning/labs/notebooks/README.md)；已阅读题型与本人实验/独立解释分别记录。

近期面试补强：[Reinforcement Learning 专题 R1–R11](robot_learning/notes/Reinforcement%20Learning/README.md)，与 ML 并列。从 MDP、Bellman、TD 走到 PPO、SAC、Offline RL 和机器人诊断，配 33 道题卡与专题内的自包含 Colab notebook。首次从 [R1：reward 与 return](robot_learning/notes/Reinforcement%20Learning/01_mdp_and_return.md) 开始。

项目学习入口：[你的DM Walker PPO](robot_learning/notes/Reinforcement%20Learning/07d_dm_walker_project.md)，专讲24维observation、6维motor control与实际训练数据流；配套 [RL02](robot_learning/notes/Reinforcement%20Learning/notebooks/RL02_walker_ppo_tensor_trace.ipynb) 只做合成tensor/gradient机制实验。

同步补基础可读 [Robotics Fundamentals B0–B6](robot_learning/notes/Robotics%20Fundamentals/README.md)。从关节、任务与动作空间开始，依次复习坐标系、FK、IK、Jacobian、动力学和控制，以两连杆和单摆为例。

[Motion Planning 与 Trajectory Generation MP1–MP8](robot_learning/notes/Motion%20Planning/README.md) 单独复习规划与运控，从 A*、RRT 和碰撞检查推进到限幅轨迹、时间参数化、轨迹优化和在线换轨。

2026-11-09 FP2更新：[源码架构与训练精读](robot_learning/notes/03d_fp2_source_architecture_and_training.md) 已接入课程，按peg/RLT1024配置追踪token→FiLM/GRU/gate→三头、冻结与标签，并区分论文系统、源码变体和Colab toy。

```text
RobotLearning/
├── paper/                   # 论文原文与新增 PDF
├── robot_learning/
│   ├── README.md            # 学习入口与课序
│   ├── notes/               # 课程讲义、专题笔记、面试题
│   ├── labs/                # 实验代码与说明
│   │   ├── notebooks/       # 独立上传 Colab 的 .ipynb
│   │   └── results/         # 小型实验结果与报告
│   ├── planning/            # 课程计划与学习进度
│   └── resources/           # 论文索引、主题路线、外部资源
├── .gitignore
└── .gitattributes
```

- [课程目录](robot_learning/README.md)
- [当前学习路线：30个小单元](robot_learning/planning/vla_learning_path.md)
- [VLA面试练习册](robot_learning/notes/vla_interview_workbook.md)
- [第6课：VLA多模态表示](robot_learning/notes/06_multimodal_policy_representation.md)
- [ML 面试补强课程](robot_learning/notes/ml_interview/README.md)
- [Colab Notebook 运行入口](robot_learning/labs/notebooks/README.md)
- [Pi 系列专题：π0 → FAST → π0.5 → 工程与面试](robot_learning/notes/pi_series/README.md)
- [VLA / Robot Learning 岗位能力与学习证据](robot_learning/resources/vla_job_competency_map.md)
- [论文文件目录](paper/README.md)
- [第 4 课：概率视角的 BC](robot_learning/notes/04_probabilistic_bc_and_training.md)
- [第 5 课：训练、数据与观测](robot_learning/notes/05_training_data_and_observation.md)
- [后续课程计划](robot_learning/planning/course_plan.md)
- [论文索引](robot_learning/resources/paper_index.md)

## Git 使用

在本目录执行：

```bash
git status
git diff
git add robot_learning
git commit -m "Study: record lesson notes and experiments"
```

新论文放入 `paper/`，使用 `git add paper/<论文文件名.pdf>` 加入版本管理，并更新论文索引。论文、讲义、代码及小型结果文件会记录在 Git 中；`tmp/`、Obsidian 本地状态、Python 缓存、虚拟环境和大型训练 checkpoint 由 `.gitignore` 排除。

学习资料有变动时，同时检查课程目录、实验命令与相对链接，并在提交中写清本次学习或实验主题。
