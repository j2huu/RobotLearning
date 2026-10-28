# VLA / Robot Learning 工程岗位：能力、课程与证据

查阅日期：2026-10-17。目标是把求职准备从“读论文、背八股”推进到能解释和交付策略系统。课程入口：[Pi 专题](../notes/pi_series/README.md)。

本页区分**招聘页面明确列出的能力**与**据此整理的学习建议**。以下是少量美国机器人团队的岗位样本，不代表所有公司/地区的门槛，不是正在投递的职位清单；页面和岗位状态可能变化。

## 1. 官方岗位样本给出的信号

| 原始岗位说明 | 明确涉及的工作 | 本仓库可对应的准备 |
|---|---|---|
| [FieldAI：Robotics Autonomy Engineer, Manipulation](https://jobs.lever.co/field-ai/669f149b-a7a9-4f6e-8080-79b8b7903c83) | 示范学习、VLA 适配、真机部署、接触/力控、仿真评估；Python/C++、ML frameworks、运动学/动力学；ROS/ROS2 等为扩展项 | 既要会训练 policy，也要解释 action/controller、重复 rollout 和失败闭环 |
| [Figure：Helix AI Engineer, Data Infrastructure](https://job-boards.greenhouse.io/figureai/jobs/4345915006) | 机器人日志从采集、传输、存储到训练访问的基础设施；Python、Linux、可靠软件工程 | episode schema、数据质量、版本、采样吞吐与可复现配置 |
| [Figure：Helix AI Engineer, Training Performance](https://job-boards.greenhouse.io/figureai/jobs/4705296006) | 分布式训练、profiling、混合精度、checkpointing、数据管线与 GPU kernel 优化 | 先建立内存/耗时模型，进一步方向再深入分布式与 CUDA |

后两项是基础设施和性能专项岗，不能把它们的全部资深要求都压到初级 manipulation policy 工程岗位上。样本表明的是能力方向有分工；下列优先级是课程设计建议。

## 2. 能力矩阵与完成证据

| 优先级 | 能力 | 对应课程 | 面试能说什么 | 项目应留下什么 |
|---|---|---|---|---|
| 必修 | Python、数组/tensor、调试、Git/Linux | 现有 Lab 01/02；P2/P5 | shape/broadcast、dtype、seed、异常定位 | 独立命令、环境、可重复结果 |
| 必修 | 概率、优化、BC/生成模型 | 第 4/5 课；P2/P3/P6 | MSE/NLL、FM、CE、train/rollout gap | 推导与同条件实验 |
| 必修 | Transformer/VLM→VLA | P1/P4/P6 | Q/K/V、mask、cache、state 与 action expert | 计算图、tensor 表、扰动检查 |
| 必修 | 机器人数据工程 | 第 5 课、Lab 02、P5 | schema、因果时间、episode split、normalization | 数据 contract、泄漏检查、round trip |
| 必修 | fine-tuning 与诊断 | P5/P6；主线后续 | trainable parameters、LoRA、overfit、gradients | 配置、loss 曲线、held-out 结果 |
| 必修 | 部署与评估 | P5/P6 | latency budget、command rate、sensor age、OOD | p50/p95、rollout protocol、失败分型 |
| 必修 | 运动学与控制接口 | 第 3/3b 课、P6 | frame、Jacobian、PD/impedance/hybrid | 单位与 frame 说明、执行链 |
| 研究特色 | 力/触觉与 FP2 | 第 3/3b 课；P4/P6 | causal force、modality collapse、regulation | phase 指标、force 消融和边界 |
| 第二阶段 | RL / offline RL / dynamics / MPC | 主线第 16–18 课；P6 地图 | expert label 与 reward、Bellman、模型误差 | 小环境验证与清楚的结论范围 |
| 方向扩展 | 大规模数据/分布式/CUDA、ROS2/C++ | P5/P6 入门后定向深化 | 根据职位说明选择深度 | 真实 profiling 或系统集成证据 |

“必修”表示这条求职学习路线优先补齐，不代表每份招聘启事都逐项要求。主线第 6–19 课仍有未编写讲义/代码，不把课程计划当成完成证据。

## 3. 结合当前仓库的判断

仓库材料在接触控制与 force-aware VLA 上已经有明显积累；现有课程也重视概率、动作接口、因果 mask 和英文说明。尚需系统补齐的是：Pi 架构与配方的细读、公式到代码的数据流、训练/部署排错、独立项目与评价证据。

这只是对**仓库内容覆盖范围**的判断，不是对学习者当前能力的测评。已有讲义、助手跑过实验、学习者独立完成，是不同状态。

建议定位是：先建立能够适配和诊断 manipulation VLA 的通用能力，再用接触与力控作为深入方向。若只熟悉 force-aware 论文名，却不能解释一个训练 batch 或机器人 command，求职论述仍会断在工程环节。

## 4. 三档作品，分别适合什么条件

| 档位 | 前提 | 可以交付 | 简历表述边界 |
|---|---|---|---|
| A：机制与接口 | CPU、现有仓库 | 数据窗口、flow/attention/delay 检查、openpi trace | 可以说实现教学验证与源码审计；不能说训练了 π0.5 |
| B：模型适配 | 兼容 GPU、checkpoint、合法可用数据 | 小数据适配、LoRA/采样/数据处理的一项受控对照 | 报告实际任务、模型版本、训练预算和离线/仿真结果 |
| C：真实闭环 | 机器人、控制接口、评估条件 | repeated rollout、部署延迟、接触实验和恢复分析 | 真实成功率要附次数与条件，清楚说明个人贡献 |

先完成 A，再按资源做 B/C。没有硬件时可以把结论做到扎实，但无法凭文档补出真机部署经验。

## 5. 八股怎样变成面试能力

每个概念做四件事：

1. 30–90 秒定义，回答它解决什么问题。
2. 一条公式或一张接口图，说明输入、输出、监督和推理。
3. 一个失败反例，说明何时不够用。
4. 一个运行记录或实验设计，说明怎样验证。

使用 [30 道专题题卡](../notes/pi_series/interview_drill.md)；错题优先修正“概念混淆”和“接口说不完整”，其次再追求更快、更顺畅的英文表达。

## 6. 后续补课优先级

完成 Pi 专题后：

1. 补 Transformer 和 multimodal representation 的独立实现，对应主线 6/7。
2. 在同一小环境实现/比较 generative action heads，对应主线 8/9。
3. 做一项真实数据适配与评估，接上主线 10–15 和 Lab 05/06。
4. 再补 DAgger、RL、offline support、MPC，对应主线 16–18。
5. 按实际目标 JD 决定加深 C++/ROS2、数据系统或训练性能；不同时启动所有工具链。

真正投递前，用最新目标岗位说明重新做一次差距表；本页的日期和样本只作为当前课程设计依据。
