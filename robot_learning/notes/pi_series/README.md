# Pi 系列专题：从论文、公式到 VLA 工程与面试

建立日期：2026-10-17。目标岗位：VLA / Robot Learning Engineer，优先面向 manipulation policy 的训练与部署。

2026-10-19：机制实验优先使用 [Colab notebook](../../labs/notebooks/Pi_mechanisms.ipynb)，不需要本机环境；概率 BC 和数据窗口也有 [notebook 入口](../../labs/notebooks/README.md)。通用 ML 面试基础见 [M1–M10](../ml_interview/README.md)。

这是独立专题，使用 **P1–P6** 标识，不占用主线第 6–19 课。它把已有第 1–5 课的概念落实到 π0 / π0.5；之后回到主线时，可以用专题产出抵扣相应阅读与实验，不必重复学习。

本仓库的特色是 force-aware VLA 和 FP2。专题既补通用 VLA 工程主干，也保留“语义动作生成如何接上高频接触控制”这条研究线。学完论文摘要只能作为起点；通过需要能推导、读代码、定位失败并解释实验。

## 1. 先修与学习顺序

先修：[第 1 课](../01_vla_policy_foundations.md) 的策略接口、[第 2 课](../02_action_chunk_and_generative_policy.md) 的 chunk/生成模型、[第 4 课](../04_probabilistic_bc_and_training.md) 的条件分布、[第 5 课](../05_training_data_and_observation.md) 的数据 contract。不要求先学完第 6–19 课。

读到 attention、LoRA、Jacobian 或 RL 卡住时，先查 [P6 基础补齐](06_vla_foundations_and_capstone.md)，再返回当前课。

| 专题课 | 学什么 | 建议投入 | 独立产出 |
|---|---|---:|---|
| [P1：π0 的完整架构](01_pi0_architecture.md) | VLM、action expert、tokens、Q/K/V、block mask、cache | 3–4 h | 手画计算图、shape 表和 attention mask |
| [P2：Flow Matching 的推导与实现约定](02_flow_matching.md) | 概率路径、velocity target、ODE、时间方向、训练与采样 | 4–5 h | 推导、手算、运行机制实验 |
| [P3：FAST 与动作表示](03_fast_action_representation.md) | 连续/离散动作、压缩、CE、teacher forcing、误差来源 | 2–3 h | π0 / π0-FAST / π0.5 对照表 |
| [P4：π0.5 的完整训练与证据](04_pi05_generalization.md) | 六类数据、两阶段训练、hierarchy、消融与泛化边界 | 4–5 h | 数据配方图、论文证据卡和实验设计 |
| [P5：openpi 数据流与部署诊断](05_openpi_engineering.md) | 固定 revision、transforms、loss、sampling、执行与 profiling | 4–6 h | 从 raw episode 到 command 的 trace 报告 |
| [P6：基础补齐与求职项目](06_vla_foundations_and_capstone.md) | Transformer/优化、LoRA、几何、IL/RL、评估与作品证据 | 5–7 h | 一个可复查的小项目和 10 分钟口述 |
| [面试题卡](interview_drill.md) | 短答 → 推导/代码 → 系统追问 | 穿插 2 h | 录音、错题与评分 |

合计约 24–32 小时，按每周 8–10 小时安排约 3–4 周；这是完成基础产出的估计，完整模型训练和真机实验另计。每次按“读 45 分钟 → 推导/画图 → 练习 → 中文解释 → 英文复述”推进。

## 2. 论文与代码版本必须分开

| 材料 | 本专题使用方式 |
|---|---|
| [本地 π0，2410.24164v1](../../../paper/2410.24164v1.pdf) | 主读全文：正文、附录 B 架构、D 推理、E 评分；对应 2024-10-31 稿件 |
| [本地 π0.5，2504.16054v1](../../../paper/2504.16054v1.pdf) | 主读全文：正文、附录 B–D 评估、E 实现；对应 2025-04-22 稿件 |
| [FAST 原文](https://arxiv.org/abs/2501.09747) | 动作 tokenizer 的桥梁；当前没有本地 PDF |
| [openpi 固定 revision](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) | 代码阅读基准；核对日期 2026-10-17，不将 `main` 当成固定版本 |
| [Knowledge Insulation 官方说明](https://www.pi.website/research/knowledge_insulation) | 后续配方扩展；与本地 π0.5 v1 分开说明 |

**三处读论文时就应标记的差异：**

1. π0 v1 的线性路径、velocity 符号和积分方向存在不一致，P2 给出自洽推导；openpi 采用反向时间约定。
2. π0.5 v1 包括 FAST 预训练、联合目标和高层 subtask 推理；公开代码的 flow policy 接口不能直接当作全文训练与移动机器人系统的完整复现。
3. 论文 action 物理维度、代码 padding 维度和 action horizon 的下标约定不同。专题统一用 $N_a$ 表示 chunk 的**动作个数**，避免 $H$ 的歧义。

论文事实均注明章节/图表；代码行为注明 revision；教学例子与改进建议单独标明。PDF 页码按本地文件从第一页算起。

## 3. 贯穿六课的例子

任务：“把桌上的杯子放进水槽”。始终追问：

```text
任务指令 + 相机 + robot state
                ↓
       语义条件 / subtask
                ↓
      action distribution → chunk
                ↓
  反归一化 / action conversion / scheduler
                ↓
          low-level controller
                ↓
       执行与新观测 / failure log
```

杯子识别错、抓取坐标错、动作过时、接触力过大，是不同失败。改 VLM、action expert、数据或 controller 前，要先指出错误发生在哪条边上。

## 4. 立即能做的实验

当前默认方式是上传 [Pi_mechanisms.ipynb](../../labs/notebooks/Pi_mechanisms.ipynb) 到 Colab，内含全部机制代码和图表；下面本地命令仅保留为原脚本入口。

从仓库根目录运行，Python 3 标准库即可：

```bash
python3 robot_learning/labs/pi_series_mechanics.py --output robot_learning/labs/results/pi_series_mechanics.json
```

实验检查 flow 时间方向、attention 的间接泄漏与动作时延。它使用可计算的教学量，不训练 VLA、不下载 checkpoint、不报告机器人成功率。解读和扩展任务见 P2/P4/P5；[助手运行记录](../../labs/results/pi_series_mechanics_run_notes.md) 只证明脚本可用。

完整 openpi inference / fine-tuning 作为资源允许时的扩展。先完成不需要 GPU 的计算图、源码追踪与接口验收，不能把它们写成已完成模型复现。

## 5. 如何判断接近岗位要求

对照 [岗位能力与课程映射](../../resources/vla_job_competency_map.md)，按证据记录：

| 层次 | 通过的证据 |
|---|---|
| 讲得清 | 90 秒说明 π0；5 分钟比较 π0 / FAST / π0.5；追问时能说明假设 |
| 算得出 | 从路径推导 target；画 mask；算 latency 与 execution horizon |
| 读得懂 | 记录 revision、配置、函数、shape、units、mask 与 gradient path |
| 做得出 | 一个独立运行的实验，包含 baseline、受控变量、失败分析和局限 |
| 上得了机器人 | 在有硬件的条件下验证时钟、controller、执行日志和重复 rollout |

前四项可以先在本仓库完成，第五项需要真实硬件条件。不要用 toy 实验替代真机经验，也不以“读完多少篇”推断掌握程度。

当前状态：**讲义与机制实验已准备，学习者尚未验收**。记录入口：[学习进度](../../planning/progress.md)。
