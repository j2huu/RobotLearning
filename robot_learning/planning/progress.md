# Robot Learning 学习记录

当前细化版本：2026-10-21。日常按 [循序学习路线](vla_learning_path.md) 的6A–15C推进；[总体计划](course_plan.md) 保留第4–19课的覆盖范围。每周8–10小时是建议预算，掌握检查优先于赶课表。

“已讲解”表示已有讨论/讲义；“通过”需要独立推导、代码解释和口述。当前不根据阅读数量推断掌握程度。

## 2026-11-09 FP2材料同步

[3d源码精读](../notes/03d_fp2_source_architecture_and_training.md) 已依据Ai2 09笔记与当前工作树定向核对，20个记录文件摘要匹配；补齐peg/RLT1024的context、梯度、FiLM/GRU/gate、三头与单步标签。旧论文笔记和第14课保留原推导，新增配置限定与当前时钟差异。

材料状态：已更新。本人验收仍待记录：画两条context路径、解释alpha/冻结、手算35/40与36/40、区分任务微调和FP2增量。本次未加载权重、运行模型、训练或连接机器人，未改写Colab运行状态。

## 当前学习位置（2026-10-21）

2026-11-09 RL项目入口已补充：[R7D DM Walker](../notes/Reinforcement%20Learning/07d_dm_walker_project.md) 先学24维observation与6维motor，再接actor/critic、GAE和PPO更新；RL各课添加项目对照。RL02仅做合成机制实验，实际Colab运行和W1–W4本人验收见 [RL进度](../notes/Reinforcement%20Learning/progress.md)，尚未代填完成。

用户反馈：ML题型已阅读完，希望VLA正课更细致、循序渐进并能服务面试。当前从 [第6课6A](../notes/06_multimodal_policy_representation.md) 开始，尚未代填本人掌握情况。第6–9课已细化讲义，第6–15课每课拆A/B/C三个单元；Pi材料在第11/12课深入，第16–19课仍按原计划准备。

新增 [VLA01](../labs/notebooks/VLA01_token_action_trace.ipynb) / [VLA02](../labs/notebooks/VLA02_multirate_force_execution.ipynb) 仅做静态检查，实际云端运行待完成。前者检查小型随机模型的数据流，后者为规则式spring接触实验，均非真实VLA/FP2复现。

## 当前小单元记录

在[路线](vla_learning_path.md)完成当前单元的手算和解释后再填“完成”。阶段口述使用 [面试练习册](../notes/vla_interview_workbook.md) 的四维量表。下表留空不代表未阅读，只表示尚未收到独立产出或自评记录。

| 课 | A单元 | B单元 | C单元 | 本人产出 / 薄弱点 |
|---|---|---|---|---|
| 6 | 像素到token：待记录 | 文字与状态：待记录 | 融合与意义：待记录 | 当前从6A开始 |
| 7 | Q/K/V：待记录 | mask：待记录 | cache与泄漏：待记录 | |
| 8 | chunk与CVAE：待记录 | diffusion训练：待记录 | 采样与执行：待记录 | |
| 9 | path/target：待记录 | loss/gradient：待记录 | sampler/time：待记录 | |
| 10 | VLM→VLA：待记录 | 三模型对照：待记录 | 方案选择：待记录 | |
| 11 | π0架构：待记录 | training step：待记录 | inference/源码：待记录 | |
| 12 | tokenizer：待记录 | AR policy：待记录 | π0.5配方：待记录 | |
| 13 | action contract：待记录 | frame：待记录 | 多速率：待记录 | |
| 14 | force入口：待记录 | compressor/训练：待记录 | 系统/边界：待记录 | |
| 15 | 问题/指标：待记录 | 对照/诊断：待记录 | 证据/口述：待记录 | |

单次记录：`单元 → 我的解释/手算 → 错误假设 → 修正后的反例 → 下次补哪里`。实验另填是否实际运行及结果依据；读完、推导通过、Colab运行、口述通过分别记录。

## ML 面试与 Colab（2026-10-19 新增）

入口：[M1–M10](../notes/ml_interview/README.md)、[40 题](../notes/ml_interview/interview_drill.md)、[9 份 notebook](../labs/notebooks/README.md)。默认在 Colab 运行，不要求本地环境。本轮仅完成文件准备和静态检查，没有执行本机训练或 Colab runtime。

| 课程 | 材料状态 | 本人验收 |
|---|---|---|
| M1/M2：概率、线性/逻辑回归 | 讲义 + ML01/BC notebook 已准备 | 待手推梯度、云端运行与解释 |
| M3：经典模型 | 讲义 + ML02 已准备 | 待 baseline 比较与模型取舍口述 |
| M4/M5：backprop 与优化 | 讲义 + ML03 已准备 | 待 CE gradient、train/eval、LoRA 梯度检查 |
| M6：评价与泄漏 | 讲义 + ML02/Lab02 已准备 | 待 group split、阈值选择、指标与区间解释 |
| M7：PCA 与表示 | 讲义 + ML04 已准备 | 待 SVD、低方差信号反例 |
| M8：attention/compressor | 讲义 + ML05/Pi 已准备 | 待 Q/K/V、mask 扰动、冻结图 |
| M9：生成模型 | 讲义 + ML06 已准备 | 待训练/采样信息差异、模式与耗时解释 |
| M10：迁移与系统题 | 讲义 + 题卡已准备 | 待 LoRA 参数量与完整方案 |

9 份 notebook 的 Colab 首次运行均待完成，具体检查边界见 [记录](../labs/notebooks/VALIDATION.md)。现有脚本的历史结果不代替 notebook 云端执行，也不计作本人验收。

## Pi 独立专题（2026-10-17 新增）

专题使用 P1–P6，不改变下面主线课号。入口：[课程顺序](../notes/pi_series/README.md)，[岗位能力](../resources/vla_job_competency_map.md)。以下仅记录材料状态，不代填学习者完成情况。

| 专题 | 材料状态 | 学习者验收证据 |
|---|---|---|
| P1：π0 架构 | 讲义已准备 | 待完成：计算图、shape、mask 与 cache 口述 |
| P2：Flow Matching | 讲义与机制脚本已准备 | 待完成：双时间约定推导、Euler 手算与反例解释 |
| P3：FAST | 讲义已准备 | 待完成：codec/预测/系统误差与 CE 解释 |
| P4：π0.5 | 全文讲义与证据卡已准备 | 待完成：六类数据、两阶段配方、消融和泛化边界 |
| P5：openpi 工程 | 固定 revision 阅读路径已准备 | 待完成：raw episode→command trace；全模型运行未完成 |
| P6：VLA 基础与 capstone | 讲义与项目规格已准备 | 待完成：独立项目、局限与英文说明 |
| 面试题卡 | 30 题与评分已准备 | 待完成：限时口述与错题复盘 |

助手的机制脚本验证见 [运行记录](../labs/results/pi_series_mechanics_run_notes.md)；不计作学习者验收，也不计作 π0/π0.5 模型复现。

## 之前的课程

| 课程 | 状态 | 下一次复盘重点 |
|---|---|---|
| 1：机器人策略与 VLA | 已讲解，待独立验收 | MSE 多峰平均、distribution shift、语义与物理泛化 |
| 2：chunk 与生成策略 | 已讲解，待独立验收 | episode/prediction/execution horizon；CVAE μ/σ；flow vs token |
| 3：slow-fast 接触策略 | 已讲解，待独立验收 | causal leakage、force correction vs regulation、缓存与调度 |
| 3b：FP2 adaptor 与 twist–wrench | 已讲解，待独立验收 | 双对偶变换、控制 mask、desired/measured wrench |

## 后续课程

| 周/课 | 状态 | 代码/推导证据 | 英文口述 | 未解决问题 |
|---|---|---|---|---|
| 1 / 4：概率 BC | 讲义与 Lab 01 已准备 | 待本人运行与解释 | 待完成 | |
| 1 / 5：训练、数据与观测 | 讲义与 Lab 02 数据窗口已准备 | 待本人运行与解释 | 待完成 | |
| 2 / 6：多模态表示 | 讲义/VLA01入口已准备；当前下一课 | 待tensor trace | 待完成 | |
| 2 / 7：Transformer 与因果性 | 讲义/机制实验入口已准备 | 待mask与扰动解释 | 待完成 | |
| 3 / 8：CVAE 与 DP | 讲义/已有生成实验入口已准备 | 待训练推理图 | 待完成 | |
| 3 / 9：Flow Matching | 主线讲义与P2深读已准备 | 待path/target/sampler trace | 待完成 | |
| 4 / 10：VLM → VLA | 讲义与原始模型对照已准备 | 待OpenVLA/Octo/π0接口表 | 待完成 | |
| 4 / 11：π0 | 主线任务/P1/P2/P5深读已准备 | 待完整源码数据流 | 待完成 | |
| 5 / 12：FAST 与 π0.5 | 主线任务/P3/P4深读已准备 | 待表示/训练/推理区分 | 待完成 | |
| 5 / 13：动作与控制接口 | 讲义/VLA02入口已准备 | 待frame/time与多速率解释 | 待完成 | |
| 6 / 14：力觉融合与 adaptation | 讲义/FP2阅读任务已准备 | 待freeze/force接口与消融 | 待完成 | |
| 6 / 15：泛化与评估 | 讲义与阶段交付物已准备 | 待独立实验报告 | 待完成 | |
| 7 / 16：RL 与 Actor-Critic | 已安排 | | | |
| 7 / 17：Offline RL / DAgger | 已安排 | | | |
| 8 / 18：Dynamics / MPC | 已安排 | | | |
| 8 / 19：代码与面试整合 | 已安排 | | | |

## 每次课后的记录模板

```text
日期 / 课号：
本次能独立解释的三个概念：
一条自己推导的公式（含假设）：
代码入口 / seed / 配置 / 指标：
实验支持什么、不能支持什么：
英文口述中的一个薄弱点：
尚未解决的问题：
下一次具体动作：
```

## 阶段验收

- [ ] 第 2 周：从 conditional BC loss 解释到 causal attention 和模态消融。
- [ ] 第 4 周：写出 flow loss，并解释 π0 的条件、动作表示与推理。
- [ ] 第 6 周：独立设计一种 force-aware VLA 与公平的消融方案。
- [ ] 第 8 周：完成数据流报告、可重复小实验和 FP2 英文面试。
