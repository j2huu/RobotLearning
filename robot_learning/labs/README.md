# Robot Learning 代码实验

实验配合 [第 4–19 课计划](../planning/course_plan.md)。基础实验使用小数据、小模型与可控合成环境；与论文复现区分。每个实验记录配置、seed、数据 split 和评价指标，并写出结果的解释。

当前已经实现：Lab 01 和 Lab 02 的数据窗口阶段。Lab 02 的 action/frame conversion 在第 13 课扩展；其他条目是后续实现规格。

## 运行位置与结果

下面的命令都从论文工作区根目录 `Paper/` 执行。实验代码在 `robot_learning/labs/`，讲义在同级 `notes/`，运行结果写入本目录的 `results/`。小型 JSON 与实验说明由 Git 跟踪，大型 checkpoint 与运行缓存会被忽略。

| 当前代码 | 对应讲义 | 输出 |
|---|---|---|
| [01_probabilistic_bc.py](01_probabilistic_bc.py) | [第 4 课](../notes/04_probabilistic_bc_and_training.md) | `results/lab01_seed7.json` 与运行记录 |
| [02_episode_windows.py](02_episode_windows.py) | [第 5 课](../notes/05_training_data_and_observation.md) | `results/lab02_episode_windows.json` |

## 后续实验规格

| Lab                    | 课程       | 要实现的机制                                                                                            | 验收                                                                                    |
| ---------------------- | -------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 01：概率 BC               | 4        | deterministic MSE、Gaussian NLL、mixture NLL；episode split                                          | 推导 loss；解释 mean、sample、NLL 和 mode-compatibility 的区别                                   |
| 02：数据 pipeline         | 5/13     | timestamps、history/chunk、episode boundary、padding mask、train-only normalization、action round trip | window 不跨 episode；未来信号不泄漏；反归一化恢复 action 单位；frame 变换正确                                 |
| 03：因果与模态               | 6/7      | 小型 attention；future-force perturbation；force ablation/shuffle                                     | 改未来 force 不影响当前 action；证据按 contact/free-space 分段                                      |
| 04：生成式 action head     | 8/9      | 同一数据上的 diffusion/flow；time embedding、sampling loop                                                | 解释 target 和 timestep；报告 mode coverage 与采样耗时，控制数据和模型规模                                 |
| 05：openpi trace        | 10/11/19 | 固定代码 revision；observation/prompt/transforms/prefix/suffix/loss/sampling/denormalization           | 一份 shape+units+mask 报告；小型 mock tensor 检查；资源允许时单 batch/full inference                  |
| 06：slow-fast execution | 13/14/15 | cached context、力觉反馈、旧/新 chunk 对齐、接触 spring toy environment                                        | 用同一扰动比较无 feedback、低频 feedback、高频 correction、explicit regulation、stale/updated context |
| 07：交互与 RL              | 16/17    | tabular Bellman/TD；同小环境 BC/DAgger；offline support 反例                                              | 区分 expert action supervision 与 reward；展示 rollout-state shift 和 OOD action 的影响         |
| 08：dynamics/MPC        | 18/19    | fit forward dynamics；候选 action rollout；receding-horizon planning                                  | 比较 one-step 与 multi-step model error；用模型误差/扰动分析 MPC 的效果                               |

## Lab 01

依赖：Python、PyTorch。CPU 即可运行，没有额外数据或 checkpoint。

```bash
python robot_learning/labs/01_probabilistic_bc.py --steps 800 --seed 7 --output robot_learning/labs/results/lab01_seed7.json
```

实验输出包含训练配置、held-out 指标以及 $o=0$ 时的预测分布。没有绘图依赖。脚本的 synthetic episodes 不是物理仿真，mode-compatibility 不是机器人任务成功率。

## 每份实验的简短报告

```text
Question:
Data / assumptions:
Models / loss:
Controlled variables:
Seed / config:
Metrics:
Observation:
Explanation:
What this experiment does not establish:
Next change:
```

## Lab 02：数据窗口阶段

```bash
python robot_learning/labs/02_episode_windows.py --output robot_learning/labs/results/lab02_episode_windows.json
```

构造按 episode 划分的 synthetic dataset，进行因果 force 对齐、history/chunk padding、train-only normalization，并检查有效动作的 round trip。讲义见 [第 5 课](../notes/05_training_data_and_observation.md)。本阶段尚未实现图像 preprocessing 或真实 frame conversion。

## Capstone：FP2/VLA 接触控制设计

把各实验的结果映射回真实系统：

1. 画出 RFM、compressor、force policy、controller 的接口和频率。
2. 定义 observation/action/wrench 的单位、frame 与 timestamp。
3. 设计控制变量一致的 force/no-force、stale/update、correction/regulation 消融。
4. 区分 backbone inference、policy update、waypoint sampling、controller bandwidth 与实际 latency。
5. 报告两类失败：高层 motion 错误和局部 interaction 错误。
6. 将个人实习贡献、团队方法与 toy experiment 分别陈述。

真实 FP2 复现需要代码、数据、硬件与控制实现；本课程的 toy capstone 用于验证机制和实验设计。
