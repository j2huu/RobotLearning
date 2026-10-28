# 课程资源地图：Robot Learning 与 VLA

2026-10-21：当前按 [VLA循序路线](../planning/vla_learning_path.md) 从6A开始，主线6–15课分成30个小单元。本页用于查资料，日常学习按当前单元选读；OpenVLA/Octo/ACT/DP按原始版本作架构对照，Pi专题在第11/12课深入，ML按需查缺。Colab教学实验尚无云端运行证据。

[ML面试M1–M10](../notes/ml_interview/README.md) 包含CS229/PML、PyTorch与sklearn的定向参考；当前面试训练使用 [VLA场景练习](../notes/vla_interview_workbook.md)，已读ML题型无需重新通刷。

原资源查阅日期：2026-10-09。2026-10-17 补充：[Pi 独立专题](../notes/pi_series/README.md) 已提供 π0 / π0.5 全文讲义和固定版本代码阅读；[岗位能力地图](vla_job_competency_map.md) 将课程连接到工程求职。原连续课号保留，本页记录两个资源库如何进入主线课程。

## 1. 两个仓库各提供什么

| 仓库 | 查到的主要内容 | 课程中的用途 |
|---|---|---|
| [RayYoh/Awesome-Robot-Learning](https://github.com/RayYoh/Awesome-Robot-Learning) | survey、研究组、tactile 资源、simulation/benchmark、datasets | 帮助建立 manipulation 的全景，补数据、仿真、评估与 sim-to-real 的视角 |
| [DelinQu/awesome-vision-language-action-model](https://github.com/DelinQu/awesome-vision-language-action-model) | ACT、DP、RT-1/RT-2、PaLM-E、Octo、OpenVLA、HPT、RDT、VQ-BeT 等方法入口 | 帮助比较 policy representation、action generation、异构预训练和 VLM→VLA 的架构选择 |

两个列表作为资源导航。算法定义和具体结论回到作者论文或官方项目；不根据列表顺序安排知识依赖，也不把列表中的机构、版本和宣传性描述直接当作当前事实。

## 2. 补入后续课的内容

| 接入课程 | 资源/概念 | 要回答的问题 |
|---|---|---|
| 5：训练、数据与观测 | episode、dataset schema、benchmark protocol | demonstration 怎样成为监督样本？train/test 到底隔离了什么？ |
| 6–7：表示与 Transformer | 多模态、task/embodiment-specific inputs、shared trunk | image、state、force 如何成为 tokens？统一 token dimension 是否等于物理接口一致？ |
| 8–9：生成策略 | ACT、DP；VQ-BeT 作动作离散化补充 | 为什么同一个 BC 问题会用 reconstruction、denoising 或 discrete-token loss？ |
| 10：generalist policy | RT 系列、Octo、OpenVLA；HPT 作异构对照 | 是从 VLM 迁移语义，还是从多机器人数据学习共享 policy representation？ |
| 11–12：π 系列 | π0、FAST、π0.5；RDT 作 diffusion-foundation 对照 | backbone、action head、动作接口和预训练数据各贡献什么？ |
| 13：控制接口 | RDT unified action space、Force Policy、FP2 | 相同维度的向量是否具有相同单位、frame 和执行含义？ |
| 15：泛化与评估 | robosuite、SimplerEnv；domain randomization、sim-to-real | benchmark 是否覆盖论文声称的能力？模拟成功能支持什么实际结论？ |
| 16–18：交互学习 | offline datasets、simulation、dynamics、MPC | 数据来自固定示范还是当前 policy 的交互？模型误差如何影响规划？ |

## 3. 原始资源与定向阅读

| 资源 | 已核对的主题 | 阅读任务 |
|---|---|---|
| [Toward General-Purpose Robots via Foundation Models](https://arxiv.org/abs/2312.08782) | foundation model 在 robotics 中的使用方式、分类与泛化障碍 | 第 10 课前查 taxonomy，不以 survey 替代方法原文 |
| [HPT](https://arxiv.org/abs/2409.20537) | heterogeneous proprioception/vision 输入对齐到 tokens，学习 shared policy trunk | 第 10 课画 input-specific stem → shared trunk → output interface；比较 FP2 compressor |
| [RDT-1B](https://arxiv.org/abs/2410.07864) | bimanual diffusion Transformer、physically interpretable unified action space | 第 11/13 课比较 π0 的生成方法和 action contract |
| [VQ-BeT](https://arxiv.org/abs/2403.03181) | hierarchical vector quantization 将连续动作 token 化，Transformer 预测动作编码 | 第 8/12 课对照 CVAE、RDP latent 与 FAST tokenizer |
| [robosuite](https://robosuite.ai/) | MuJoCo-based robot-learning simulation framework 与 benchmark environments | 第 15 课定义 observation/action/controller/reward/success 的实验接口 |
| [SimplerEnv](https://github.com/simpler-env/SimplerEnv) | 对现实 manipulation policy 做 simulation evaluation | 第 15 课检查视觉匹配、物理变化、real/sim correlation 的评价边界 |

HPT 是 heterogeneous generalist policy 的案例；不能仅因使用 Transformer 就把它称为语言条件 VLA。RDT 与 RDP 是不同模型：前者是 bimanual diffusion foundation model，后者是本地已学的 reactive diffusion policy。

## 4. 与当前论文库的结合

每篇外部主干论文都与一个本地问题连接：

- ACT/DP/flow：为什么 RDP 和 ImplicitRDP 要改变 chunk 内的反馈路径？
- HPT/Octo/OpenVLA：不同 backbone 的 context 怎样交给 FP2？跨机器人数据为什么仍需 action/frame 对齐？
- RDT/FAST/π0：连续与离散动作表示怎样影响高频接触、采样延迟和执行？
- datasets/benchmark：FACTR、CRAFT、FAVLA 等声称“使用了 force”，需要什么反事实或消融证据？
- offline/interactive learning：CR-DAgger 的专家 correction 与 reward-driven RL 有什么区别？

每课维持一篇主读和少量对照章节。HPT、RDT、VQ-BeT 是定向对照，不新增三门独立课程。

## 5. 仿真与评估补充

第 15 课加入三层检查：

1. **接口一致**：sensor、单位、frame、action meaning、control frequency 与 latency。
2. **分布变化**：新背景/对象/任务/embodiment；摩擦、质量、stiffness、sensor noise 与时间延迟。
3. **结论范围**：simulation 成功、real-robot 成功和高频 force regulation 分别需要对应证据。

一个测试语义/视觉泛化的 benchmark 未必能检验瞬时接触调节。力控比较还需报告 wrench error/peaks、contact loss、controller rate 和实际延迟，并控制低层 controller 与数据量。

在 [第5课](../notes/05_training_data_and_observation.md) 之后先进入6A，Pi材料在第11/12课衔接。课程笔记放在 `robot_learning/notes/`，实验和结果放在同级的 `labs/`，本资源地图归入 `resources/`。
