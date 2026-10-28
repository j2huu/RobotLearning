# 第 11 课：π0 的 action expert、训练与执行

[上一课](10_from_vlm_to_vla.md) · [下一课](12_fast_and_pi05.md) · 深读：[P1 完整架构](pi_series/01_pi0_architecture.md)、[P2 推导](pi_series/02_flow_matching.md)

本课把 Pi 专题接回连续主线。已读 P1/P2 时直接完成本课 trace 与验收，不重复通读。对象是本地 π0 v1 与专题固定版本代码，不把新版本接口自动回填到原论文。

按 [循序路线](../planning/vla_learning_path.md) 分三次：**11A读§1–2和P1画架构；11B读§3和P2追一个训练batch；11C读§4–6和P5追inference及源码**。每次只为一个问题查专题，记录实际版本、shape与变量来源；最后用 [第11课面试卡](vla_interview_workbook.md) 检查前向依赖与反向梯度能否分别解释。

## 1. 需要能画出的完整路径

```text
image/language → VLM prefix ───────────┐
state → state embedding ──────────────┼→ 与 action expert 逐层 attention
noisy action + flow time → action emb ┘             ↓
                                         predicted velocity field
                                                    ↓ 多次 ODE 更新
                                           normalized action chunk
                                                    ↓ 反归一化/映射
                                             native robot actions
```

Action expert 使用专门参数，与图文 tokens 交互；不是简单在最后一个语言 token 后接 MLP，也不是任意 learned router 动态选择的普通稀疏 MoE。

## 2. 三个维度不要混淆

```text
d_native：真实机器人 action contract 的维度
d_model_action：模型对齐/padding 后的动作宽度
d_hidden：expert 的 hidden width
```

输入 Xτ=[B,H,d_model_action] 经 projection 成 [B,H,d_hidden]；输出再投影回 [B,H,d_model_action]。模型 padding 维度不等于机器人 DoF，隐藏宽度更不是动作维度。

精确默认配置见 P1/P5 标注的版本。不要把本课教学 notebook 的 d=32、H=8 与真实 π0 超参数混写。

## 3. 一次 training step

1. Dataset 按 episode 取当前图文/state 与未来 clean chunk、valid masks。
2. 使用 train statistics 归一化有效动作并补齐模型维度。
3. 采 noise 与 flow time，构造 Xτ 和对应 velocity target。
4. Prefix/state/action tokens 按 block mask 交互。
5. 从 action positions 投影出 vector field，masked loss，更新指定参数。

Forward 输入 state 不意味着 force 已经存在；预训练不等于冻结；初始化来源和当前训练梯度是不同问题。

## 4. 一次 inference call

获取 c；编码可复用条件；从 noise 初始化 chunk；按当前代码约定迭代 velocity；反归一化并输出 native chunk。随后执行前 K 个动作，再重新获取 observation。

同一次采样的 context cache 不会带来新视觉反馈。Noisy suffix 每步变，而固定条件 prefix 是否可缓存取决于 mask 与实现。

## 5. 必做源码 trace

以 [P5 固定 revision](pi_series/05_openpi_engineering.md) 为阅读基准，写出 observation transforms→normalization→prefix/suffix→loss→sample_actions→output transforms 的表。每列至少包含 shape、单位、来源、mask、可用时间与训练/推理差异。

VLA01 复现的是这些通用接口检查，不复现 openpi 的多 expert 权重、预训练数据或缓存实现。不要以随机网络输出“看起来平滑”证明学会操作。

## 6. 论文证据的边界

论文架构、训练配方、post-training、任务评估是四块内容。对 base 与 post-trained policy 的结论要分别陈述，normalized task score 与整 episode success 也不能互换。

阅读报告时至少记录 task、robot、数据来源、是否 post-trained、trial 数、指标与失败类型；准确数字与论文章节回到 P1 的证据记录。

## 验收

不看笔记画完整 forward graph；写一次 training batch；解释 block mask 与缓存；说明 H、K、flow steps 和 action rate 的区别。用两句话说明为什么 π0 能生成动作仍不自动保证接触力正确，为第14课作准备。
