# 第 10 课：从 VLM 到 VLA——语义怎样变成机器人动作

[上一课](09_flow_matching_for_actions.md) · [下一课：π0](11_pi0_policy_system.md) · [Colab VLA01](../labs/notebooks/VLA01_token_action_trace.ipynb)

目标：区分多模态理解、任务规划和低层 action prediction，并用输入、初始化、输出表示、loss、执行接口比较原始模型。

本课按 [循序路线](../planning/vla_learning_path.md) 分三次：**10A读§1–2画VLM→action接口；10B读§3–4填原始模型对照；10C读§5–7在具体任务约束下选方案**。前置是第8/9课的训练/采样图区分。完成后使用 [第10课面试场景卡](vla_interview_workbook.md) 检查能否解释不同loss不可直接比较、同shape不可直接执行。

## 1. VLM 的输出不是现成的控制量

VLM 通常学习图文理解/生成。它可以知道“红杯在左侧”，但机器人动作还需要坐标、单位、夹爪、时序与反馈。

把回答文字交给机器人解释执行是一种系统；直接条件化动作 token 或连续 action expert 是另一种系统。不要仅凭“用了语言模型”就把二者画成同一个计算图。

## 2. 主要结构选择

```text
RGB → vision encoder → projector ─┐
instruction → tokenizer/embed ────┼→ shared / interacting backbone
robot state → state adapter ─────┘                  ↓
                                        action representation / head
                                                    ↓
                                          native action → controller
```

关键设计问题：复用什么预训练知识？新机器人 state 怎样编码？action 是离散符号还是连续变量？训练中谁接收梯度？部署时执行多久才获得新观测？

## 3. 原始模型对照，不混入后续变体

| 对照对象 | 初始化与条件 | 动作路径 | 学习重点 |
|---|---|---|---|
| 原始 OpenVLA | pretrained Prismatic VLM；图像与任务语言 | LLM 预测 tokenized actions，再解码连续动作 | VLM→action tokens、量化/反归一化与任务适配 |
| 原始 Octo | robot-policy Transformer；语言/goal image、历史与灵活观测 | diffusion action decoding | 不同机器人/sensor/action 的模块化适配 |
| π0 v1 | pretrained VLM + action expert；图像/语言/state | conditional flow 生成 chunk | expert 交互、mask、cache 与连续动作 |

原始 OpenVLA 官方项目描述 fused SigLIP/DINOv2 vision encoders、projector 与 Llama 2 backbone。Octo 原项目将自己描述为 Transformer-based diffusion policy，强调灵活 task/observation/action。这些是原始版本对照，不代表所有后续 OpenVLA/Octo 变体接口不变。

不要由名称推断“只要语言条件就一定初始化自 VLM”，也不要由 diffusion head 推断模型没有语言条件。

## 4. Action token 与 language token

假设每个 scalar action 被量化成一个符号，dₐ=7、H=8，则最朴素序列有56个 action tokens；文本 prefix 长度与这56个动作符号是不同量。

AR 训练用 teacher forcing 与 CE；推理生成符号后还需 decode、反归一化、恢复 controller frame。量化误差与策略预测误差要分开。

连续 head 直接在 action tensor 上生成，不要求每个动作值对应词表 ID；它仍可以用 Transformer tokens 作为内部序列位置。

## 5. 预训练、目标任务适配与 embodiment

大数据预训练提供先验，目标任务 adaptation 改善特定 observation/action/task 分布。跨机器人数据混合仍需明确 action normalization、padding、mask、gripper 和机器人标识。

不同机器人都输出7维，不表示 joint、Cartesian、delta、absolute 可以无条件混合。Data adapter 不是一行 reshape。

“Zero-shot”也要问相对什么：未做目标任务 post-training、未见对象、未见场景、未见技能、未见 embodiment，是不同设定。

## 6. 语言有没有被使用

设计相同 scene/state、不同 instruction 的 counterfactual test：保持相机不变，要求选不同对象或不同顺序。检查训练数据是否让背景或任务 ID 足以预测目标，导致语言被忽略。

不能用训练文本 CE 或“会回答物体颜色”代替语言条件动作的成功证据。也不能将一次改指令的 out-of-distribution 失败直接认定为完全没用语言。

## 7. 读源码的顺序

1. Dataset schema 与 action definition。
2. Observation transforms、camera/text/state masks、normalization provenance。
3. Vision/text/state adapters 与 token order。
4. Training target、loss mask 与 gradient routing。
5. Sampler/AR decoder、postprocessing、chunk execution。
6. Controller 与 rollout metrics。

Notebook VLA01 只用小型随机网络追这条 tensor 路径；不能因它有 RGB/text/action 就称其具备 pretrained VLA 能力。

## 验收

五分钟对比 OpenVLA/Octo/π0 原始路径；解释 VLM knowledge 与 action skill；说明自己会怎样检查语言利用、数据异构与 controller contract。下一步进入第11课的 π0 逐层阅读。

参考：[OpenVLA 原始项目](https://openvla.github.io/)、[Octo 原始项目](https://octo-models.github.io/)、[π0/P1](pi_series/01_pi0_architecture.md)。
