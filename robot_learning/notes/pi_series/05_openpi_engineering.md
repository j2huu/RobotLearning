# P5：openpi 工程——从 raw episode 到机器人 command

前课：[P4](04_pi05_generalization.md)。本课以源码阅读、接口设计和故障定位为基础验收。完整 inference/fine-tuning 需要另行确认运行环境和数据。

核对日期：2026-10-17。固定代码 revision：[`215abfb217dbac7d5f1273282331b9b1866c0479`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)。这是一份可复查的阅读基准，不声称是以后始终不变的最新版本。

## 1. 先明确公开代码覆盖什么

固定版本 [README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md) 说明 π0.5 的公开训练与推理支持 flow head，并将发布模型关联到 KI 配方。不要把加载 `pi05` checkpoint 等同于复现本地论文的 FAST+文本+flow 全部 co-training，更不能等同于部署完整移动机器人高层系统。

本课追踪 **JAX `Pi0` 类中的 π0 / pi05 分支**。仓库也有 PyTorch 支持；两条路径的名字、dtype、训练配置和实现细节应各自验证，不能把两种框架的张量和 optimizer 对象混用。

## 2. 按数据流阅读，不按目录逐个打开

| 顺序 | 固定版本源码入口 | 记录什么 |
|---|---|---|
| 1 | [training/config.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py) | 模型、dataset、action horizon、变换和冻结配置 |
| 2 | [training/data_loader.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/data_loader.py) | `create_torch_dataset`、`transform_dataset` 如何得到 chunk 与 batch |
| 3 | [transforms.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py) | 字段重排、动作转换、Normalize/Unnormalize 的顺序 |
| 4 | [models/tokenizer.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py) | prompt、state 的序列化、长度和 masks |
| 5 | [models/pi0_config.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py) | `pi05` 分支、模型输入规格、LoRA freeze filter |
| 6 | [models/pi0.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py) | prefix/suffix、`compute_loss`、`sample_actions` |
| 7 | [policies/policy_config.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy_config.py) | checkpoint 和 normalization assets 怎样装配 |
| 8 | [policies/policy.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py) | `infer` 的输入变换、batch、采样、输出变换 |

阅读报告至少记录：revision、config 名、函数、输入/输出 shape、dtype、mask、units、时间语义、训练/部署差异。看完一个文件后若仍说不出它在整条链的位置，先回到上表。

## 3. 先写自己的 observation/action contract

**教学 schema，不是 openpi 原生配置文件**：

```text
episode_id / task_id / object_id / scene_id / robot_id
image.front / image.wrist + capture/arrival timestamp
state: joint angles [rad], gripper [约定单位]
prompt: overall task 或当前 subtask，注明来源
action: joint target [rad] + gripper，absolute，20 Hz
valid_step / terminal / intervention / success
optional wrench: [N,Nm], TCP frame, timestamp
```

从这个例子接入某个模型之前，回答：

- 相机名字和顺序是否与训练配置一致？缺失相机如何 mask？
- action 是 command、target 还是 executed state？示范时的 controller 是否一致？
- “增量动作”相对 chunk 开始的 $q_t$，还是相邻 waypoint？
- 数据采样间隔与机器人执行间隔是否一致？
- episode 结束处的窗口怎样处理？实际 loader 是否提供并使用有效位？

现有 [Colab Lab02](../../labs/notebooks/Lab02_episode_windows.ipynb) 已覆盖部分窗口、mask 和因果时间问题；它没有实现真实图像与机器人 action conversion。

## 4. Training pipeline 的关键顺序

固定版 `transform_dataset` 的主链：

```text
dataset row / chunk
  → repack transforms
  → dataset-specific data transforms
  → Normalize
  → model transforms
  → Observation + actions
```

LeRobot 路径按 dataset fps 请求 action offsets。张量窗口正确，也要确认日志因果顺序和 episode 边界，不能让数据格式替代物理验证。

教学反例：若动作是绝对关节目标，先转换为相对当前 state 的位移，再归一化，就需要**相同转换后的**训练统计。拿 absolute-action stats 去归一化 delta actions，数值有限也可能完全错。

训练统计应只来自允许的 train partition。部署把 normalization assets 与 checkpoint 配对，避免“加载模型 A，却用了数据集 B 的统计”。

## 5. State token 与 masks：四种东西分别检查

固定版 `Pi0Config` 默认 `action_dim=32`、`action_horizon=50`。`pi05=True` 的配置使用离散 state 输入和 adaRMS 条件化，默认 token budget 也不同。它们是代码默认值，不是物理维度或任意配置的保证。

`PaligemmaTokenizer` 的 pi05 路径将 state 离散后写进 prompt。state 变了，prefix 就变了；过长 prompt 截断还可能切掉关键 state/任务信息。

| mask | 决定什么 | 常见错误 |
|---|---|---|
| input validity | 相机/token 是否真的存在 | padding 被当成真实观测 |
| attention/block/AR | 哪个位置能读取哪个位置 | teacher-forced 答案泄漏 |
| token loss mask | 哪些文本/action tokens 计 CE | prompt 预测掩盖动作损失 |
| action time/dimension validity | 哪些动作元素有真实监督 | padding 或无效 DoF 主导 reduction |

并非每个模型入口都自动支持你希望的 action loss mask。要从 dataset 到 loss reduction 逐项核对。`compute_loss` 返回的维度，也是阅读证据的一部分。

## 6. 从 compute_loss 追到 sample_actions

按 [P2](02_flow_matching.md) 的反向时间约定解释源码：

1. 动作与 noise 形成 $x_s$；target 是 $\epsilon-A$。
2. prefix/suffix embeddings 经 block attention 交互。
3. 动作位置 hidden states 投影回 action dimension，计算向量场误差。
4. 推理先缓存 prefix，再用负 `dt` 运行 expert，最后返回 normalized chunk。

本课的验收是**自己为这些量写 shape 和依赖图**，不是复制源码。检查训练时用到了什么标签，部署时哪里生成随机数，何时重新编码观测。

尤其注意该实现只预填充图文 prefix cache；π0 的 state token 位于重新处理的 suffix。π0.5 则把离散 state 放在 prefix。不要因为论文提到 state 可缓存，就声称这两个实现走完全相同的缓存路径。

## 7. 输出变换不只是乘一个标准差

`create_trained_policy` 装配的输入路径包含 dataset transforms、Normalize、model transforms；输出反向经过 model output transforms、Unnormalize、dataset output transforms。

如果模型学的是 delta joint targets，输出还需恢复到机器人期望的 absolute command。若模型动作含 padding，要在对应输出转换中正确裁掉。最后得到 `[50,d]` 不意味着可以立即把所有数值作为关节力矩发送。

**教学 round trip**：

$$
a_{abs}\xrightarrow{-q_t}a_{delta}
\xrightarrow{normalize}\tilde a
\xrightarrow{unnormalize}\hat a_{delta}
\xrightarrow{+q_t}\hat a_{abs}.
$$

冻结同一 $q_t$，应恢复有效动作。若使用了 clipping、量化或旋转投影，需要分别报告其不可逆误差。

## 8. Chunk scheduling：算得出才谈实时性

令动作采样率为 $f_a$，每次执行 $k$ 个动作，下一次推理耗时为 $L$，其他开销为 $J$。执行窗口：

$$
T_{exec}=k/f_a.
$$

若在执行当前 chunk 时启动下一次推理，维持队列的必要预算是 $L+J\lesssim T_{exec}$，还要考虑 jitter 与剩余动作数量。若执行完再同步推理，一轮约为 $T_{exec}+L+J$，会出现等待或降低更新率。

教学例子：50 Hz、$k=10$，窗口 200 ms。新 chunk 来自 120 ms 前的图像，已有 $\lceil0.12\times50\rceil=6$ 个动作间隔过去。不能把第 0 个 waypoint 无条件当作当前目标。

丢弃过时前缀是一种时间对齐思路，也可能丢失关键接触动作；应结合状态偏差、轨迹一致性和重规划判断。DTW、平滑 transition 或 RTC 等机制也各有假设，不是通用补丁。本专题的 latency 实验只展示离散时钟算术，不模拟动力学。

## 9. Profiling 该记录哪些时间

下面是**工程测量建议**：

| 环节 | 单独记录 |
|---|---|
| 观测 | capture、arrival、policy start，得到 sensor age |
| preprocessing | 图像解码/resize、prompt、CPU→device |
| 模型 | prefix、每个 expert step、总 sampling |
| 输出 | device→CPU、反归一化、坐标转换、队列发布 |
| 执行 | command send、controller receive、实际跟踪 |

报告 warm-up 后的 p50/p95、deadline miss、最大队列 age，并区分冷启动/编译耗时。GPU/JAX 操作可能异步，要用同步点或设备计时，单纯包一层 Python wall-clock 可能低估真实耗时。

优化顺序取决于瓶颈：数据加载慢时减 expert steps 没用；prefix 占大头时单独优化小 action head 的收益有限。先测量，再决定 mixed precision、cache、batch、kernel 或模型缩减。

## 10. 失败定位练习

| 现象 | 第一轮排查 | 可区分原因的实验 |
|---|---|---|
| loss 很低，机械臂反向移动 | 单位、关节顺序、delta/absolute、frame | 固定 observation 做动作 round trip |
| 换一条指令仍抓同一物体 | prompt 处理、截断、语言 shortcut | 固定场景交换指令/目标，控制随机噪声 |
| 离线好，线上总慢半拍 | sensor age、sampling latency、队列 timestamp | 注入已知 delay，画 command 与观测时间轴 |
| 自由空间好，插接卡住 | 几何、contact 样本、力觉、controller | phase 分段与 force/no-force 对照 |
| train 好，未见家庭差 | split、场景多样性、数据配方 | 固定样本预算，按 scene/object 分组测试 |

## 11. 分层实践与交付

**基础，无 checkpoint**：完成一个源码 trace 报告，包含固定 revision、8 个入口、一个 batch shape 表、三种 masks、单位与时序；运行本专题机制实验并解释反例。

**有合适环境和 checkpoint**：取一个兼容 observation，完成 inference，记录配置和耗时，检查 normalized / physical outputs；再在小 batch 做 overfit 并检查实际可训练参数和梯度。

**有数据与机器人**：同一数据、相机、controller 下做 held-out rollout，记录多次试验和失败。接入新机器人和复现大规模预训练是不同范围的项目。

本次只准备了讲义、源码阅读基准与标准库实验，没有安装 openpi、下载权重、跑全模型或操作机器人。资源与系统修改不能当作默认前置条件。

英文口述：

> Trace one observation through preprocessing, conditioning, action sampling, denormalization, and execution. Then diagnose a policy with low offline loss but poor real-time performance.

下一课：[P6：基础补齐与项目](06_vla_foundations_and_capstone.md)。
