# 第 1 课：从机器人策略到 VLA

## 1. 先纠正一个容易误导的图

“Perception → Planning → Control”与“Vision/Language/State → VLA → Action”并不是严格对立。VLA 把部分感知、任务理解和动作生成合并到一个条件策略中，但真实系统仍然需要低层 servo、坐标变换、约束、安全检查和执行反馈。

更准确的系统图是：

```text
camera + language + robot state
              ↓
        learned policy / VLA
              ↓
 action target or action chunk
              ↓
 low-level position / velocity / impedance / force controller
              ↓
             robot
              ↑
       next observation (feedback)
```

因此，判断一个 VLA 是否“闭环”，不能只看它是否读取 state；还要看 observation 更新频率、每次预测多少步、实际执行多少步，以及 chunk 内能否被新反馈修正。

## 2. Policy 到底学什么

最简单的 behavior cloning 数据是：

$$
D=\{(o_t,\ell,a_t)\}_{t=1}^{N}.
$$

若假设动作服从单峰高斯且方差固定，最大似然训练等价于均方误差：

$$
\mathcal L_{\text{BC}}=\mathbb E_D\|\pi_\theta(o_t,\ell)-a_t\|_2^2.
$$

问题有两个：

1. **多峰性**：绕过障碍物可以向左也可以向右，均值动作可能正好撞上障碍物。
2. **分布偏移**：训练只见过专家到达的状态；模型一旦犯小错，就可能进入训练集没有覆盖的状态，误差继续累积。

- $o_t$ ​：机器人当前看到的图像、关节状态等 observation
- $l$：`"Pick up the red cup"` 这条语言指令
- $a_t$：专家在这个状态下执行的动作，比如末端执行器位移、关节速度、gripper command
生成式策略主要处理第一个问题。DAgger、人类干预、在线数据聚合主要处理第二个问题。本地的 [CR-DAgger](../../paper/NeurIPS-2025-compliant-residual-dagger-improving-real-world-contact-rich-manipulation-with-human-corrections-Paper-Conference.pdf) 正好用于观察这一区别。

## 3. 为什么预测 action chunk

把单步动作换成：

$$
a_{t:t+H-1}=(a_t,a_{t+1},\ldots,a_{t+H-1}).
$$

它带来三个好处：

- 一次预测能表达“靠近 → 接触 → 插入”这样的短时结构。
- 减少每个控制步都重新做昂贵大模型推理的需求。
- 训练标签包含局部时间相关性，动作通常更平滑。

也带来一个关键代价：如果一次预测 50 步并全部开环执行，模型在这 50 步内看不到新的遮挡、接触或卡滞。实际系统通常只执行前 $k\le H$ 步便重规划，或在 chunk 内叠加高频 residual/force controller。

这正是 [ImplicitRDP](../../paper/2512.10946v1.pdf)、[FAVLA](../../paper/2602.23648v1.pdf) 和 [PhaForce](../../paper/2603.08342v1.pdf) 反复解决的问题。它不是“是否使用 Transformer”的问题，而是感知、推理与控制的时间尺度问题。

## 4. 三种动作生成范式

### 4.1 连续回归

模型一次输出 $H\times d_a$ 个连续值。训练和推理简单、快，但普通 MSE 容易把多种合理动作平均掉。ACT 通过 latent variable 和 chunking 改善序列建模，但仍应追问其概率假设与推理方式。

### 4.2 离散 token 自回归

先把动作量化成 token：

$$
a_{t:t+H-1}\xrightarrow{\text{tokenizer}}z_{1:L},
$$

再训练：

$$
\mathcal L_{\text{CE}}=-\sum_{i=1}^{L}\log p_\theta(z_i\mid z_{<i},I,s,\ell).
$$

优点是直接复用 LLM 的 next-token machinery；代价是量化误差和长动作序列。FAST 的重点是利用动作在时间上的平滑结构进行频域表示和压缩，使连续轨迹变成更短的 token 序列。π0-FAST 是基于该 tokenizer 的自回归 VLA。

### 4.3 连续生成模型

Diffusion Policy 学习逐步去噪；π0 用 conditional flow matching 学一个随时间变化的 velocity field，把噪声连续运输成 action chunk。两者都可以表达多峰连续动作分布，但训练 target、采样动力学和推理步数不同。

不要把“diffusion”和“flow matching”只记成两个 loss 名称。每次比较都写出：随机路径是什么、网络预测什么、推理从哪里开始、要迭代几次、条件信息怎样进入网络。

## 5. VLA 比普通 visuomotor policy 多了什么

普通 visuomotor policy 也可以吃图像和 state，甚至也可以用 Transformer 或 diffusion。VLA 的关键增量是使用 language-conditioned、通常带有 VLM 预训练的统一策略，让视觉语义、语言指令和机器人动作发生迁移。

它希望解决：

- 同一观测下，根据语言执行不同任务。
- 利用网络图文预训练识别未在机器人数据中充分出现的对象与概念。
- 在多任务、多机器人数据上训练一个 generalist policy。

但 VLM “认识物体”不等于机器人“能稳定抓取它”。动作数据仍需要教会模型坐标、动力学、接触和时序。π0 的 action expert 解决连续高维动作生成；π0.5 的 heterogeneous co-training 进一步加强语义、子任务和开放环境迁移。

## 6. 四层时间尺度

分析任何 VLA 系统时，建议把频率拆成四层：

| 层 | 典型职责 | 需要的反馈 |
|---|---|---|
| 高层任务 | “先拿杯子，再放入水槽” | 语义进度，低频 |
| VLM/VLA 规划 | 根据图像和语言产生短时意图或 chunk | 图像、本体状态，中低频 |
| action expert / fast policy | 生成或修正连续动作 | state、force/tactile，中高频 |
| low-level controller | 跟踪位置/速度/力/阻抗目标 | encoder、F/T sensor，高频 |

“50 Hz action”也不自动意味着整个 VLM 在 50 Hz 完整运行。必须查看模型更新、chunk 发布、动作执行和传感器反馈各自的频率。

## 7. 用当前论文做一个统一比较

| 论文                           | 条件输入                                               | policy 输出                         | chunk 内反馈                |
| ---------------------------- | -------------------------------------------------- | --------------------------------- | ------------------------ |
| Admittance Visuomotor Policy | vision + proprioception + measured force $F$       | policy 预测 desired pose $p_d$、hand joints 与 desired contact force $F_d$ | admittance loop 根据 $F-F_d$ 生成修正后的 position command |
| Adaptive Compliance Policy   | vision/state 等示范条件                                 | motion + approximate compliance   | compliance 改变接触响应        |
| FAVLA                        | slow V/L + fast force/state                        | action chunk / fast correction    | 有，fast stream 利用高频力反馈    |
| ForceVLA2                    | V/L/state/force                                    | hybrid force-position control     | 有，action expert 内跨尺度融合   |
| CR-DAgger                    | base policy observation + human correction + force | residual action                   | 人在策略失败分布上提供纠正            |

这张表的重点不是记模型名，而是练习把论文宣传语还原成接口和闭环结构。

## 8. 本课自测

不看笔记回答：

1. 为什么 MSE behavior cloning 可能生成专家从未做过的“平均动作”？
2. action horizon $H$ 与每次实际执行步数 $k$ 分别控制什么？
3. 生成式动作模型主要解决多峰性，DAgger 主要解决什么？
4. π0 与 π0-FAST 在动作建模上最根本的区别是什么？
5. 一个模型读取 force，但每 1 秒才更新一次，为什么仍可能处理不好瞬时卡滞？
6. VLM 的语义泛化为何不能直接保证物理泛化？

通过标准：能用自己的机械臂例子回答每题，并画出从 sensor 到低层 controller 的完整闭环。

## 9. 下一课的准备

下一课进入 action chunk 与生成式动作建模。先精读 ACT 和 Diffusion Policy 的摘要、方法总图、训练目标与推理算法；阅读时统一记录 $o_t$、$a_t$、$H$、训练 target、采样步数和 replanning 频率。
