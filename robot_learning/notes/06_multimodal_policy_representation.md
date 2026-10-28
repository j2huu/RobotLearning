# 第 6 课：多模态表示——从观测到一串可被策略读取的 tokens

[上一课：数据与观测](05_training_data_and_observation.md) · [循序学习路线](../planning/vla_learning_path.md) · [下一课：Transformer 与因果性](07_policy_transformer_and_causality.md) · [Colab VLA01](../labs/notebooks/VLA01_token_action_trace.ipynb)

本课只解决一个核心问题：**相机、语言和机器人状态，怎样从不同类型的数据，变成策略网络能够共同处理的输入？** 先弄清这件事，再学第 7 课的 attention，最后到第 8/9 课生成动作。

贯穿案例是“把红杯放进盘子”。本课的尺寸与 notebook 一致：batch 2、相机 2、RGB 为 32×32、patch 为 8×8、文本占 6 个 slots、state 为 7 个合成坐标、hidden width 为 32。它们是教学配置；notebook 的随机图像、随机 token ID 和合成 state 没有真实任务语义。

## 怎么学这一课

| 小节 | 本次只解决什么 | 建议时间（估计） | 离开前留下什么 |
|---|---|---:|---|
| 6A | 先分清观测和tensor轴，再把RGB变成patch tokens | 60–75 分钟 | 输入/目标表与完整图像shape链 |
| 6B | 文字和state怎样编码，并与图像组成条件？ | 45–60 分钟 | 三条支路与prefix组合表 |
| 6C | tokens 在哪里融合，怎样判断模型真的用了某种输入？ | 45–60 分钟 | 一个反事实实验和 90 秒口述 |

前置自检：能否说明 batch、训练 target 和 train-only normalization？若不清楚，只回看 [第 5 课](05_training_data_and_observation.md) 对应段落。先手推，再在 Colab 对照；本课不要求本机运行。

## 6A：从观测和 tensor 轴，到像素变成 token

### 6A.1 把模型放回任务中

看到红杯和盘子的图像说明场景里“有什么、在哪里”；语言说明“这次要做什么”；本体状态说明“机器人当前在哪里、是什么姿态”。这些信息共同约束合理动作。

将当前可用条件写为：

$$
c_t=(I^1_{t_1},\ldots,I^{N_c}_{t_{N_c}},\ell_t,p_{t-T_p+1:t}),
\qquad A_t\sim\pi_\theta(A\mid c_t).
$$

这里 $t$ 是决策时刻，$t_i$ 是第 $i$ 路图像的采集时刻；到达时刻还可能更晚。$p$ 是可测 proprioception，不默认等于环境完整 state。历史长为 $T_p$，必须是当前已经到达、允许读取的历史；它与文字token数量不同。

| 量 | 当前策略能否得到？ | 在系统里的角色 |
|---|---|---|
| 已到达的 RGB、任务指令、本体状态 | 可以 | 条件输入 |
| 离线 demonstration 中后续 8 个动作 | 训练时可以作为标签 | 监督目标；生成模型还会用它构造训练样本 |
| 机器人执行后才发生的 force / RGB | 当前部署时不可以 | 之后的反馈或离线评价 |
| 模型预测的后续 8 个动作 | 推理结束后可以 | 待执行的预测结果 |

“数据集里有”与“部署这个时刻能用”是两个判断。先写出这张表，才能判断训练输入是否泄漏。

### 6A.2 `[B,N,d]` 的三个轴各做什么

把一个 token 想成“序列中的一个信息位置”，该位置用 $d$ 个连续特征表达。这里的 token 不一定是文本词表中的整数。

| 轴 | 本课含义 | 改变它意味着什么 |
|---|---|---|
| $B$ | 一批样本数 | 同时处理更多独立样本；标准 attention 不跨 batch 读取 |
| $N$ | 一个样本的 token 数 | 更多图像 patch、文字或动作位置 |
| $d$ | 每个 token 的特征宽度 | 每个位置由多少个 learned coordinates 表达 |

例如 `[2,32,32]` 中，第二个 32 是两台相机合计的 patch 数，第三个 32 是每个 patch 的表示宽度。二者数值恰好相同，含义完全不同。增加 hidden width 不会增加相机覆盖区域；增加 batch 也不会增加单个样本的历史长度。

### 6A.3 Shape、单位和坐标系要分开写

假设真实机器人 state 是 7 维关节位置，shape `[B,7]` 还不够，需要说明单位为 rad、关节次序以及是否 normalized。另一套 `[B,7]` 可能是 position 3 + quaternion 4，与前者没有相同物理含义。

经过 Linear 后得到 `[B,1,32]`，32 个分量通常不再分别表示某个关节或笛卡尔轴。它是 learned feature，但输入的 physical contract 仍必须正确。**feature width 统一提供计算兼容性；物理语义一致需要额外的数据约定。**

**停下来手答。** 将 state 从 degree 转成 rad，shape 是否变化？将 $B=2$ 改为 $B=4$，一个样本的 token 数是否翻倍？

**答案提示。** 两题都不变。前者改变数值及单位，后者改变并行样本数。若只用 shape 检查数据，这两类语义错误很容易漏掉。

### 6A.4 RGB → patches → image tokens

一张 32×32 RGB 图像，按不重叠的 8×8 patch 划分：每个方向有 4 块，共 $4\times4=16$ 块。每块有 $8\times8\times3=192$ 个像素通道值。

将第 $j$ 块按固定次序展平为 $x_j\in\mathbb R^{192}$，再用相同的 Linear 投影每一块：

$$
e_j=x_jW_{\mathrm{patch}}+b,
\qquad W_{\mathrm{patch}}\in\mathbb R^{192\times32},
\qquad e_j\in\mathbb R^{32}.
$$

上式把输入当作行向量，因此数学权重写为 `[192,32]`；PyTorch `Linear(192,32).weight` 实际存储为 `[32,192]`，内部使用其转置。比较公式和代码时先确认乘法约定。

共享投影的意思是每个 patch 使用相同参数；不同空间位置稍后由 position embedding 区分。两个相机也共享这套 patch projection，但 camera embedding 标明来源。

Notebook 用 `Conv2d(3,32,kernel_size=8,stride=8)` 实现这个操作：一个输出格点看一个 8×8 RGB patch，32 个卷积核产生 32 个特征。**在无重叠、无 padding、权重排列匹配时，它等价于对每个展平 patch 应用共享 Linear。** 并非任意卷积都等价于这里的不重叠 patchification。

```text
images                          [2,2,3,32,32]
合并 batch 与 camera 轴          [4,3,32,32]
Conv2d 的输出                    [4,32,4,4]
flatten(2)                      [4,32,16]
transpose(1,2)                  [4,16,32]
恢复 batch 与 camera 轴          [2,2,16,32]
加入 camera/patch positions      [2,2,16,32]
合并 camera 与 patch 轴          [2,32,32]
```

为什么要 `transpose`？卷积输出的 channel 轴是表示宽度，Transformer 输入约定最后一轴为表示宽度，因此要把 channel 移到末尾。只做 `reshape` 去凑出相同 shape，可能重新解释了错误的元素顺序。

这里把两路相机暂时并入 batch，只是为了让同一个图像算子处理 4 张图片；恢复 camera 轴后，它们仍属于各自的两个样本。并没有让样本 1 的相机与样本 2 的语言配对。

本例参数量也能手算：patch projection 有 $32\times3\times8\times8+32=6176$ 个参数，最后 32 来自 bias。相机数量改变不会改变这层共享卷积的参数量，但会改变相机 embedding 的数量和输入序列长度。

### 6A.5 Embedding 的广播为什么不改 shape

Notebook 中：

```text
patch features    [B,2,16,32]
camera embedding  [1,2, 1,32]   每台相机一个32维向量
patch position    [1,1,16,32]   每个patch位置一个32维向量
相加后             [B,2,16,32]
```

广播只是在 batch 或 patch 位置复用同一个向量。例如相机 0 的 camera embedding 加到相机 0 的全部 16 个 patches 上。这个操作标注“来自哪台相机”，并不自动计算相机外参、深度或三维对应关系。

真实系统还要说明 RGB 通道顺序、像素范围、resize/crop、相机时间戳。随机教学图像已经是 `[0,1]` float；真实预训练 encoder 需要其对应 preprocessing，不能直接套本例。

**手算练习。** 两相机图像由32×32改为64×64，patch仍为8×8、hidden width仍为32，image tokens是什么shape？patch projection参数量是否增加？

**答案提示。** 每相机8×8=64个patch，image tokens为`[2,128,32]`。共享projection仍为6176个参数，但patch position embedding长度要从16改成64；计算量和序列长度也会改变。

**6A 完成标准：** 能区分observation/label/prediction，独立从`[2,2,3,32,32]`推到`[2,32,32]`，解释flatten与transpose，并给出一个shape正确而物理含义错误的输入例子。

## 6B：语言和 state 编码，再组合完整条件

### 6B.1 Language：整数 ID → 连续向量

```text
任务文字           "把红杯放进盘子"
真实 tokenizer     得到若干词表 ID；本教学 notebook 用随机 ID
text IDs          [2,6]
embedding table   [64,32]     教学词表64个条目
查表结果           [2,6,32]
加 text positions [2,6,32]
text valid mask   [2,6]
```

ID 17 与 ID 18 的数值相近，不代表语义相近。`Embedding` 依据 ID 选取一行可学习向量；训练目标及数据影响这些向量如何组织。Tokenizer 与 embedding 是两个步骤。

若第二个样本只有 4 个有效 slots，剩余 2 个仍占 tensor 位置，但 `text_valid[1,-2:] = False`。**“占位置”与“允许其他 token 读取”分开处理。** 第 7 课再落实到 attention mask。

### 6B.2 Proprioception：连续坐标 → state token

对 normalize 后的 $p\in\mathbb R^7$：

$$
e_p=pW_p+b_p,\qquad W_p\in\mathbb R^{7\times32}.
$$

```text
state                      [2,7]
Linear(7,32)               [2,32]
unsqueeze(sequence axis)   [2,1,32]
```

7 个原始值一起构成一个 state token；本例没有为每个关节单独创建一个 token。两种设计都可能存在，读论文时要查实际实现。

Notebook state 是 7 个随机合成坐标，不能据此声称已经实现了关节或末端 pose 的物理处理。真实 state 的 normalization 统计应仅来自训练集；缺失状态也应有明确约定。

### 6B.3 现在可以拼接了

```text
image tokens       [2,32,32]
text tokens        [2, 6,32]
图文 prefix P      [2,38,32]   沿 sequence 轴 concat
state token S      [2, 1,32]
当前可用条件 P + S [2,39,32]
```

沿 sequence 轴 concat 需要每个 token 的 width 兼容。本教学网络使用相同 width；某些模型有不同 expert 的 residual width，再通过投影实现 attention 维度兼容，不能从本例推断所有 VLA 都要全程同宽。

**手算练习。** 其余配置不变，只保留 1 台相机：image、prefix、P+S 的 shape 分别是什么？只把 patch 从 8 改成 4 呢？

**答案提示。** 单相机为 `[2,16,32]`、`[2,22,32]`、`[2,23,32]`。两相机且 patch=4 时，每相机 64 个 patches，三者为 `[2,128,32]`、`[2,134,32]`、`[2,135,32]`。改 patch 还会改 projection 的输入大小和位置 embedding 长度；仅改一处常量后应检查模型构造与 masks。

**6B 完成标准：** 能从 RGB 追到 `[2,38,32]`，解释每次 flatten / transpose / concat 改了哪条轴；能指出 shape 推导里哪些值来自数据、哪些来自设计选择。

## 6C：从“能拼起来”走向“确实能用”

### 6C.1 Concat 只是排列，融合还需要信息交互

刚拼接时，一个图像 token 仍来自图像编码，文字 token 仍来自文字编码。后续 attention 可以让读取者根据当前条件聚合其他位置的信息；这一步使某个位置的表示依赖其他模态。第 7 课将具体写出 Q/K/V。

| 融合方式 | 计算路径 | 阅读时要追问 |
|---|---|---|
| token-level fusion | 各模态编码 → compatible tokens → attention | 哪些 tokens 允许互读，如何处理缺失输入？ |
| conditioning | context 通过 cross-attention、FiLM 等影响动作网络 | Q/K/V 来自哪里，条件注入哪些层？ |
| downstream interface | RFM action 保留；context 与实时物理反馈进入 sidecar | 哪些信息缓存，哪些实时更新，最终谁生成控制指令？ |

FP2 的 compressor 从 RFM context 提取缓存表示；force policy 在下游读取 wrench/proprioception。实时 force 没有进入其 compressor。详见 [FP2 tensor 数据流](03c_fp2_tensor_walkthrough.md)。一般 VLA 是否读取 force，仍需查它自己的 observation interface。

### 6C.2 为什么 representation loss 不能直接证明控制成功

Reconstruction loss 问“能否恢复输入或 feature”；contrastive loss 根据正负对及训练目标塑造表示；policy loss 问“是否预测专家动作”。三者的监督信号不同。

图像重建更清楚，可能仍忽略决定夹爪开合的细微接触信号。语言 embedding 改变了，也不保证 action 分支实际利用了它。因此要从表征指标继续检查动作和闭环行为。

### 6C.3 设计一个可解释的小实验

对同一个包含红杯、蓝杯的场景，保持图像和 state 一致，将指令分别设为“拿红杯”与“拿蓝杯”。先声明两条指令在此场景都合理，再比较动作是否朝对应目标变化。

如果动作相同，可能是模型没有使用语言，也可能是输入预处理把两条指令处理成相同 tokens，或者动作最初几步本来相同。应继续查 token IDs、中间表示和整段/闭环行为，不能只凭一个首步动作下结论。

将 force 随机 shuffle 能检查模型对输入扰动的敏感性，但会制造不符合当前接触状态的输入。性能下降同时可能来自分布外破坏；应结合训练时一致的 ablation、任务阶段和闭环指标解释。

缺失相机、文本 padding、历史 padding 也不同：一个是传感器不存在/失效，一个是字符串长度不足，一个是 episode 开头历史不足。它们都可能用 mask，但不能只填零而不解释含义；零力本来就可能是真实观测。

### 6C.4 Colab 只做到本课需要的范围

打开 [VLA01](../labs/notebooks/VLA01_token_action_trace.ipynb)，先读“2. 合成 observation/action contract”和“3. 三条输入支路与 P/S/A block mask”，找到 `images`、`self.patch`、`self.text`、`self.state` 与 `prefix`。

模型类位于同一个 code cell，因此运行该 cell 会执行后续 action 支路；本课先解释 image/text/state/prefix 四行输出即可。代码中的 noisy actions、flow target 与 sampler 留给第 9 课。不要因暂时不懂后半部分就跳过本课的 shape 推演。

该 notebook 是随机小模型的数据流与梯度检查，不加载预训练 VLM，不训练语言理解。本课编辑没有执行 notebook；云端实际结果需运行后再记录。

**6C 完成标准：** 提出一个模态使用假设，说明固定什么、改变什么、观察什么，以及结果不能排除什么；能把 FP2 的实时 force 放到正确支路。

## 面试时怎样组织答案

**题目：VLA 怎样融合 RGB、语言和机器人状态？**

先给一句结论：“先把不同来源的数据编码为可交互的表示，再通过明确的信息依赖生成动作；每种数据的时间、单位和有效性也属于接口。”接着用本课例子支持：两个相机产生 32 个视觉 tokens，6 个文本 slots 组成 38-token prefix，7 维 state 投影为一个 token。最后说明这只是教学配置，真实架构可能使用其他 encoder 或不同 expert。

追问“统一到 32 维是否就理解了物理含义”，应回答：32 维使某些矩阵运算兼容；语义依赖训练数据与目标，物理一致性还依赖单位、坐标系、动作定义和同步。用 degree/rad 的同 shape 反例说明，而不是只说“需要 normalization”。

90 秒英文练习可按三句扩展：

> I first specify which observations are available at the decision time. I then trace each modality into tokens, including the sequence length, feature width, and validity mask. Finally, I verify how the action network uses those conditions and test modality use with controlled interventions.

课后在 [学习进度](../planning/progress.md) 写下三样证据：手画 shape 链、一个错误输入反例、自己的口述录音或文字。能独立解释后进入第 7 课。

参考：[Octo 原始项目](https://octo-models.github.io/)、[OpenVLA 原始项目](https://openvla.github.io/)。需要补表示学习背景时查 [M7](ml_interview/07_representation_learning.md)。
