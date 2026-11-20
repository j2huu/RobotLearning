# ML 面试补强专题：M1–M10

新增日期：2026-10-19。接在已学 BC、FP2 和 Pi 专题之后补基础；使用 M 编号，不改动主线第 6–19 课或 P1–P6 的课序。面向通用机器学习面试，同时要求能把概念落回机器人数据、action loss、force feedback 与模型部署。

代码入口是 [Colab notebooks](../../labs/notebooks/README.md)。标准扩展名是 `.ipynb`。每份 notebook 自带代码和合成数据，上传一份即可运行，不需要本地 Python、仓库路径、模型权重或 Drive 挂载。默认用 Colab CPU；学习者运行后再记录耗时与结果。

## 课程与实验

| 课 | 讲义 | 需要掌握的主问题 | Colab 实验 |
|---|---|---|---|
| M1 | [概率、统计与学习目标](01_probability_and_objectives.md) | Bayes、MLE/MAP、期望风险、MSE/CE/NLL、bias/variance | [ML01](../../labs/notebooks/ML01_linear_logistic.ipynb)、[BC](../../labs/notebooks/Lab01_probabilistic_bc.ipynb) |
| M2 | [线性与逻辑回归](02_linear_and_logistic.md) | 梯度、正则化、特征缩放、分类阈值 | [ML01](../../labs/notebooks/ML01_linear_logistic.ipynb) |
| M3 | [经典模型与选择](03_classical_models.md) | KNN、SVM、树、Random Forest、Boosting 的假设与取舍 | [ML02](../../labs/notebooks/ML02_models_and_evaluation.ipynb) |
| M4 | [Tensor 与反向传播](04_tensors_and_backprop.md) | broadcasting、chain rule、softmax CE、autograd | [ML03](../../labs/notebooks/ML03_autograd_optimization.ipynb) |
| M5 | [优化与训练诊断](05_optimization_and_training.md) | SGD/AdamW、初始化、归一化、正则化、训练故障 | [ML03](../../labs/notebooks/ML03_autograd_optimization.ipynb) |
| M6 | [数据、指标与实验设计](06_evaluation_and_leakage.md) | group split、PR/ROC、阈值、校准、置信区间与泄漏 | [ML02](../../labs/notebooks/ML02_models_and_evaluation.ipynb)、[数据窗口](../../labs/notebooks/Lab02_episode_windows.ipynb) |
| M7 | [PCA、聚类与表示学习](07_representation_learning.md) | SVD、K-means、AE、contrastive、shortcut | [ML04](../../labs/notebooks/ML04_pca_kmeans.ipynb) |
| M8 | [Attention、Transformer 与压缩](08_attention_and_compression.md) | Q/K/V、形状、mask、cache、FP2 compressor | [ML05](../../labs/notebooks/ML05_attention_compressor.ipynb)、[Pi 机制](../../labs/notebooks/Pi_mechanisms.ipynb) |
| M9 | [生成模型与动作分布](09_generative_models.md) | mixture、CVAE、diffusion、flow 的训练与采样 | [ML06](../../labs/notebooks/ML06_diffusion_flow.ipynb)、[BC](../../labs/notebooks/Lab01_probabilistic_bc.ipynb) |
| M10 | [迁移、LoRA 与面试系统题](10_transfer_and_system_design.md) | freeze/grad、LoRA、ablation、端到端方案与边界 | [ML03](../../labs/notebooks/ML03_autograd_optimization.ipynb)、[ML05](../../labs/notebooks/ML05_attention_compressor.ipynb) |

另有 [40 道面试题与评分要点](interview_drill.md)。讲义是可学习材料，不代表已学完或已通过面试验收。

## Transformer 由浅入深专题

2026-11-18 新增 [Transformer 专题 T1–T8](transformer/README.md)：8 课、24 个小单元，从 attention 手算进入完整 block、位置编码/RoPE、mask、训练目标、KV cache、推理效率及 coding/system 面试。M8 保留为概览与 FP2 应用入口。

建议从 [T1：从加权平均到多头 Attention](transformer/01_attention_from_scratch.md) 顺序推进；重点疑问可直接读 [T4：causal / padding / loss / prefix mask](transformer/04_masks_and_information_flow.md) 和 [T6：KV cache 与非方阵 mask](transformer/06_kv_cache_and_decoding.md)。另配 [40 道 Transformer 题卡](transformer/interview_drill.md)、[速查表](transformer/cheatsheet.md)及自包含 [TR01 机制 Notebook](transformer/notebooks/TR01_attention_masks_and_cache.ipynb)。检查范围见 [VALIDATION](transformer/VALIDATION.md)，材料准备与本人掌握、Colab 运行分别记录。

## 面试时间紧时的顺序

建议按知识点安排，不强求十课顺序通读。以下是学习时间预算，不是 notebook 实测运行耗时。

| 可用时间 | 优先内容 | 必须留下的证据 |
|---|---|---|
| 3 天，每天 2–3 小时 | M1/M2 → M4/M5 → M6；最后口述 M8 的 FP2 | 手写 logistic gradient、解释训练循环、正确区分 validation/test、画出 Q/K/V shapes |
| 7 天，每天 2 小时 | 前三天基础；第 4 天 M3；第 5 天 M7/M8；第 6 天 M9/M10；第 7 天模拟面试 | 至少 3 份本人执行的 notebook、3 个失败反例、一段英文架构说明 |
| 完整补强，约 20–30 小时 | M1–M10 + 6 个 ML notebook + 题卡 | 每课一个推导、一个消融、一个可解释结果 |

已有第 4/5 课不必重做相同内容：BC 的概率推导对应 M1/M9，episode split 对应 M6，FP2 的 learned query 对应 M8。主线第 16–18 课继续负责 RL/offline RL/MPC，本专题不把它们伪装成已完成的 ML 实验。

## 每课的学习方式

1. 先回答讲义开头的问题，再看公式。
2. 手算一个小例子，写清 shape、目标和假设。
3. 在 Colab 从干净 runtime 按顺序运行 notebook；只改一个实验变量。
4. 写出结果支持什么、不支持什么，再做题卡追问。
5. 在 [学习记录](../../planning/progress.md) 填本人证据。助手静态检查不计作本人运行，也不计作论文复现。

## 资料依据

- [Stanford CS229](https://cs229.stanford.edu/main_notes.pdf)：监督学习、概率与优化推导。
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/)：概率与统计建模。
- [PyTorch Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)、[Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)：训练图与优化接口。
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)、[common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：经典模型与评估实践。
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)：上传 `.ipynb`、云端执行和保存方式。

讲义采用 Obsidian 兼容的 `$...$` 和独占行的 `$$` 数学格式。代码是教学实现，固定输入或合成数据不代表真实机器人、完整 VLA 或 FP2 训练。
