# Colab Notebook 实验入口

2026-10-19 起，课程的默认运行方式改为 **Google Colab + `.ipynb`**。不需要在自己的电脑安装 Python、PyTorch、Jupyter 或训练环境。

## 怎么运行

1. 选择下面的一份 `.ipynb` 文件。
2. 打开 [Google Colab](https://colab.research.google.com/)，使用 File → Upload notebook 上传它。
3. 连接 Python runtime，保留默认 CPU 即可；从第一个 cell 开始，按顺序运行或 Run all。
4. 修改参数后，重新从相关数据/模型初始化单元开始运行；不确定状态时用干净 runtime 重新运行。
5. 实验完成后保存或下载 `.ipynb` 保留图表、输出和自己的回答。可选指标导出写到 Colab runtime 的 `outputs/`，需自行下载保留。

官方说明：[上传与执行位置](https://research.google.com/colaboratory/faq.html)。Colab 是运行环境；notebook 文件本身不包括已经训练的模型或实际运行结果。

## VLA 正课当前入口（2026-10-21）

第5课之后按 [循序学习路线](../../planning/vla_learning_path.md) 进入6A–15C。第一次只需理解当前单元相关部分；按下表逐步使用现有notebook，已经完成的实验可复用。

| 学到哪里 | 打开哪份 / 聚焦章节 | 先想清什么 |
|---|---|---|
| 6A–6C | [VLA01](VLA01_token_action_trace.ipynb) §1–3 | RGB/text/state的shape；§3末尾forward打印shape，暂不要求理解flow目标 |
| 7A–7C | VLA01 §3–5；[Pi机制](Pi_mechanisms.ipynb) 的attention检查 | Q/K/V、mask、扰动不变性与间接泄漏 |
| 8B–9C | [ML06](ML06_diffusion_flow.ipynb) §2–6 | diffusion与flow训练label、sampler可见信息 |
| 9B–11B | VLA01 §4–6 | masked loss、一次backward、采样与反归一化 |
| 13C | [VLA02](VLA02_multirate_force_execution.ipynb) §2–4 | 固定输出频率，比较不同反馈时钟与stale context |
| 14B | [ML05](ML05_attention_compressor.ipynb) §3–7 | learned query、重建、冻结、force sidecar |
| 15 | 本人保存的运行版本 | 控制变量、失败解释和结论边界 |

“聚焦章节”指阅读重点；执行仍应从环境和定义cell按顺序进行，不跳过前置变量。VLA01随机模型没有优化训练，ML06是一维合成分布，VLA02是规则式spring实验；各自能验证的范围不同。当前没有新增云端运行记录。

## 已准备的 11 份 notebook

| 文件 | 学习内容 | 数据与资源 |
|---|---|---|
| [Lab01：概率 BC](Lab01_probabilistic_bc.ipynb) | MSE/Gaussian/mixture，观察双峰采样 | 合成 episodes、小型 PyTorch MLP |
| [Lab02：数据窗口](Lab02_episode_windows.ipynb) | 因果对齐、episode split、history/chunk、padding、normalization | 合成数据；不训练模型 |
| [Pi：机制检查](Pi_mechanisms.ipynb) | Flow 符号、两层 mask 泄漏、异步时延预算 | 标准库机制代码，附 NumPy/Matplotlib 图 |
| [ML01：线性/逻辑回归](ML01_linear_logistic.ipynb) | 手写梯度、有限差分、feature scaling、Ridge 选择 | NumPy + sklearn 合成数据 |
| [ML02：经典模型与评估](ML02_models_and_evaluation.ipynb) | Logistic/KNN/SVM/tree/RF/boosting；AP/F1/ROC/校准/group leakage | sklearn 合成分类数据 |
| [ML03：Autograd 与优化](ML03_autograd_optimization.ipynb) | CE 梯度、SGD/AdamW、Dropout/BN、LoRA 首步梯度 | 小型 PyTorch 张量与网络 |
| [ML04：PCA 与聚类](ML04_pca_kmeans.ipynb) | PCA/SVD、K-means、尺度、低方差信号反例 | sklearn/NumPy 合成数据 |
| [ML05：Attention compressor](ML05_attention_compressor.ipynb) | learned query、重建训练、冻结、frame/mask/wrench sidecar | 合成 context；不加载 RFM/论文数据 |
| [ML06：Diffusion 与 Flow](ML06_diffusion_flow.ipynb) | 两个小网络的训练、DDPM/Euler 采样、模式与耗时 | 一维合成条件分布 |
| [VLA01：Token/action trace](VLA01_token_action_trace.ipynb) | RGB/text/state→P/S/A tokens→masked flow loss→sampler/反归一化 | 随机权重小模型，非预训练VLA |
| [VLA02：多速率接触执行](VLA02_multirate_force_execution.ipynb) | 分开context/sensor/policy/output时钟；反馈与stale context对比 | 规则式调节器与spring toy，不训练FP2 |

基础参考：[ML 面试 M1–M10](../../notes/ml_interview/README.md)。当前主线优先VLA01→生成机制实验→VLA02；对应第6/7/8/9/13/14课。

## 文件约定

- 每份 notebook 都包含所需函数、模型和数据生成代码；上传单个文件即可，不使用 `%run`、`git clone`、本仓库路径或用户电脑的绝对路径。
- 所有 code cell 的 `outputs=[]`、`execution_count=null`，由你在 Colab 运行产生输出。
- 默认使用云端 CPU；不强制申请 GPU，不下载大数据或 checkpoint，不挂载 Google Drive。
- 首单元打印 seed 和主要依赖版本。若自定义 Colab runtime 缺包，按开头说明仅在该云端 runtime 安装。
- 绘图用英文轴标签；讲解、实验观察与面试题用中文，保留英文术语。
- 现有三个 `.py` 文件保留为来源/历史入口，日常学习直接使用本目录的 notebook。

## 当前检查范围

已准备文件并做 JSON/nbformat 结构、代码语法、输出清空、依赖路径与链接检查。**没有在本机执行 notebook 或训练，也没有实际连接 Colab runtime。** 云端 Run all 的实际结果、版本兼容性与耗时待首次 Colab 运行确认，见 [检查记录](VALIDATION.md)。

## 维护者说明

生成器在 [tools/build_colab_notebooks.py](../tools/build_colab_notebooks.py)，负责把内嵌实验生成 JSON 文件；学习者不需要上传/运行它。原始脚本与 notebook 的 hash 记录在 [manifest](manifest.json)。修改 notebook 后如需永久保留编辑，应同时更新生成器，避免下一次重建覆盖手工改动。
