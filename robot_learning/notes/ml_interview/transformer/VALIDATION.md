# Transformer 专题检查记录

日期：2026-11-18。范围仅为本次新增 T1–T8 专题及 ML 目录、M8 和原题卡的入口链接；仓库其他已有未提交改动未纳入本次验证。

## 实际执行

使用本机 Python 3.9.6，读取 [TR01](notebooks/TR01_attention_masks_and_cache.ipynb) 的代码 cells，在同一个新建 namespace 中按顺序执行。共 19 cells，其中 9 个代码 cells，均先通过 Python AST 解析。

| 机制检查 | 结果 |
|---|---|
| `[0,ln3]` attention 与屏蔽手算 | 8 与 2，断言通过 |
| 两层未来扰动及全可见错误对照 | 正确 mask 隔离过去；错误对照产生变化 |
| Padding key 隔离 | 有效输出保持不变，错误对照产生变化 |
| C←P←D 间接路径 | 一层未泄漏、两层泄漏；切断路径后恢复 |
| 全屏蔽行 | 按设计抛出 ValueError |
| 二维 RoPE 相对位置 | 固定向量手算断言通过 |
| 两层 full 与逐 token cache | 最大绝对差 0.0，仅限该 toy 输入/算术 |
| 非方阵 cache 位置 mask | 正确全局 mask 等价；局部三角错误对照不等价 |

Notebook 输出与 execution counts 保持空白，便于学习者独立执行；本文件记录上述本机执行证据。它不表示已经通过 Jupyter/Colab 平台端到端运行，也不表示学习者已经运行。

## 静态检查范围

检查通过：16 个 Markdown 文件（专题 13 个、现有入口 3 个）、101 个本地链接、4 个讲义 Python 代码块；Notebook 生成结果重复生成后 SHA-256 一致。

- 专题 Markdown、本次新增入口的本地文件链接及页内锚点。
- Python generator、Notebook code cells 与讲义 Python 代码块的语法。
- Notebook JSON 结构、cell IDs、输出留空及 generator 再生成一致性。
- 教学数字：attention 权重与 score 梯度、norm、参数数量、KV bytes、显式 score 矩阵容量。

## 未执行与边界

没有执行 T8 的 PyTorch 参考代码，没有运行 PyTorch attention kernel、完整 Transformer 训练、GPU benchmark、Colab 或机器人实验。两层 cache toy 没有集成 RoPE，RoPE 仅单独验证相对位置恒等式。所有结果均为固定小输入的机制检查，不代表模型性能或完整算法复现。

## 维护

Notebook 来源为 [生成脚本](tools/build_notebook.py)，修改示例应同步维护脚本，再生成 notebook，避免下次生成覆盖手工改动。从仓库根目录执行：

```bash
python3 robot_learning/notes/ml_interview/transformer/tools/build_notebook.py
```

生成脚本只写本专题的 `.ipynb`，不运行训练、不安装依赖。学习者运行后应另记录环境版本、修改变量和观察结果。
