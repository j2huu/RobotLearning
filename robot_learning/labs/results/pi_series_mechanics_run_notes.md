# Pi 专题机制实验：助手运行记录

日期：2026-10-17。运行环境：Python 3.9.6、macOS / arm64。仅使用 Python 标准库；未安装第三方依赖。

本记录证明 [脚本](../pi_series_mechanics.py) 可运行，不代表学习者已完成验收，也不是 π0 / π0.5 模型训练或机器人复现。配套 [专题讲义](../../notes/pi_series/README.md)。

## 1. 命令与配置

从仓库根目录执行：

```bash
python3 robot_learning/labs/pi_series_mechanics.py --output robot_learning/labs/results/pi_series_mechanics.json
```

默认 `seed=7`、`steps=10`。机器可读结果：[pi_series_mechanics.json](pi_series_mechanics.json)。

## 2. 结果与解释

| 检查 | 实际结果 | 支持的解释 |
|---|---:|---|
| 路径导数的有限差分最大误差 | 1.59e-10 | 线性路径的导数与 target 一致 |
| 正向 / 反向时间的端点最大误差 | 各 8.88e-16 | 同时改变 target 与 dt 方向，得到同一端点 |
| 两种约定的输出差异 | 0 | 该教学计算下完全一致 |
| 只翻 target 符号的端点误差 | 6.8965 | target 与积分方向必须匹配 |
| 正确两层 mask 下，改 FAST target 对连续动作位置的影响 | 0 | 直接和间接答案路径均被隔离 |
| 错误 prefix mask，一层时的影响 | 0 | 只检查一层可能漏掉间接依赖 |
| 错误 prefix mask，两层时的影响 | 0.33867 | FAST→prefix→continuous 路径产生泄漏 |
| 修改 padding / 未来 FAST token 的受保护输出变化 | 各 0 | padding 与 AR 因果限制生效 |

延迟算例：50 Hz、每执行 10 个动作重规划，窗口为 200 ms。120 ms 对应 6 个动作间隔；260 ms 比窗口多 60 ms；1200 ms 已超过 50 个动作覆盖的 1 秒。异步预算忽略其他开销，只表示必要的时钟条件。

另外检查了 flow 的 `(seed,steps)=(0,1)、(19,5)、(7,10000)`，端点误差均小于 1e-10；单步和更多步数都没有改变时间约定的一致性结论。这里的常速度由已知端点构造，不用于声称更多采样步数能提高真实模型质量。

## 3. 局限与学习者任务

实验没有 learned vector field、真实 Transformer、gradient routing、接触动力学、传感器或实际 scheduler。它验证的是公式、依赖路径和时钟算术，不能输出机器人成功率或评价 π0.5 泛化。

学习者独立完成：

1. 用一维数字解释 target/dt 的符号，并解释为什么 oracle 端点不能作为推理输入。
2. 手画两层间接泄漏路径，再改一个 mask 检查自己的预测。
3. 为延迟加入 camera age、网络 jitter 和 transition 开销，说明预算如何变化。
4. 将这些检查映射到固定 openpi revision 的具体函数，并记录仍未验证的真实模型行为。
