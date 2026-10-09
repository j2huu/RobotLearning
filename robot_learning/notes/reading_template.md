# 论文标题

- 文件/链接：
- 阅读日期：
- 路线位置：M__ / S__
- 一句话结论：

## 1. 论文解决的问题

失败场景是什么？现有方法为什么不够？

## 2. 系统接口

| 项目 | 内容 |
|---|---|
| observation | |
| language/goal | |
| action space | |
| action horizon | |
| model update frequency | |
| control frequency | |
| low-level controller | |
| training-only signals | |

## 3. 方法

用一张自己画的 data-flow 图解释，不复制论文总图。

## 4. 目标函数

逐项写出符号含义、tensor shape、监督信号来源。

## 5. 实验证据

- 最能支持核心主张的实验：
- 最重要的消融：
- 数据规模、任务数、机器人、成功率口径：

## 6. 局限与可迁移结论

- 方法依赖的传感器、控制器或数据假设：
- 未被实验排除的替代解释：
- 可以迁移到我的问题中的设计：
- 与上一范式相比，真正改变了什么：

## 7. 复现检查

- [ ] 能从 raw data 说到 model input
- [ ] 能从 loss 说到 inference
- [ ] 能从 predicted action 说到 robot command
- [ ] 能说明系统在何处闭环
- [ ] 能指出至少一个失败模式
