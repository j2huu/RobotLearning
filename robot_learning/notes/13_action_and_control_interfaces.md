# 第 13 课：动作、坐标系与多速率执行接口

[上一课](12_fast_and_pi05.md) · [下一课：力觉适配](14_force_aware_adaptation.md) · [Colab VLA02](../labs/notebooks/VLA02_multirate_force_execution.ipynb)

目标：从模型输出走到机器人执行，能解释 shape 相同但含义不同的动作，以及 prediction、sampling、feedback、interpolation 各是什么频率。

按 [循序路线](../planning/vla_learning_path.md) 分三次：**13A读§1–2写一条动作的完整contract；13B读§3手算参考点变化；13C读§4–6画时间轴并使用VLA02**。先声明坐标和时间约定再代数计算。课后做 [第13课面试卡](vla_interview_workbook.md)，解释轨迹平滑、反馈时效与接触稳定性的不同证据。

## 1. Action contract 必须包括什么

| 字段 | 例子与问题 |
|---|---|
| 控制变量 | joint position、TCP pose、velocity、torque、wrench？ |
| 表示 | absolute target 还是 delta；旋转用 axis-angle、quaternion 或6D code？ |
| 单位与 frame | rad/m/N；base、TCP、sensor 还是 interaction frame？ |
| 时间 | command 对应哪个执行时刻；waypoint 间隔；何时到达？ |
| gripper | continuous width、binary open/close、符号与 clipping？ |
| masks | padded action dimensions 与无效未来 waypoint？ |

一个 [B,H,7] 可能是7个关节，也可能是6D末端运动加夹爪。不能依据 shape 选 inverse kinematics 或直接发给 controller。

## 2. 从 normalized action 到 native command

```text
[B,H,d_model] sample
   → 按 action-dimension mask 选 native 维度
   → 反归一化：恢复训练统计对应的物理尺度
   → absolute/delta / joint/Cartesian / rotation conversion
   → trajectory alignment、时间戳与当前 reference
   → controller API
```

Delta 的参考点可能是当前状态、上一条 command 或 chunk 起点；三种积分结果不同。Rotation delta 要声明左乘/右乘与 frame，不能对任意旋转编码逐元素累加。

## 3. Pose、twist、wrench

Pose 属于 SE(3)；twist=[v,ω] 是瞬时运动；wrench=[f,m] 是广义力。二者配对得到机械功率 $f^Tv+m^T\omega$，不能将 m/s 与 N 直接相加。

只改变同一点处的坐标轴时，可分别用 R 旋转 linear/angular 或 force/moment 部分。改变参考点时还有耦合项：若 r 从旧点指向新点，且均在同一表达轴下，

$$
v_{new}=v_{old}+\omega\times r,
\quad m_{new}=m_{old}-r\times f.
$$

TCP/sensor 偏移会影响力矩解释。完整对偶变换与符号约定见 [twist–wrench 补充](03b_fp2_adaptor_twist_wrench.md)，不能只旋转前三个数后忽略参考点。

## 4. 多速率要逐层写表

| 频率口径 | 问题 |
|---|---|
| policy inference | 每秒重新读取条件并生成几次 chunk？ |
| waypoint sampling | chunk 内相邻参考点间隔多久？ |
| force-policy update | 每秒重算几次 frame/mask/wrench？ |
| sensor availability | 多久获得一次新的可用测量，有多少延迟？ |
| interpolation/output | 每秒输出多少平滑采样点？ |
| control law / plant | 反馈律与硬件/仿真积分各在什么周期执行？ |

此前用户实现说明将1kHz标为插值后输出，论文III-D另报内部低层控制律1kHz。2026-11-09核对的peg源码则只确认source15Hz、planner/fast目标50Hz、F/T buffer轮询200Hz；不能把历史口径当成当前Python循环。`max_infer_dt=.5s`也不等于实测2Hz，详见 [3d §10](03d_fp2_source_architecture_and_training.md)。

## 5. Chunk alignment 与连续性

推理耗时后，新 chunk 的起点可能已经过时。用 recent execution history 识别当前相位，按明确规则选/丢 waypoint，再连接轨迹。DTW 是一种对齐思路，不是对任何接触都安全的通用删点操作。

Quintic spline 等方法可以约束连接的 position/velocity/acceleration 连续性。SO(3) 旋转、二值 mask、gripper switch 与 wrench 不应默认采用相同的逐元素插值。

论文的quintic spline与所选peg代码的linear+SLERP分别记录；当前`interp_continuous_dynamics`转调linear，不可仅凭函数名宣称实现了速度/加速度连续性。

## 6. VLA02 教学实验

合成平面/曲面接触，用 spring 关系计算 force；固定相同的1 kHz输出网格，比较 position-only、slow feedback、fast feedback 与 stale context。Sensor availability、policy update 和 output grid 独立配置。

该实验是规则式 controller 与简化环境，不学习 force policy，不是 Flexiv/FP2 低层控制器或真实机器人仿真。相同增益用于隔离机制，不代表每个频率的最佳调参结果。

## 验收

为一条7维动作写完整 contract；手算一个参考点平移对 moment 的影响；解释1 kHz输出为什么不等于1 kHz新决策；列出过时 chunk 的处理风险与评价方式。
