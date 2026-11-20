# Robotics Fundamentals：机器人基础轻量复习

建立日期：2026-11-03。与 [ML](../ml_interview/README.md)、[RL](../Reinforcement%20Learning/README.md) 并列，使用独立 **B0–B6** 编号。先分清任务变量与动作接口，再把机械臂的位置、速度、力和控制串起来。

**从 [B0 关节空间、任务空间与动作空间](00_joint_task_and_action_spaces.md) 开始。** 接着学坐标系与 FK/IK，动力学先理解各项含义，再手算一个单关节例子。

| 课 | 核心问题 | 一个必须会的小例子 |
|---|---|---|
| [B0 关节、任务与动作空间](00_joint_task_and_action_spaces.md) | 机械臂怎么摆、任务关心什么、策略发什么？ | 比较同一笔尖任务下的末端目标、关节目标和力矩接口 |
| [B1：坐标系与变换](01_frames_and_transforms.md) | 这个位置是相对谁说的？ | 旋转一个点，再加平移 |
| [B2：正运动学 FK](02_forward_kinematics.md) | 已知关节角，末端在哪里？ | 两连杆从关节角算位置 |
| [B3：逆运动学 IK](03_inverse_kinematics.md) | 想去这个位置，关节怎么摆？ | 同一个目标算出两组关节角 |
| [B4：Jacobian 与奇异点](04_jacobian_and_singularities.md) | 关节动一点，末端怎么动？ | 算末端速度，识别伸直奇异点 |
| [B5：动力学入门](05_dynamics_basics.md) | 需要多大力矩才能产生加速度？ | 单摆的惯性、重力和摩擦 |
| [B6：控制与接触入门](06_control_and_contact.md) | 怎样让实际运动跟上目标？ | PD、重力补偿、阻抗的直觉 |

复习时查 [一页速记与 14 个自测问题](cheatsheet.md)。可选代码：[RB01：两连杆 FK/IK/Jacobian 与单关节动力学](notebooks/RB01_kinematics_and_dynamics.ipynb)。

学过 IK、动力学和跟踪控制后，可以继续读独立的 [Motion Planning 与 Trajectory Generation 专题](../Motion%20Planning/README.md)。

## 先记住这五个对应关系

| 名称 | 输入 → 输出 |
|---|---|
| FK | 关节位置 $q$ → 末端位姿 $T$ |
| IK | 目标末端位姿 $T_d$ → 一组或多组关节位置 $q$ |
| Jacobian | 关节速度 $\dot q$ → 末端速度 |
| 正向动力学 | $q,\dot q,\tau$ → 关节加速度 $\ddot q$ |
| 逆向动力学 | $q,\dot q,\ddot q$ → 所需关节力矩 $\tau$ |

控制器把目标与实际状态的误差变成命令；它可以用到上面这些关系。IK 得到的是几何解，如何平滑到达、是否避障、力矩是否足够，还需要其他环节。

## 怎么同步复习

每次只读一课，按“直觉 → 一个公式 → 手算 → 两个自测问题”走。每课先留 20–30 分钟，七课约 2.5–3.5 小时，代码练习另算。

可以穿插在 RL 复习之间：

- RL 的 state/action 配 B0–B3，明确 observation、关节目标、末端目标和 frame。
- RL 的 transition：配 B4/B5，理解动作如何变成速度、加速度与下一状态。
- RL 的执行与接触：配 B6，区分策略输出与底层 controller。

第一轮只看平面两连杆。设 $l_1=l_2=1\,\mathrm m$，$q_1$ 从基座 x 轴逆时针量，$q_2$ 是第二杆**相对第一杆**的角度。公式用 rad，手算时可以用角度。B5 则另用竖直平面单摆，并重新声明角度零点。

学习后再勾选：

- [ ] B0 说清任务空间与动作空间的区别，从归一化动作算出一条关节位置目标。
- [ ] B1：说清 frame，算对一次点坐标变换。
- [ ] B2：独立写出两连杆 FK。
- [ ] B3：算出两组 IK，并用 FK 验回目标。
- [ ] B4：算对一个速度，解释一个奇异构型。
- [ ] B5：区分正/逆动力学，说清惯性、速度项和重力。
- [ ] B6：解释 PD、重力补偿和“软一点”的接触行为。

## 参考与代码状态

主要参考 Lynch & Park 的 Modern Robotics 官方材料，按需查：

| 内容 | 原始资料 |
|---|---|
| 任务空间与工作空间 | [第 2.5 节](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-5-task-space-and-workspace/) |
| 刚体运动、坐标变换 | [第 3 章](https://modernrobotics.northwestern.edu/chapters/chapter3/) |
| FK | [第 4 章](https://modernrobotics.northwestern.edu/chapters/chapter4/) |
| Jacobian、静力学 | [第 5 章](https://modernrobotics.northwestern.edu/chapters/chapter5/) |
| IK | [第 6 章](https://modernrobotics.northwestern.edu/chapters/chapter6/) |
| 动力学 | [第 8 章](https://modernrobotics.northwestern.edu/chapters/chapter8/) |
| 控制 | [第 11 章](https://modernrobotics.northwestern.edu/chapters/chapter11/) |

本专题的数字、练习和小模型是教学构造。Notebook 可单文件上传 Colab，无外部数据或仓库依赖；计算使用 Python 标准库，可选示意图使用 IPython。**当前未执行 notebook，未在 Colab 验证运行。** 检查范围见 [验证记录](VALIDATION.md)。
