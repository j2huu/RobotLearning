# B6：控制与接触——知道目标后，怎样跟上

[目录](README.md) · [上一课：动力学](05_dynamics_basics.md) · [速记与自测](cheatsheet.md)

## 控制器把误差变成命令

以力矩接口为例，关节 PD 加重力补偿：

$$
\tau_{cmd}=K_p(q_d-q)+K_d(\dot q_d-\dot q)+g(q).
$$

P 像弹簧，把关节拉向目标；D 像阻尼，抑制速度误差；重力补偿提供托住所需的基础力矩。该式没有完整补偿惯性/速度耦合，不等于精确跟踪任何快速轨迹。

手算单关节：$q_d=0.5,q=0.2\,\mathrm{rad}$，$\dot q_d=0,\dot q=0.1\,\mathrm{rad/s}$，$K_p=10,K_d=2$，则反馈项为 $10(0.3)+2(-0.1)=2.8\,\mathrm{N\,m}$，再加当前构型的重力补偿。

如果机器人接口接收关节位置目标，而非 torque，底层驱动器还会负责另一层控制。策略输出的 pose/action 不是最终电机力矩。

## 位置目标还需要时间安排

IK 给出目标 $q_d$；轨迹生成再安排 $q_d(t),\dot q_d(t),\ddot q_d(t)$。直接跳到很远的位置目标可能产生很大的误差和命令。路径 path 说明经过哪里，轨迹 trajectory 还说明何时经过。

```text
末端目标 → IK → 关节目标 → 轨迹 → controller → actuator → 实际运动
                                      ↑                       ↓
                                      └──── 状态反馈 ──────────┘
```

## 接触时：阻抗的直觉

位置控制强调到达目标；阻抗控制希望外力与运动偏差呈现指定的质量、弹簧、阻尼关系。固定目标的一维理想形式为：

$$
M_d\ddot e+D_x\dot e+K_x e=f_{ext},\qquad e=x-x_d.
$$

这里 $M_d$ 是希望呈现的惯性，$D_x$ 是阻尼，$K_x$ 是刚度。这是在描述期望的接触行为，不是直接把机器人的物理质量换掉。

静态时 $e=f_{ext}/K_x$：10 N 外力下，刚度 1000 N/m 对应 1 cm 偏移，100 N/m 对应 10 cm。较小刚度更容易顺着外力让开，但不自动保证任务成功或接触稳定。

导纳 admittance 常见形式是测量外力，再生成运动参考；阻抗常通过运动误差生成力/力矩，具体实现还依赖机器人已有接口。

## Jacobian 也把末端力映射成关节力矩

对 B4 的平面点、同一 frame 中的外力：

$$
\tau_{ext}=J^T f_{ext}.
$$

来源是功率一致：$\tau_{ext}^T\dot q=f_{ext}^T\dot p$。本式定义 $f_{ext}$ 为**环境施加在机器人末端上的力**。若静止时需要平衡它，执行器还应满足 $\tau_{cmd}=g(q)-\tau_{ext}$；不要把外力与机器人对环境的反作用力混用。

B4 中 $q=(0,\pi/2)$、$f_{ext}=(0,10)\,\mathrm N$，得到 $\tau_{ext}=(10,0)\,\mathrm{N\,m}$。完整三维 wrench 还包含力矩，必须与 Jacobian 的 frame、参考点和分量顺序一致。

## 自测

1. FK/IK 正确，是否就保证机械臂跟踪正确？**还需时间安排、反馈控制、执行器能力和正确接口。**
2. 为什么位置误差几乎为零时仍可能需要非零 torque？**例如补偿重力或平衡接触外力。**

回接已有课程：[动作与控制接口](../13_action_and_control_interfaces.md)、[RL 机器人诊断](../Reinforcement%20Learning/11_robotics_and_diagnostics.md)。参考：[Modern Robotics 控制](https://modernrobotics.northwestern.edu/chapters/chapter11/) 与 [Jacobian/静力学](https://modernrobotics.northwestern.edu/chapters/chapter5/)。
