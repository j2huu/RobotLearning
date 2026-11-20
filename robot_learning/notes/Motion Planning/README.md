# Motion Planning 与 Trajectory Generation

独立的 **MP1–MP8** 专题，适合已经学过 A*、RRT 和基础运控，想把搜索、轨迹生成与执行重新串起来的人。[45 分钟速记](quick_review.md) 保留了原来的公式、两关节算例和 8 道题。

```mermaid
flowchart LR
    A[场景与目标集合] --> B[离散或采样规划]
    B --> C[碰撞验证与路径平滑]
    C --> D[定时与动态约束]
    D --> E[轨迹采样与跟踪]
    E -->|状态估计与环境变化| A
```

| 课 | 内容 | 复习后能独立完成 |
|---|---|---|
| [MP1 配置空间与碰撞](01_configuration_space.md) | C-space、状态/边验证、IK 目标集合 | 用线段距离构造端点合法但边碰撞的反例 |
| [MP2 A* 与图搜索](02_astar_and_graph_search.md) | 启发式、open/closed、reopen、lattice | 手推一次 A*，解释何时必须重新打开节点 |
| [MP3 RRT、RRT-Connect、RRT* 和 PRM](03_sampling_based_planning.md) | 局部扩展、双向连接、重连、采样保证 | 手算一次 extend 和一次 rewiring，说明窄通道为什么慢 |
| [MP4 路径平滑与 Cartesian 路径](04_path_geometry_and_smoothing.md) | shortcut、Bezier 圆角、连续性、TCP 路径 | 构造安全圆角，解释如何保住几何余量 |
| [MP5 点到点与多段轨迹](05_point_to_point_trajectories.md) | 三次、五次、梯形速度、jerk 限幅 S 曲线 | 给定边界状态和上限，算出可行的段时间 |
| [MP6 路径时间参数化](06_time_parameterization.md) | 路径导数、力矩限制、TOPP-RA、与 OTG 对照 | 手算一个路径网格的前向速度和后向制动约束 |
| [MP7 Kinodynamic Planning 与轨迹优化](07_kinodynamic_and_trajectory_optimization.md) | 动态状态、运动基元、shooting、transcription、MPC | 写出带动力学、碰撞和初末状态的离散优化问题 |
| [MP8 在线换轨与验证](08_online_replanning_and_execution.md) | 交接状态、时效、参考与实测、多速率执行 | 根据时间轴判断可接哪段，解释冲击来自哪里 |

**第一轮用 2–3 次复习。** 先做 MP2、MP3 的手算题，恢复搜索过程。再并读 MP4–MP6，把避障、圆角和限幅串成一次规划。最后用 MP7–MP8 解释动态约束什么时候进入搜索、当前轨迹怎样交接。MP1 可作为建模速查。复习 IK、奇异性和力矩公式时回查 [Robotics Fundamentals](../Robotics%20Fundamentals/README.md)。

## 贯穿算例

两关节配置 $q=(q_1,q_2)$，单位 rad。令起点 $A=(0,0)$，目标 $B=(2,0)$，无效区域为以 $O=(1,0)$ 为圆心、半径 $0.5$ 的闭圆盘。$A\to B$ 穿过障碍，可先用 $A\to W\to B$，其中 $W=(1,1)$。每个关节的速度、加速度和 jerk 上限取 $1\,\mathrm{rad/s}$、$2\,\mathrm{rad/s^2}$、$10\,\mathrm{rad/s^3}$。

各课逐步回答四个问题。搜索怎么拿到一条可行的路？如何过 $W$ 又保持余量？每段最快能怎么走？收到新目标后，在实际执行时刻还能不能接上？圆盘是教学构造，实际机械臂通过全身几何与碰撞检测来实现这些判定。

**综合口述。** 给定一次七轴机械臂避障任务，讲清目标集合、局部连接、首解和后处理、时间律、跟踪和换轨。每个环节给一个可测指标，例如首解耗时、最小 clearance、峰值力矩、跟踪误差或观测到执行的延迟。
