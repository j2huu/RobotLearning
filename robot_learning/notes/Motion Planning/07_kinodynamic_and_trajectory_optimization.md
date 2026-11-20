# MP7 Kinodynamic Planning、轨迹优化与 MPC

[专题目录](README.md) · [上一课 时间参数化](06_time_parameterization.md) · [下一课 在线换轨](08_online_replanning_and_execution.md)

## 动力学什么时候要进入规划

几何路径加定时适合能沿配置曲线自由调速的任务。有些问题同时要求起始速度、制动距离、输入限幅、非完整约束或动态避障，导致几何上可连的两个点之间缺少可行运动。此时常把节点从位置改成状态 $x=(q,\dot q)$，局部连接从直线改成控制作用下的 rollout。动态障碍还需要在碰撞查询里写清执行时间。

- **Kinodynamic search。** 对有约束的车辆可用一组满足模型的 motion primitive，组成 lattice 再搜索。采样式方法可采样控制及持续时间并积分，或用满足模型的局部求解器连接状态。可达性可以是有方向的，尤其是含速度、不可倒车或时变约束时。
- **Trajectory optimization。** 把一条初始参考变成满足模型与代价的轨迹。可以同时调路径、状态、控制和时长，常用于利用已有路径做局部优化，也可以用多初值寻找不同绕行路线。
- **MPC。** 每个规划周期用当前或预测的实际状态重新求有限时域轨迹，只提交短前缀。它可以跟踪全局层给出的参考，也可以在预测窗口里本身承担局部避障。需要考虑计算截止时间、终端设计和超时后的已验证行为。

[LaValle 第 13–14 章](https://lavalle.pl/planning/) · [Tedrake 的 Trajectory Optimization 讲义](https://underactuated.mit.edu/trajopt.html)

## 写出一个可求解的离散问题

给定 $x_0=\hat x(t_0)$，以 $x_k,u_k$ 为变量：

$$
\begin{aligned}
\min_{x,u}\quad & \phi(x_N)+\sum_{k=0}^{N-1}\ell(x_k,u_k)\\
\mathrm{s.t.}\quad & x_{k+1}=f_d(x_k,u_k),\\
& x_k\in\mathcal X_k,\quad u_k\in\mathcal U,\quad x_N\in\mathcal X_f .
\end{aligned}
$$

$\ell$ 可以含跟踪误差、控制开销或平滑代价。$\mathcal X_k$ 可以含速度、关节范围与在 $t_k$ 的全身碰撞距离。若把碰撞设成软代价，结果仍需查碰撞；若硬约束只放在节点，同样需要检查段间。姿态误差可用姿态空间的局部误差表示，避免把四元数的两种符号当成不同姿态。

| 离散思路 | 变量和约束 | 适合用来理解的权衡 |
|---|---|---|
| Single shooting | 优化控制 $u_{0:N-1}$，从 $x_0$ 积分得到状态 | 变量少，早期控制对远期状态的敏感度可能很大 |
| Multiple shooting | 各段可有独立起始状态，段内积分，段间用等式匹配 | 变量更多，得到局部稀疏结构，并能直接初始化各段状态 |
| Direct transcription | 把各节点的状态与控制作为变量，用离散动力学限制相邻点 | 容易加节点约束。Direct collocation 是一类做法，在选定节点或内部配点约束轨迹多项式和动力学的一致性 |

非凸碰撞约束下，局部求解可能卡在原路径的绕行侧。先用 MP2/MP3 给出不同几何候选，再优化和验证，是一种可解释的组合。MPC 的有限窗口也会影响看到的绕行机会。[MIT Underactuated Robotics](https://underactuated.mit.edu/trajopt.html)

## 手算双积分器

取 $x=(q,v)$、控制 $u=\dot v$，$|u|\le1$。假设每个 $0.5$ s 区间内 $u$ 恒定：

$$
v_{k+1}=v_k+u_k\Delta t,\qquad
q_{k+1}=q_k+v_k\Delta t+\tfrac12u_k\Delta t^2.
$$

从 $(0,0)$ 到 $(1,0)$，序列 $(1,1,-1,-1)$ 给出节点 $(0.125,0.5),(0.5,1),(0.875,0.5),(1,0)$。几何上只有一段直线，状态与控制序列仍要包含加速和制动。若用一个 $0.5$ s 区间完成，终端静止会迫使该段 $u=0$，因此无法移动 1 rad。这个二阶模型允许加速度阶跃，若要限制 jerk，可把加速度纳入状态，用 jerk 作控制。

**自测。** 如果有限时域的优化频繁超时，为什么只放宽碰撞代价会掩盖问题？它可能把违反碰撞约束的参考作为可用结果。应区分计算超时、模型不可行、初值差以及实际约束冲突，并验证求解器最后提交的那段参考。
