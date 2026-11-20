# B1：坐标系与位姿——先问“相对谁”

[目录](README.md) · [上一课 B0 关节、任务与动作空间](00_joint_task_and_action_spaces.md) · [下一课：FK](02_forward_kinematics.md)

## 直觉

“杯子在 $(1,0,0)$”还不完整：相对相机、桌子还是机械臂基座？同一个物理点，在不同 frame 里会有不同坐标。常见 frame 有 world、base、camera、tool。

位置 position 描述原点在哪里；姿态 orientation 描述轴朝哪里；两者合起来是位姿 pose。关节角向量 $q$ 描述机器人构型，通常不等于末端位姿。

## 只掌握一种变换约定

采用列向量。${}^{A}T_B$ 将 **B 中的点坐标换成 A 中的坐标**：

$$
{}^A p={}^A R_B\,{}^B p+{}^A t_B,
\qquad
{}^A T_B=\begin{bmatrix}{}^A R_B&{}^A t_B\\0&1\end{bmatrix}.
$$

三维时 $R$ 是 $3\times3$，$t$ 是 $3\times1$，$T$ 是 $4\times4$。$t$ 是 B 原点在 A 中的位置。齐次坐标写为 $[p^T,1]^T$。

本课平面例子用二维点、$2\times2$ 旋转和 $3\times3$ 齐次变换：

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}.
$$

合法旋转满足 $R^TR=I,\det R=1$。只有表达在同一 frame 的量才能直接相加比较。方向向量换 frame 只乘 R，点坐标还要平移。

## 手算：工具坐标换成基座坐标

工具 B 相对基座 A 逆时针转 $90^\circ$，工具原点在 A 中为 $(1,0)\,\mathrm m$。工具中的点为 ${}^Bp=(1,0)\,\mathrm m$：

$$
{}^Ap=\begin{bmatrix}0&-1\\1&0\end{bmatrix}
\begin{bmatrix}1\\0\end{bmatrix}+\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}1\\1\end{bmatrix}\,\mathrm m.
$$

先看轴朝向，再看原点位置。不要把平移向量也当成在 B 中表达。

串联变换按中间 frame 对齐：${}^AT_C={}^AT_B\,{}^BT_C$。逆变换为：

$$
({}^AT_B)^{-1}=\begin{bmatrix}R^T&-R^Tt\\0&1\end{bmatrix}.
$$

矩阵乘法通常不可交换。Euler 角顺序、四元数的分量顺序都要查具体约定；四元数不可以直接按四个分量相减来定义一般姿态误差。第一轮先用旋转矩阵，不展开三维姿态运算。

## 自测

1. 上例 B 原点在 A 中是什么？**$(1,0)$，不是 $(1,1)$；后者是另一个点。**
2. 已知相机相对基座、物体相对相机的位姿，如何得到物体相对基座的位姿？**${}^{base}T_{object}={}^{base}T_{camera}\,{}^{camera}T_{object}$。** 按中间 frame 对齐相乘。

能口述：“做 FK、IK 或控制前，我先确认目标和实际状态的坐标系、单位与变换方向。”

参考：[Modern Robotics 第 3 章](https://modernrobotics.northwestern.edu/chapters/chapter3/)。
