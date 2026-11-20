# B4：Jacobian——关节怎么动，末端就怎么动

[目录](README.md) · [上一课：IK](03_inverse_kinematics.md) · [下一课：动力学](05_dynamics_basics.md)

## 直觉与一个公式

FK 是位置关系 $p=f(q)$；对时间求导，得到局部速度关系：

$$
\dot p=J(q)\dot q,\qquad J(q)=\frac{\partial p}{\partial q}.
$$

本课 $p=[x,y]^T$、$q=[q_1,q_2]^T$，所以 J 是 $2\times2$。J 的第 i 列回答：“只动第 i 个关节时，末端瞬时沿什么方向动？”

从 B2 的 FK 求偏导：

$$
J=\begin{bmatrix}
-l_1\sin q_1-l_2\sin(q_1+q_2)&-l_2\sin(q_1+q_2)\\
l_1\cos q_1+l_2\cos(q_1+q_2)&l_2\cos(q_1+q_2)
\end{bmatrix}.
$$

关节速度为 rad/s，末端线速度为 m/s；J 的列带有杆长尺度。三维完整末端速度常用 6 维 twist，对应 $6\times n$ Jacobian；还要说明 frame、参考点以及线/角速度排列顺序。本课只处理平面末端点的线速度。

## 手算：末端朝哪里动

取 $l_1=l_2=1$、$q=(0,\pi/2)$：

$$
J=\begin{bmatrix}-1&-1\\1&0\end{bmatrix},\qquad
\dot q=\begin{bmatrix}0.1\\0.2\end{bmatrix}
\Rightarrow \dot p=\begin{bmatrix}-0.3\\0.1\end{bmatrix}\,\mathrm{m/s}.
$$

相反，给定期望末端速度，可以用逆或伪逆求关节速度；B3 的数值 IK 正是用它把小位置误差转成关节修正。

## 奇异点：某些瞬时方向做不出来

两杆完全伸直 $q=(0,0)$ 时：

$$
J=\begin{bmatrix}0&0\\2&1\end{bmatrix},\qquad
\det J=l_1l_2\sin q_2=0.
$$

此时任意有限关节速度产生的一阶 x 速度都是 0，末端瞬时只能沿 y 方向运动。不是“机械臂完全不能动”，也不表示有限时间折叠后不能改变 x。

奇异是相对于所选任务 Jacobian 的秩下降。接近奇异点时，为实现某些末端速度，伪逆可能要求很大的关节速度。阻尼最小二乘可缓和这种放大：

$$
\dot q=J^T(JJ^T+\lambda^2 I)^{-1}\dot p_d,\qquad \lambda>0.
$$

它用跟踪误差换取较平缓的关节解，不能让不可达的方向凭空变成可达。

## 自测

1. J 的第 2 列是什么意思？**当前构型下，只给第 2 关节单位速度时产生的末端速度。**
2. 伸直时要求 $\dot p=(0.1,0)$，能精确做到吗？**在该构型的一阶速度模型中不能。**

参考：[Modern Robotics 第 5 章](https://modernrobotics.northwestern.edu/chapters/chapter5/)。
