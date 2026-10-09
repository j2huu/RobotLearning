# 第 3 课补充：FP2 adaptor 与 twist–wrench 机制

本节回答两个问题：FP2 中的 adaptor 到底适配什么，以及 twist、wrench、interaction frame 和 hybrid controller 怎样构成一条可执行的控制链。

参考论文：

- [FP2: Equipping Robotic Foundation Models with Force Control](../../ICRA27_2601_MS.pdf)
- [Force Policy](../../2602.22088v2.pdf)

---

## 1. FP2 的 adaptor 不是 LoRA

这里最容易混淆三件事：

1. **RFM task adaptation**：第一阶段可以用 full fine-tuning 或 LoRA，让 π0、π0.5、GR00T 等 foundation policy 学会具体任务。
2. **context compressor**：把 RFM 内部的一长串 contextual tokens 变成一个固定维度的 task-context token。
3. **force-control sidecar**：读取这个 task context 和实时力觉，预测 hybrid controller 需要的控制参数。

通常把后两项连同 hybrid controller 合称为 FP2 的 downstream adaptor/interface。它不是往 backbone 某些线性层里插入低秩矩阵，而是在原 action policy 下游增加一条接触控制通道。

```text
language + images + state
          │
          ▼
task-adapted RFM, 2 Hz
          ├──────────────► action chunk ─► alignment/interpolation ─┐
          │                                                        │
          └► contextual tokens C_t                                 │
                         │                                         │
                         ▼                                         │
              context compressor ψ                                 │
                         │ cached z_t                               │
                         ▼                                         ▼
measured wrench/state ─► force policy π_F, 50 Hz ─► Σ, S, W_ref ─► hybrid controller, 1 kHz
                                                                        │
                                                                        ▼
                                                                  joint commands
```

这个接口适配的是两种不同抽象层级：

- RFM 给出“往哪里、怎样完成任务”的运动参考；
- force policy 给出“接触坐标系是什么、哪些轴控力、目标力是多少”；
- hybrid controller 把二者变成稳定的闭环执行。

---

## 2. Context compressor 具体做什么

RFM 的隐藏表示是一串 token：

$$
C_t\in\mathbb R^{S\times D},
$$

其中 $S$ 是 token 数，$D$ 是 backbone hidden dimension。不同 foundation model 的 $S,D$ 和 token 组织方式都可能不同，50 Hz 的小型 force policy 也不适合每次处理整串大模型 token。

FP2 引入一个 learned query $Q$，让它通过 cross-attention 从 $C_t$ 中提取与当前任务和阶段有关的信息：

$$
z_t=\psi(C_t)
=f_{enc}\bigl(Q,\operatorname{Proj}(C_t)\bigr)
\in\mathbb R^{1\times d}.
$$

可直观理解为：

```text
Q: “为了下游接触控制，我现在需要知道什么？”
K,V: RFM 的所有 contextual tokens
cross-attention output: 一个紧凑的任务上下文 z_t
```

训练时没有直接规定 $z_t$ 必须表示“擦拭阶段”或“插入阶段”。论文接一个 reconstruction decoder：

$$
\hat C_t=f_{dec}(z_t),
$$

并优化：

$$
\mathcal L_{comp}
=\|C_t-\hat C_t\|_2^2
+\lambda\,\mathcal L_{cos}(C_t,\hat C_t).
$$

MSE 约束数值重建，cosine loss 约束 token 表示的方向。训练完成后 decoder 被丢弃，encoder/compressor 冻结。部署时 $z_t$ 只在 RFM 更新时重新计算，并在两个 slow step 之间缓存。

它的作用是：

- 统一不同 RFM 与 force policy 的表示接口；
- 给 fast policy 提供 task identity 和 task phase；
- 避免 50 Hz 重跑大模型或处理全部 token。

它不负责读取实时力，也不直接输出 force command。真正从观测决定接触控制参数的是：

$$
(\Sigma_t,S_t,\hat W_t)
=\pi_F(z_t,W_{t-H:t},P_{t-H:t}).
$$

这里 $W_{t-H:t}$ 是**测得的 wrench 历史**，$hat W_t$ 是**期望 wrench**，两者不要混为一谈。

---

## 3. Twist 是什么

刚体末端的瞬时运动可以写成 6D twist：

$$
\xi=
\begin{bmatrix}
v\\
\omega
\end{bmatrix}
\in\mathfrak{se}(3),
$$

其中：

- $v=[v_x,v_y,v_z]^\top$：某个参考点的线速度，单位通常为 m/s；
- $\omega=[\omega_x,\omega_y,\omega_z]^\top$：角速度，单位通常为 rad/s。

twist 描述“末端此刻怎样动”，不是有限时间后的 pose。有限位姿属于 $SE(3)$，twist 属于它的切空间 $\mathfrak{se}(3)$。

twist 与参考点有关。若把参考点从 $o$ 移到 $p$，且 $r$ 是从 $o$ 指向 $p$ 的向量，则：

$$
v_p=v_o+\omega\times r,
\qquad
\omega_p=\omega_o.
$$

所以谈 twist 时必须说明表达坐标系和参考点。

---

## 4. Wrench 是什么

作用在末端上的广义力写成 6D wrench：

$$
W=
\begin{bmatrix}
f\\
m
\end{bmatrix}
\in\mathfrak{se}^*(3),
$$

其中：

- $f=[f_x,f_y,f_z]^\top$：力，单位 N；
- $m=[m_x,m_y,m_z]^\top$：关于指定参考点的力矩，单位 N·m。

wrench 同样与参考点有关。平移参考点后，力本身不变，但力矩会多出力臂项：

$$
m_p=m_o-r\times f
$$

具体正负号取决于 $r$ 的定义方向。因此 F/T sensor 数据必须先变换到与控制器一致的 TCP、interaction-frame origin 和坐标约定。

---

## 5. 为什么 twist 和 wrench 是对偶量

二者最核心的关系不是“都是六维向量”，而是它们的配对给出瞬时机械功率：

$$
\boxed{\mathcal P=W^\top\xi=f^\top v+m^\top\omega.}
$$

- $\mathcal P>0$：按当前符号约定，环境/执行器向该运动注入能量；
- $\mathcal P<0$：该作用阻碍运动并耗散能量；
- $\mathcal P=0$：理想约束力与允许运动不做功。

例子：末端沿桌面 $x$ 方向擦拭，表面法向为 $z$。理想情况下：

$$
\xi^*=[v_x,0,0,0,0,0]^\top,
\qquad
W^*=[0,0,f_z,0,0,0]^\top,
$$

因此：

$$
W^{*\top}\xi^*=f_zv_z=0.
$$

机器人可以沿切向运动，同时在法向施力。这就是 hybrid position/force control 能成立的几何基础。

### 坐标变换为何必须成对

若刚体变换为 $T$，twist 用 adjoint 变换：

$$
\xi'=\operatorname{Ad}_T\xi.
$$

为保持功率不变，wrench 必须用 inverse-transpose adjoint：

$$
W'=\operatorname{Ad}_T^{-\top}W.
$$

于是：

$$
W'^\top\xi'=W^\top\xi.
$$

这也是为什么不能只把 force 的三个分量乘一个旋转矩阵，却忽略 moment、参考点和平移带来的耦合。

还有一个面试细节：twist 中混合了 m/s 与 rad/s，wrench 中混合了 N 与 N·m。$W^\top\xi$ 作为对偶配对有明确物理单位，但不能在没有尺度或 metric 的情况下，把 twist 和 wrench 当作同一欧氏空间中的普通向量随意计算距离。

---

## 6. 从环境 stiffness 得到 interaction subspaces

在一个局部接触点附近，可以用线性化关系近似环境响应：

$$
W_{el}\approx-K_{env}\,\delta\chi,
$$

其中 $\delta\chi$ 是微小的 6D 位姿扰动，$K_{env}$ 是局部环境 stiffness。对称情形下可作特征分解：

$$
K_{env}=Q\Lambda Q^\top.
$$

- 大特征值方向：环境很硬，继续强行走位置轨迹容易产生大力；这些方向构成 constraint subspace $\mathcal U$。
- 小特征值方向：环境允许运动；这些方向构成 admissible-motion subspace $\mathcal T$。

理想目标满足：

$$
\xi^*\in\mathcal T,
\qquad
W^*\in\mathcal U,
\qquad
W^{*\top}\xi^*=0.
$$

也就是说，允许运动的方向使用 motion/position control，受约束方向使用 force control。这样不会在同一方向同时强制位置和力，因为两者通常会发生控制冲突。

Force Policy 并不要求部署时显式估计完整 $K_{env}$。它利用 demonstrations 中的 pose、twist 和 wrench 恢复 interaction-frame/control-mode targets，训练 policy 在线预测这些结构。stiffness eigenspace 是其物理解释。

---

## 7. Interaction frame 是怎样定义的

世界坐标系或 robot base frame 不会自动与接触几何对齐。擦拭一个曲面时，表面法向不断旋转；若始终在 base-$z$ 方向控力，机器人很快就会把力施向错误方向。

因此 policy 输出一个局部 interaction frame：

$$
\Sigma_t=(R_{\Sigma,t},p_{\Sigma,t}).
$$

Force Policy 的核心构造思想是：

- 让一个主轴与 intended wrench $W^*$ 的主要方向对齐，通常把它作为局部 $z$ 轴；
- 把 intended twist $\xi^*$ 投影到与该轴正交的平面，作为局部 $x$ 轴；
- 用右手定则得到 $y$ 轴；
- 在 wrench 很小、方向退化或含旋转自由度时，使用 twist 或参考轴消除歧义。

以曲面擦拭为例：

```text
Σ-z: 当前表面法向，force-controlled
Σ-x: 当前擦拭方向，position-controlled
Σ-y: 表面内的另一个切向，position-controlled
```

frame 的价值是把一个随物体姿态和接触几何变化的问题，转成局部坐标系中近似固定的控制模式。

---

## 8. Demonstration 中为何 twist 与 wrench 不完全正交

真实数据不会满足完美的 $W^\top\xi=0$。论文把观测量理解为：

$$
\xi=\xi^*+\xi_c,
\qquad
W=W^*+W_c.
$$

其中：

- $\xi^*,W^*$：任务意图中的允许运动与约束力；
- $W_c$：沿运动方向的摩擦、黏性等 dissipative residual；
- $\xi_c$：沿受力方向发生的柔顺变形或结构 residual。

展开功率：

$$
\mathcal P
=(W^*+W_c)^\top(\xi^*+\xi_c).
$$

理想的 $W^{*\top}\xi^*$ 为零，真实非零功率主要来自：

$$
W_c^\top\xi^*
\quad\text{和}\quad
W^{*\top}\xi_c.
$$

因此不能看到 $W^\top\xi\neq0$ 就断言 interaction frame 错了。它可能来自摩擦或结构柔顺。

论文使用的直觉是：

- 若结构 residual 较明显，先相信测得的 wrench 方向，再从 twist 中去掉沿 wrench 的分量；

$$
\xi^*\approx\xi-\operatorname{Proj}_{W}(\xi).
$$

- 若耗散 residual 较明显，先相信测得的 twist 方向，再从 wrench 中去掉沿 twist 的分量；

$$
W^*\approx W-\operatorname{Proj}_{\xi}(W).
$$

这些 projection 应理解为在经过坐标统一和尺度处理后的任务空间中恢复主方向，而不是把 N、N·m、m/s、rad/s 不加处理地直接做欧氏投影。

---

## 9. Selection mask 怎样把 motion 和 force 分开

FP2 的 force policy 输出：

$$
(\Sigma_t,S_t,\hat W_t).
$$

$S_t\in\{0,1\}^6$ 是 interaction frame 下的 selection mask：

- $S_i=1$：第 $i$ 个轴使用 force control；
- $S_i=0$：第 $i$ 个轴使用 motion/position control。

定义互补矩阵：

$$
S=\operatorname{diag}(s_1,\ldots,s_6),
\qquad
I-S.
$$

一个概念性的 hybrid command 是：

$$
u_\Sigma
=(I-S)u_{motion}+S u_{force}.
$$

其中：

- $u_{motion}$ 来自对齐和插值后的 RFM trajectory tracking error；
- $u_{force}$ 来自期望与实测 wrench 的误差：

$$
e_W=\hat W-W_{meas}.
$$

例如表面擦拭可用：

$$
S=\operatorname{diag}(0,0,1,0,0,0),
$$

即局部 $x,y$ 跟踪 RFM 的运动，局部 $z$ 闭环维持目标法向力。自由空间通常有 $S=0$，全部跟踪 RFM motion。

具体低层实现可能是 operational-space position/force controller、admittance realization 或 torque controller。若最终计算 task-space wrench command，常见的关节映射形式为：

$$
\tau=J(q)^\top W_{cmd}+\tau_{dyn}+\tau_{null},
$$

但这只是通用结构；不能在没有实现细节时声称 FP2 论文使用了某一个固定增益公式。FP2 的关键贡献是为低层控制器预测 $\Sigma,S,\hat W$，而不是提出一条新的关节力矩控制律。

---

## 10. 一次 50 Hz / 1 kHz 执行周期

假设当前 RFM action 正让机械臂沿盘子边缘擦拭：

1. 2 Hz RFM 读取 image、language、state，产生 action chunk 和 contextual tokens。
2. compressor 计算 $z_t$ 并缓存；RFM trajectory 经 DTW、quintic spline 后变为 50 Hz motion reference。
3. 每 20 ms，force policy 读取最新一段 wrench/proprio history 和 $z_t$。
4. 它预测当前 interaction frame、force-position mask 和 desired wrench。
5. 每 1 ms，hybrid controller 把 wrench 转到 interaction frame。
6. mask 为 0 的轴继续跟踪 RFM trajectory；mask 为 1 的轴根据 force error 闭环修正。
7. interaction frame 随曲面法向改变，法向控力方向也随之改变。

因此 fast policy 的作用不是每 20 ms 重写“擦盘子”的运动计划，而是不断更新“当前怎样接触盘子”。

---

## 11. 三个容易答错的面试点

### 11.1 “FP2 adaptor 输出 force 吗？”

更准确的回答是：force-control policy 输出 desired wrench、interaction frame 和 selection mask；实际闭环控制 command 由 1 kHz hybrid controller 根据 desired wrench 与 measured wrench 的误差产生。

### 11.2 “既然有 desired wrench，它是不是 impedance control？”

不能直接这样归类。Impedance control 指定 motion–force 的动态关系，如：

$$
F_{cmd}=K(x_d-x)+D(\dot x_d-\dot x).
$$

FP2 的 learned policy 主要选择 frame、mode 和 wrench reference，低层是 hybrid force-position regulation。某个具体实现可以含 impedance/admittance 环节，但这不改变 FP2 policy output 的定义。

### 11.3 “为什么不用 RFM action 自己处理接触？”

因为低频 action chunk 没有足够的带宽来处理瞬时接触，而且 foundation policy 可能没有 force input。FP2 保留 RFM 的语义和运动先验，把最新 wrench 留给小型 50 Hz policy，再由 1 kHz controller 实现闭环。

---

## 12. 一句话口述版本

> FP2 uses a frozen context compressor as a semantic adapter from arbitrary foundation-model tokens to a compact task representation, then combines that representation with high-rate wrench and proprioceptive histories to predict an interaction frame, a force–position selection mask, and a desired wrench. Twist and wrench are dual motion/force quantities whose pairing gives mechanical power; the predicted interaction frame aligns their motion and constraint subspaces so that a hybrid controller can track the foundation model's trajectory on free axes and regulate contact force on constrained axes.

