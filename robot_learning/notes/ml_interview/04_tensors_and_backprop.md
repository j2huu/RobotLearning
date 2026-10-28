# M4：Tensor、计算图与反向传播

[目录](README.md) · [上一课](03_classical_models.md) · [下一课](05_optimization_and_training.md) · [Colab ML03](../../labs/notebooks/ML03_autograd_optimization.ipynb)

目标：不只会调用 `backward()`，还能从 loss 追到每个参数的梯度、shape 与数值稳定性。

## 1. 先定义轴，再写运算

| tensor | shape | 轴含义 |
|---|---|---|
| observations | [B,T,d] | batch、历史时间、feature |
| linear weight | [d,h] | 输入 feature、输出 feature；数学约定 |
| hidden | [B,T,h] | 每个历史时刻的 hidden vector |
| action target | [B,H,dₐ] | batch、未来动作、动作分量 |
| action valid mask | [B,H] | 哪些未来动作真实存在 |

PyTorch `nn.Linear(d,h).weight` 存储为 [h,d]，计算等价于 x @ weight.T + bias。数学公式中的 W 与框架存储方向必须区分。

`reshape` 改形状，`transpose/permute` 改轴顺序；二者一般不等价。`squeeze()` 可能意外删除 batch=1 的轴，优先明确 `squeeze(dim)`。

## 2. Chain rule 的一个完整例子

$$
z=XW+b,\quad h=\operatorname{ReLU}(z),\quad \hat y=hV+c.
$$

如果 $L=\frac1{2B}\sum_i(\hat y_i-y_i)^2$，则：

$$
\frac{\partial L}{\partial V}=h^\top\frac{\partial L}{\partial\hat y},
\quad
\frac{\partial L}{\partial z}
=\left(\frac{\partial L}{\partial\hat y}V^\top\right)\odot\mathbf1[z>0].
$$

反向传播反复应用局部导数和 chain rule。Autograd 通常执行 vector-Jacobian products，不需要构造每层完整 Jacobian。

## 3. Softmax 与 CE 的梯度

$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j),
\quad
\frac{\partial L_{CE}}{\partial z}=p-y_{onehot}.
$$

若 batch mean，则再除以 B。Notebook 比较手推梯度、autograd 与 double precision 的有限差分。有限差分步长太大有截断误差，太小有浮点消减误差；不能追求步长无限接近零。

## 4. 训练循环的每个动作

```python
model.train()
optimizer.zero_grad(set_to_none=True)
prediction = model(x)
loss = criterion(prediction, target)
loss.backward()
optimizer.step()
```

`backward()` 累积 `.grad`；不清空就会跨 step 累积。`step()` 根据当前梯度更新参数；仅 `backward()` 不会替你完成优化。梯度累积可以是有意设计，但 loss 缩放与清空时刻必须正确。

`model.eval()` 改变 Dropout/BatchNorm 的行为，不等于关闭 autograd；`torch.no_grad()` 禁用该段计算的梯度记录，不等于切到 eval。

`detach()` 切断图；`requires_grad=False` 冻结参数；是否包含在 optimizer 中又是另一层设置。冻结 RFM 不意味着下游分支没有梯度。

## 5. Masked loss 的分母

如果 error=[B,H,dₐ]，mask=[B,H]，对每个有效 scalar 平均：

$$
L=\frac{\sum_{b,h,j}m_{bh}e_{bhj}^2}{d_a\sum_{b,h}m_{bh}}.
$$

分母不是固定 B×H×dₐ；否则 padding 多的 batch 改变 loss 尺度。没有有效元素时要明确跳过或报错，不能默默除零。

## 6. 面试现场 debug 顺序

1. 检查 shape/dtype/device、目标值范围、是否错误 broadcast。
2. 用极小 batch 看 forward/loss 是否有限。
3. 确认参数 `.grad` 是否存在、是否 finite、有无异常范数。
4. 观察 `step()` 前后参数是否改变。
5. 尝试 overfit 一个小 batch，再检查数据、正则和 validation protocol。

## 验收

手写 linear backward 与 stable softmax；解释 `eval`/`no_grad`/`detach`；指出 [B,1] 与 [B] 的错误广播；从 masked loss 写出分母。

参考：[PyTorch Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。
