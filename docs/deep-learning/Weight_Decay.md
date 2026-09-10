# 权重衰减：L2 正则化与 AdamW

权重衰减（weight decay）会在更新参数时把权重向 0 拉回一点。它常用于控制模型复杂度，但需要区分两个实现：把 L2 惩罚加进 loss，以及把衰减从梯度更新中解耦。SGD 下二者可以等价，Adam 等自适应优化器下则不是一回事。

## 1. 为什么限制权重

训练样本少、特征多时，模型可能用很大的正负权重精确配合，记住训练集里的噪声。加入参数范数偏好后，优化器会在拟合误差和较小权重之间取舍：

$$
J(w,b)=L(w,b)+\frac{\lambda}{2}\lVert w\rVert_2^2.
$$

(lambda) 越大，偏向小权重的力量越强。它不是“权重越小一定越好”；过大的正则会让模型欠拟合。

对线性模型，L2 惩罚让参数解更平滑、对共线特征更稳定。对深度网络，它是常用的正则手段之一，但实际效果还与归一化层、学习率、训练时长和数据增强有关。

## 2. SGD 中为什么叫“衰减”

L2 正则后的梯度为

$$
\nabla_w J=\nabla_w L+\lambda w.
$$

代入 SGD：

$$
w_{t+1}
=w_t-\eta(\nabla_wL+\lambda w_t)
=(1-\eta\lambda)w_t-\eta\nabla_wL.
$$

每步先把旧权重乘以 (1-\eta\lambda)，再减去任务梯度。从更新式看，权重确实按比例衰减。

这里的等价关系有条件：普通 SGD 使用统一学习率，且没有额外改变各坐标梯度尺度的机制。加入 momentum 后仍能定义相应实现，但不同框架的具体耦合方式要看优化器代码。

## 3. Adam 中 L2 与 weight decay 不再等价

Adam 会根据一阶、二阶矩对梯度的不同坐标自适应缩放。如果把 (lambda w) 混入任务梯度，它也会被 Adam 的矩估计缩放，于是各参数受到的收缩不再是简单的同一比例。

AdamW 将衰减从梯度中拆开：

$$
w_{t+1}=(1-\eta\lambda)w_t-\eta\,\operatorname{AdamUpdate}(\nabla_wL).
$$

因此：

- SGD 下，L2 penalty 与按比例 weight decay 可写成等价更新；
- Adam 下，`Adam(weight_decay=...)` 的耦合实现与 AdamW 的解耦衰减一般不同；
- 讨论实验时应写清优化器和实现，不能只说“用了 L2”。

## 4. PyTorch 写法

### 4.1 用优化器配置参数组

```python
from torch import nn
import torch

net = nn.Sequential(
    nn.Linear(200, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

decay_params = []
no_decay_params = []

for name, param in net.named_parameters():
    if param.ndim >= 2:
        decay_params.append(param)
    else:
        no_decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 1e-2},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=3e-4,
)
```

很多训练配置不对 bias 和 normalization 的 scale/shift 做衰减。这是常见经验，不是数学定律。按 `param.ndim` 分组很方便，但复杂模型最好按模块类型或经过审计的参数名规则处理，避免误分 embedding、特殊标量或自定义参数。

### 4.2 显式加 L2 penalty

```python
prediction = net(X)
data_loss = loss_fn(prediction, y)

l2 = sum(param.square().sum() for param in decay_params)
loss = data_loss + 0.5 * lambd * l2
```

这段代码表达的是“正则项属于训练目标”。它方便研究不同 penalty，但对 Adam 来说不等价于 AdamW。性能也不是选择 AdamW 的唯一理由，关键差别是更新规则。

## 5. 一个小样本、高维度实验

```python
n_train, n_test = 20, 100
num_inputs, batch_size = 200, 5
true_w = torch.ones((num_inputs, 1)) * 0.01
true_b = 0.05
```

这类设置容易出现训练误差很低、测试误差较高。可以固定随机种子和其他超参数，只改变 `weight_decay`，记录：

- train loss 与 validation loss；
- (\lVert w\rVert_2)；
- 多个随机种子下的均值和波动。

只跑一次并看到测试误差下降，还不足以判断某个衰减值普遍更好。

## 6. 怎样选 `weight_decay`

没有跨任务通用的固定值。较稳妥的做法是：

1. 先固定优化器、学习率计划、训练步数和数据增强；
2. 在对数尺度上搜索，例如 (0,10^{-5},10^{-4},10^{-3},10^{-2},10^{-1})；
3. 用验证集选择，并观察 train-validation gap；
4. 学习率改变后重新检查，因为每步实际收缩与 (eta\lambda) 有关；
5. 记录是否排除了 bias、norm 和 embedding。

权重衰减不是主要用来防止数值爆炸。若 loss 或梯度出现 `NaN`，应先检查学习率、混合精度、归一化、异常输入和梯度裁剪。

## 7. 记住这条边界

“L2 正则化等于权重衰减”只在特定更新规则下成立。看见 AdamW 时，我会把它理解成：任务梯度按 Adam 的方式更新，参数收缩单独做。这样就不容易把 loss penalty、优化器参数和实验结论混在一起。
