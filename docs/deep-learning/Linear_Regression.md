# 线性回归：用 PyTorch 走完一次训练

线性回归适合拿来熟悉深度学习框架，因为模型只有一层，数据、损失、梯度和参数更新却一个不少。下面用 (y=Xw+b+epsilon) 生成数据，再用 `nn.Linear` 把参数学回来。

## 1. 造一组可核对的数据

```python
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

print(features.shape)  # (1000, 2)
print(labels.shape)    # (1000, 1)
```

`features` 的每一行是一条样本，两个输入特征对应 (w_1,w_2)。`labels` 保留为 `(N, 1)`，和 `nn.Linear(2, 1)` 的输出形状一致。

## 2. 小批量读取

```python
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
    )


batch_size = 10
data_iter = load_array((features, labels), batch_size)
X, y = next(iter(data_iter))
print(X.shape, y.shape)  # (10, 2), (10, 1)
```

训练集通常设置 `shuffle=True`，避免每轮都用固定顺序组成 batch。验证和测试阶段不依赖随机打乱。

## 3. 模型与参数

```python
net = nn.Sequential(nn.Linear(2, 1))

nn.init.normal_(net[0].weight, mean=0.0, std=0.01)
nn.init.zeros_(net[0].bias)
```

`nn.Linear(2, 1)` 实现

$$
\hat y=XW^\top+b.
$$

PyTorch 的 `weight` 形状是 `(out_features, in_features)`，这里为 `(1, 2)`。公式中出现转置，正是因为框架用这个布局保存参数。

初始化通常放在 `torch.no_grad()` 语义下完成；`nn.init` 已经处理了这一点。相比直接改 `.data`，这种写法更清楚，也不容易绕开 autograd 后留下难查的问题。

## 4. 损失函数与优化器

```python
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
```

均方误差为

$$
L=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2.
$$

`nn.MSELoss()` 默认对 batch 内所有元素取平均。学习率 `0.03` 只是这组数据上的可用设置，不是线性回归的固定值。

## 5. 训练循环

```python
num_epochs = 3

for epoch in range(num_epochs):
    net.train()
    for X, y in data_iter:
        prediction = net(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        epoch_loss = loss_fn(net(features), labels)

    print(f"epoch {epoch + 1}: loss={epoch_loss.item():.6f}")
```

四个动作的顺序需要记住：

1. `net(X)` 做前向计算；
2. `optimizer.zero_grad()` 清掉上一次累计在 `.grad` 中的梯度；
3. `loss.backward()` 沿计算图求导；
4. `optimizer.step()` 用当前梯度更新参数。

PyTorch 默认累加梯度，所以忘记 `zero_grad()` 不等于“完全没有梯度”，而是把多个 batch 的梯度叠在一起。梯度累积训练会有意利用这个行为，普通训练循环则应显式清零。

## 6. 检查学到的参数

```python
learned_w = net[0].weight.detach().reshape(-1)
learned_b = net[0].bias.detach()

print("w error:", true_w - learned_w)
print("b error:", true_b - learned_b)
```

这一步比只看 loss 更直观：数据本来就是由 (w=[2,-3.4])、(b=4.2) 生成的，训练正常时参数应接近它们。不会完全相等，因为标签里加入了噪声，而且只训练了有限轮。

## 7. 这段代码里最容易忽略的细节

- **标签形状。** `(N,)` 和 `(N,1)` 混用时可能触发广播，代码能跑但损失算错。
- **训练/评估模式。** 当前模型没有 Dropout 或 BatchNorm，`train()` 与 `eval()` 的输出相同；保留切换能让训练循环以后安全扩展。
- **验证阶段仍建图。** 不使用 `no_grad()` 虽然通常也能得到结果，但会保存不需要的反向信息。
- **把 loss 降低当作唯一目标。** 还应检查参数误差、验证集误差和数据生成过程，避免“代码在收敛，任务却定义错了”。

线性回归把训练框架的最小闭环展示得很完整。后面的分类、MLP 和 Transformer 主要是在换模型、损失与数据，训练循环本身变化不大。
