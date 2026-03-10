# 线性回归的简洁实现 (PyTorch)

本文记录了如何利用深度学习框架（PyTorch）的高级 API 简洁、高效地实现线性回归模型。以下将按照模型训练的标准流程，依次梳理代码实现与核心要点。

## 1. 生成数据集

在开始搭建模型前，我们先利用 `d2l`（Dive into Deep Learning）库生成带有一定噪声的模拟数据集。

```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## 2. 读取数据集

我们可以借助框架内置的数据处理 API 来高效读取数据。通过组合 `data.TensorDataset` 和 `data.DataLoader`，不仅能快速构建数据迭代器，还能方便地实现打乱数据操作以及指定批量大小（Batch Size）。

```python
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 验证数据迭代器是否正常运行（获取第一批小批量数据）
next(iter(data_iter))
```

## 3. 定义模型

面对复杂的神经网络，手动编写线性代数运算不仅繁琐且容易出错。为此，我们可以直接采用框架预定义好的神经网络层（Layer）。

在 PyTorch 中，全连接层由 `nn.Linear` 类定义。同时，我们可以使用 `nn.Sequential` 容器将多个层按顺序串联，快速构建出标准前向传播的网络结构。

```python
# nn (Neural Network) 是神经网络的缩写
from torch import nn

# 定义包含单层全连接层的网络：输入特征维度为 2，输出维度（标量）为 1
net = nn.Sequential(nn.Linear(2, 1))
```

## 4. 初始化模型参数

网络构建完成后，需要对其参数（权重和偏置）进行初始化。

深度学习框架通常提供了多种预定义的初始化方法。在 PyTorch 中，我们可以通过网络层实例直接访问参数，并调用带有 `_` 后缀的就地修改（In-place）函数来设定初始值：

```python
# net[0] 代表网络中的第一层（即我们定义的 nn.Linear）
# 将权重参数初始化为均值 0、标准差 0.01 的正态分布
net[0].weight.data.normal_(0, 0.01)

# 将偏置参数初始化为 0
net[0].bias.data.fill_(0)
```

## 5. 定义损失函数

在线性回归中，我们通常使用均方误差（Mean Squared Error）作为损失函数。在 PyTorch 的 `nn` 模块中，这一需求由 `MSELoss` 类满足，其默认会返回所有样本损失的平均值（计算平方 $L_2$ 范数）。

```python
loss = nn.MSELoss()
```

## 6. 定义优化算法

PyTorch 的 `optim` 模块内置了大量主流的神经网络优化算法。这里我们实例化一个小批量随机梯度下降（SGD）优化器，并传入需要优化的参数以及学习率（Learning Rate）超参数。

```python
# 将模型的所有参数（可通过 net.parameters() 获取）交给优化器管理，并设置学习率为 0.03
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

## 7. 训练模型

借助于高级 API 提供的高度封装设施，大部分繁琐的底层逻辑和计算被隐蔽，极大简化了我们的训练代码。每个迭代周期（Epoch）的核心步骤如下：

1. **前向传播**：将输入 `X` 喂给模型生成预测，并计算当前损失 `l`。
2. **梯度清零**：调用 `trainer.zero_grad()` 清空上一次的梯度积压。
3. **反向传播**：调用 `l.backward()` 根据损失计算网络各参数的梯度。
4. **参数更新**：调用 `trainer.step()` 根据梯度更新模型参数。

```python
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad() # 1. 梯度清零
        l.backward()        # 2. 反向传播，计算梯度
        trainer.step()      # 3. 更新参数
        
    # 每个 epoch 结束后，计算当前整个测试集上的损失
    l = loss(net(features), labels)
    print(f'Epoch {epoch + 1}, Loss: {l:f}')
```

## 8. 小结

* **高级 API 的优势**：利用 PyTorch 的高级封装，可以摒弃诸多手动重复性工作，从而更高效、规范地搭建并训练深度学习模型。
* **核心模块分工明确**：
    * `torch.utils.data`：负责数据读取与处理流水线构建（如 `TensorDataset`、`DataLoader`）。
    * `torch.nn`：提供了丰富的网络层组件（如 `Linear`、`Sequential`）和损失函数（如 `MSELoss`）。
    * `torch.optim`：涵盖了各类主流的优化算法（如 `SGD`、`Adam` 等）。
* **就地操作语义**：在 PyTorch 中，带有 `_` 结尾的方法通常表示就地修改（In-place Operation），常被用来高效地初始化或重写变量参数。
