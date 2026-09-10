# Softmax 回归：从 logits 到多分类概率

线性回归预测连续值，Softmax 回归处理互斥的多分类问题。模型先给每个类别一个未经归一化的分数（logit），softmax 再把这些分数转成总和为 1 的分布。

语言模型的 next-token head 也会对词表 logits 计算 softmax，但完整语言模型还包含大量上下文建模，不能把它简单等同于 Softmax 回归。

## 1. Softmax 在算什么

对类别分数 (mathbf{o}=[o_1,ldots,o_C])，第 (i) 类概率为

$$
p_i=\frac{e^{o_i}}{\sum_{j=1}^{C}e^{o_j}}.
$$

给所有 logits 同时加同一个常数，softmax 输出不变。因此工程实现会先减去最大 logit，降低指数溢出的风险：

$$
p_i=
\frac{e^{o_i-m}}{\sum_j e^{o_j-m}},
\qquad m=\max_j o_j.
$$

softmax 会保留类别之间的相对差异，但输出概率不一定经过良好校准。模型给出 0.9，并不自动表示它在真实数据上有 90% 的命中率。

## 2. 交叉熵

若真实类别为 (y)，单样本交叉熵是

$$
\ell=-\log p_y
=-o_y+\log\sum_j e^{o_j}.
$$

PyTorch 的 `nn.CrossEntropyLoss` 直接接收 logits，并在内部组合 `log_softmax` 与负对数似然。不要先对模型输出调用 softmax 再传给它，否则既重复计算，也会让数值稳定性和梯度都变差。

## 3. Fashion-MNIST 示例

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10),
)


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)


net.apply(init_weights)
```

输入形状从 `(B, 1, 28, 28)` 经 `Flatten` 变成 `(B, 784)`，线性层输出 `(B, 10)`。这 10 个数是 logits，不要求落在 0 到 1 之间。

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    net.train()
    for X, y in train_iter:
        logits = net(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

标签 `y` 应是形状 `(B,)` 的整数类别索引，dtype 通常为 `torch.long`。若使用 one-hot 或软标签，需要重新核对损失函数版本和张量格式。

## 4. 手算一遍正确类别的损失

```python
logits = torch.tensor([
    [1.0, 2.0, 0.5],
    [0.2, -0.3, 1.4],
])
labels = torch.tensor([1, 2])

log_probs = torch.log_softmax(logits, dim=1)
row_ids = torch.arange(logits.shape[0])
loss_per_sample = -log_probs[row_ids, labels]

print(loss_per_sample)
```

高级索引 `log_probs[row_ids, labels]` 取出每条样本在真实类别上的 log probability。它比 Python 循环更直接，也能保留批量计算。

等价的框架写法是：

```python
loss_per_sample_2 = nn.functional.cross_entropy(
    logits,
    labels,
    reduction="none",
)
assert torch.allclose(loss_per_sample, loss_per_sample_2)
```

## 5. 从 logits 得到预测

```python
predicted_class = logits.argmax(dim=1)
probabilities = logits.softmax(dim=1)
```

只要类别排序即可时，直接对 logits 做 `argmax`，结果与先 softmax 再 `argmax` 相同。只有展示概率、采样或计算需要概率的指标时，才必须显式计算 softmax。

## 6. 容易出错的地方

- **softmax 维度。** 分类通常沿最后的类别维计算，batch 维不能参与归一化。
- **把概率传给 `CrossEntropyLoss`。** 该函数需要 logits。
- **标签越界或 dtype 不对。** (C) 类问题的整数标签应在 ([0,C-1])。
- **只看 accuracy。** 类别不平衡时还应看 per-class recall、confusion matrix 等指标。
- **把 softmax 当作置信度证明。** 分布可能过度自信；温度缩放等方法处理的是校准问题，不是分类能力本身。

Softmax 回归没有隐藏层，决策边界仍是线性的。下一步加入 MLP，是为了让模型先学习非线性表示，再在表示空间里分类。
