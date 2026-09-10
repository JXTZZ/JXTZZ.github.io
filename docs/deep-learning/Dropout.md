# Dropout：训练时随机丢弃，推理时保持完整

Dropout 在训练阶段随机把一部分激活置为 0，减少网络长期依赖某组固定特征的机会。它是一种随机正则化方法，不保证每个任务都有效，也不能替代数据增强、权重衰减或更合理的模型容量。

## 1. Inverted dropout

设丢弃概率为 (p)，保留概率为 (1-p)。对激活 (h) 采样

$$
m\sim\operatorname{Bernoulli}(1-p),
$$

训练时输出

$$
h'=\frac{m}{1-p}h.
$$

于是

$$
\mathbb{E}[h']
=\frac{\mathbb{E}[m]}{1-p}h
=h.
$$

训练时已经把保留下来的激活除以 (1-p)，推理时便不需要再缩放。这种实现通常称为 inverted dropout，也是 PyTorch 的做法。

期望保持不变不代表某次 forward 的总和不变。每次 mask 都不同，单次输出会波动；只有对随机 mask 取期望时才得到原激活。

## 2. 手写一遍

```python
import torch


def dropout_layer(X: torch.Tensor, p: float, training: bool = True):
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if not training or p == 0.0:
        return X
    if p == 1.0:
        return torch.zeros_like(X)

    mask = (torch.rand_like(X) > p).to(X.dtype)
    return mask * X / (1.0 - p)
```

`torch.rand_like(X)` 会跟随输入的 device，避免模型在 GPU 上、mask 却建在 CPU 上。真实项目直接使用 `nn.Dropout`，手写版本主要用于理解公式。

## 3. PyTorch 中的训练与评估模式

```python
from torch import nn

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 10),
)
```

```python
net.train()
for X, y in train_dataloader:
    logits = net(X)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

net.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        logits = net(X)
```

- `net.train()` 让 Dropout 开始采样 mask；
- `net.eval()` 让 Dropout 变成恒等映射；
- `torch.no_grad()` 关闭梯度记录，和 Dropout 的模式切换是两件事。

若忘记 `eval()`，同一输入可能得到不同结果，评测指标也会带随机波动。影响大小由模型、(p) 和任务决定，不必夸大成“输出一定完全错误”。

## 4. Dropout 放在哪里

没有一个适用于所有网络的固定位置。

- 经典 MLP 常把 Dropout 放在激活之后；
- CNN 可以放在 block 或 classifier 中，也常使用通道级的 `Dropout2d`；
- Transformer 会在 attention 权重、attention 输出、FFN 输出或 residual 分支上使用不同 dropout；
- 有 BatchNorm 的网络需要通过实验判断两种随机机制是否互相影响；
- 小数据微调与大规模预训练所需的 dropout 可能不同，部分大模型配置甚至使用 (p=0)。

所以“永远紧跟 ReLU”不准确。阅读实现时应看它随机屏蔽的是激活、通道、attention 权重还是整条 residual 分支。

## 5. Dropout 为什么可能有效

直观上，每个 batch 都在训练一个略有不同的子网络，参数必须在许多 mask 下共同工作。这会削弱特征之间脆弱的共适应。它也可从噪声注入或近似模型集成的角度理解。

这些解释帮助建立直觉，但不能直接推出某个 (p) 的最佳值。Dropout 太大时，有效容量下降、优化噪声增加，训练集都可能拟合不好。

## 6. 调试与实验

### 6.1 检查模式是否切换

```python
x = torch.ones(4, 8)
layer = nn.Dropout(0.5)

layer.train()
y1 = layer(x)
y2 = layer(x)

layer.eval()
y3 = layer(x)

assert not torch.equal(y1, y2)
assert torch.equal(y3, x)
```

随机事件有极小概率让 `y1` 与 `y2` 相同，正式单元测试最好固定随机种子或检查统计性质。

### 6.2 选择概率

从 (p=0) 建立 baseline，再比较少量候选值。MLP classifier 中 0.5 很常见，但不是默认答案；卷积特征和大模型 residual 上往往使用更小值。记录 train/validation gap，比只看最后一轮 accuracy 更有用。

### 6.3 不要混淆不同随机层

- Dropout 随机置零激活；
- DropPath/Stochastic Depth 随机跳过整条 residual 分支；
- 数据增强修改输入；
- label smoothing 修改监督目标。

它们都能产生正则效果，但作用位置和训练行为不同。

## 7. 我目前的记法

训练时采样 mask，并用 (1/(1-p)) 保持激活期望；推理时原样通过。看到 Dropout 时再问三个问题：屏蔽的是什么，(p) 多大，评估前是否调用了 `eval()`。
