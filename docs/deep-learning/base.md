# PyTorch 中的模块、参数与设备

这篇记录 `nn.Module` 最容易混淆的几件事：模块怎样调用，参数如何注册，延后初始化适合什么情况，以及模型怎样保存和搬到设备上。

## 1. 自定义一个模块

```python
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        H = F.relu(self.hidden(X))
        return self.out(H)


net = MLP()
X = torch.randn(4, 20)
output = net(X)
print(output.shape)  # (4, 10)
```

一个模块通常要做三件事：

- 在 `__init__` 中声明子模块和参数；
- 在 `forward` 中描述数据流；
- 让 `nn.Module` 负责参数遍历、设备移动、模式切换、hooks 和序列化。

### 1.1 为什么调用 `net(X)`

`net(X)` 先进入 `nn.Module.__call__`，框架处理 hooks 等机制后再调用 `forward(X)`。正常代码应使用这一入口。

直接调用 `net.forward(X)` 会绕过 module hooks 和一些框架包装，但 `forward` 内部的普通 Tensor 运算仍可由 autograd 记录。它的问题不是“一定无法反向传播”，而是绕过了模块调用协议，可能让监控、编译或分布式逻辑失效。

## 2. 子模块与参数怎样注册

把 `nn.Module` 或 `nn.Parameter` 赋给模块属性时，PyTorch 会自动登记：

```python
for name, parameter in net.named_parameters():
    print(name, parameter.shape)
```

输出中的名字会类似：

```text
hidden.weight torch.Size([256, 20])
hidden.bias   torch.Size([256])
out.weight    torch.Size([10, 256])
out.bias      torch.Size([10])
```

`state_dict()` 保存参数和 persistent buffers：

```python
for name, tensor in net.state_dict().items():
    print(name, tensor.shape)
```

buffer 不是由优化器更新的参数，但需要跟着模型移动和保存，例如 BatchNorm 的 running statistics。可以用 `register_buffer` 注册。

## 3. 自定义参数时别用普通 Tensor 冒充

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X):
        return X @ self.weight.T + self.bias
```

如果 `self.weight` 只是一个普通 Tensor：

- 它不会出现在 `model.parameters()` 中；
- `optimizer = Adam(model.parameters())` 不会更新它；
- `model.to(device)` 和 `state_dict()` 也不会把它当参数管理。

普通 Tensor 可以通过 `requires_grad=True` 得到梯度，但这仍不等于已注册进模块。对于固定状态，用 buffer；对于可训练状态，用 Parameter。

参数集合较多时使用 `nn.ParameterList`、`nn.ParameterDict`、`nn.ModuleList` 或 `nn.ModuleDict`。把子模块随手放进普通 Python list，框架无法自动发现它们。

## 4. 延后初始化

输入维度暂时未知时，可以使用 lazy module：

```python
lazy_net = nn.Sequential(
    nn.LazyLinear(256),
    nn.ReLU(),
    nn.LazyLinear(10),
)

print(lazy_net(torch.randn(4, 20)).shape)
```

第一次 forward 时，`LazyLinear` 根据输入最后一维创建参数。它适合快速原型和形状由上游决定的模块，也带来一些额外约束：

- 首次 forward 前参数尚未完全 materialize；
- 初始化、设备、dtype、checkpoint 加载和分布式包装的顺序要更小心；
- 输入 shape 写错时，错误会推迟到首次运行；
- 做静态配置审计或分片规划时，明确维度通常更方便。

不能简单说大模型“禁止” lazy initialization，或它必然导致 OOM。是否使用取决于框架和初始化流程；规模化训练通常偏向显式 shape，是因为可预测性和分布式规划更重要。

## 5. 保存和加载 `state_dict`

```python
checkpoint_path = "mlp.pt"
torch.save(net.state_dict(), checkpoint_path)

clone = MLP()
state = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)
clone.load_state_dict(state)
clone.eval()
```

保存 `state_dict` 比直接 pickle 整个模型对象更容易跨代码重构。加载时仍需先创建匹配的模型结构。

`load_state_dict` 默认严格匹配键名。迁移学习或结构调整时可用 `strict=False`，但必须检查返回的 missing/unexpected keys，不能默默忽略。

训练断点通常还要保存 optimizer、scheduler、epoch、随机数状态和混合精度 scaler。发布只用于推理的权重时，则可只保留模型状态和配置。

## 6. 设备与 dtype

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
X = X.to(device)
output = net(X)
```

参与同一运算的参数和输入通常必须位于兼容设备上。常见报错是：

```text
RuntimeError: Expected all tensors to be on the same device
```

排查时打印：

```python
print(next(net.parameters()).device)
print(X.device)
print(X.dtype)
```

把 GPU Tensor 转为 NumPy 前，需要先脱离计算图并搬回 CPU：

```python
array = output.detach().cpu().numpy()
```

`.to(device)` 对 Tensor 通常返回新对象，所以写 `X.to(device)` 而不接返回值，`X` 仍留在原设备；`nn.Module.to` 会就地修改模块并返回自身，但仍建议写成 `net = net.to(device)`，意图更明确。

## 7. `train()` 与 `eval()`

```python
net.train()
# training loop

net.eval()
with torch.no_grad():
    # validation or inference
    pass
```

`train()` 与 `eval()` 影响 Dropout、BatchNorm 等模块，不决定参数是否需要梯度。`no_grad()` 或 `inference_mode()` 才控制 autograd 是否记录图。

## 8. 一次模块检查

写完自定义模块后，我通常按这个顺序检查：

1. 用很小的随机输入跑 forward，核对每层 shape；
2. 打印 `named_parameters()`，确认自定义参数都已注册；
3. 算一个标量 loss 并 backward，检查关键参数是否有有限梯度；
4. 调用 `to(device)`，确认参数、buffer 和输入在同一设备；
5. 保存再加载 `state_dict`，比较同一输入的输出；
6. 若含 Dropout/BatchNorm，分别测试 train 与 eval 行为。

这些检查比等到完整训练后再看 loss 更省时间。模块边界清楚、参数注册正确，后面的优化问题才有讨论基础。
