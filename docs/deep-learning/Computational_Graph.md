# 计算图与反向传播：PyTorch 到底保存了什么

`loss.backward()` 看起来只有一行，背后依赖的是前向计算时动态建立的计算图。理解这张图，主要为了回答三个问题：梯度怎样传回参数，为什么训练比推理占更多显存，以及哪些操作会意外截断梯度。

## 1. 前向计算会建立依赖关系

```python
def forward(x, y, W1, W2):
    z = x @ W1
    h = torch.relu(z)
    logits = h @ W2
    return torch.nn.functional.cross_entropy(logits, y)
```

只要参与运算的张量需要梯度，PyTorch autograd 就会记录产生结果的操作。结果张量通过 `grad_fn` 指向相应的反向函数，这些关系组成有向无环图（DAG）。

计算图的节点更适合理解为“张量与产生它的算子之间的依赖”。并不是所有 Python 局部变量都会原样复制进图里。每个算子的 backward 只保存求导所需的张量或元数据；ReLU、矩阵乘法等操作需要保存的内容也不完全相同。

```python
x = torch.randn(4, 8)
W = torch.randn(8, 3, requires_grad=True)
y = x @ W

print(y.requires_grad)  # True
print(y.grad_fn)        # 矩阵乘法对应的 backward 节点
```

## 2. 叶子张量与梯度

用户创建、且 `requires_grad=True` 的参数通常是叶子张量（leaf tensor）。调用 backward 后，叶子张量的梯度累加到 `.grad`：

```python
W = torch.tensor(2.0, requires_grad=True)
loss = W ** 2
loss.backward()
print(W.grad)  # tensor(4.)
```

中间结果默认不会保留 `.grad`，因为反向传播只需把梯度继续传下去。调试时若确实要查看中间张量梯度，可调用 `retain_grad()`。

PyTorch 使用“累加”而不是“覆盖”：

```python
loss = W ** 2
loss.backward()
print(W.grad)  # 在旧梯度上继续累加
```

训练循环因此要在适当位置调用 `optimizer.zero_grad()`。有意做 gradient accumulation 时，则每若干个 micro-batch 再清零和更新。

## 3. 链式法则怎样走过图

考虑两层网络：

$$
Z=XW_1,qquad H=\operatorname{ReLU}(Z),qquad O=HW_2.
$$

若损失为 (J)，则

$$
\frac{\partial J}{\partial W_2}
=H^\top\frac{\partial J}{\partial O}.
$$

求 (W_2) 的梯度需要前向得到的 (H)。继续传到 (W_1) 时，还需要 ReLU 的激活区域和 (X)。这就是训练阶段必须保留一部分 activation 的原因。

反向传播按图的逆拓扑顺序应用局部导数。一个中间张量若流向多个分支，来自各分支的梯度会相加，再继续传向它的父节点。

## 4. 图何时释放

默认情况下，一次 `backward()` 完成后，autograd 会释放为本次反向保存的中间结果。对同一张图第二次 backward 会报错：

```python
loss.backward()
# loss.backward()  # RuntimeError: Trying to backward through the graph a second time
```

确实需要重复使用同一张图时可以传 `retain_graph=True`，但它会延长 activation 的生命周期，容易增加显存。多数训练代码不需要这个参数；如果频繁依赖它，应先检查图是否被不必要地复用。

带历史的张量被 Python 容器长期引用，也可能让整条图无法及时释放。例如把每个 batch 的 `loss` 直接追加到列表中，会保留其图；记录数值时应使用 `loss.item()` 或 `loss.detach()`。

## 5. 训练显存不只有 activation

训练显存通常包含：

- 模型参数；
- 参数梯度；
- 优化器状态，例如 Adam 的一阶、二阶矩；
- 为 backward 保存的 activation；
- 临时算子工作区、通信缓冲与内存分配器缓存。

推理不保存反向所需的 activation，也没有梯度和优化器状态，所以通常省很多显存。但推理显存不只等于参数大小：batch、中间张量、长上下文的 KV cache 和算子工作区都可能占用大量空间。训练是推理的“固定 3～4 倍”也不是通用规律，比例会随 dtype、优化器、序列长度、batch 和并行方式变化。

## 6. `eval()`、`no_grad()` 与 `inference_mode()`

这三者处理的是不同问题：

```python
model.eval()  # 改变 Dropout、BatchNorm 等模块的行为

with torch.no_grad():
    output = model(x)  # 不记录反向图
```

- `model.eval()` 不会自动关闭 autograd；
- `torch.no_grad()` 停止记录梯度，但不改变 Dropout/BatchNorm 模式；
- `torch.inference_mode()` 比 `no_grad()` 约束更强，适合纯推理路径。

验证阶段通常同时使用 `eval()` 和 `no_grad()`。

## 7. 直接调用 `forward()` 会怎样

`model(x)` 会进入 `nn.Module.__call__`，再调用用户实现的 `forward()`。这条路径还负责 forward hooks、pre-hooks 等模块机制。因此正常代码应写 `model(x)`。

直接写 `model.forward(x)` 会绕过这些 module hooks，但其中的张量运算仍由 autograd 记录，不能说它一定导致 `backward()` 失效。问题在于它破坏了 `nn.Module` 的调用约定，可能跳过调试、监控、编译或分布式包装所依赖的逻辑。

## 8. 常见的梯度截断

### 8.1 `detach()` 与 `item()`

```python
y = model(x)
z = y.detach()  # z 与 y 共享数据，但不再沿 y 的历史求梯度
value = y.mean().item()  # 转为 Python 数值，用于日志
```

`detach()` 适合明确切断梯度的边界；误用会让上游参数收不到梯度。

### 8.2 不可导或离散操作

`argmax`、整数索引选择和大多数离散采样无法提供普通意义上的梯度。训练时常用连续松弛、代理梯度或重新设计目标，而不是期待 autograd 自动处理。

### 8.3 原地修改

某些 backward 需要前向张量的原值。若该值被原地操作改写，PyTorch 可能报 version counter 错误。带下划线的方法不一定都危险，但对参与求导的中间张量做原地修改前要确认 backward 是否依赖旧值。

## 9. 用计算换显存：gradient checkpointing

梯度检查点只保存部分节点。反向传播走到缺失区间时，重新执行那段前向计算，再得到所需 activation。它降低 activation 显存，代价是额外计算时间。

这项技术不会减少参数、梯度和优化器状态，也不会自动解决所有 OOM。定位显存问题时应先判断哪一部分占主要比例，再决定用 checkpoint、混合精度、减小 batch、缩短序列或参数分片。

## 10. 我用来排查 autograd 的顺序

1. 看参数是否在 `model.parameters()` 中，且 `requires_grad=True`；
2. 看 loss 是否有 `grad_fn`；
3. 查中途是否出现 `detach()`、`item()`、NumPy 转换或离散操作；
4. backward 后检查关键参数的 `.grad is None`、梯度范数和有限性；
5. 若显存持续增长，检查列表、缓存或日志对象是否保存了带历史的张量。

计算图没有把每个前向变量“永久锁住”。它只为当前反向保留必要信息，但这些信息可能很大，也可能因为引用或 `retain_graph=True` 活得比预期更久。
