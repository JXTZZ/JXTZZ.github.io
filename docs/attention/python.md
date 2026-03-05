# Python 开发笔记

## 1. Tensor 运算中的原地操作与内存优化

在 Python 中进行深度学习开发时（如使用 PyTorch），虽然内存管理通常是自动的，但在处理大规模张量（Tensor）时，不当的操作会导致不必要的内存分配，进而影响性能甚至导致内存溢出（OOM）。

通常，简单的加法运算 `Y = Y + X` 会分配新的内存来存储结果，并让 `Y` 指向新的内存地址。如果我们希望重用 `Y` (或另一个张量 `Z`) 原有的内存空间（即原地操作，In-place Operation），可以使用索引切片 `[:]` 或者原地运算符（如 `+=`）。

### 示例代码

```python
import torch

X = torch.ones(5)
Y = torch.zeros(5)

# 1. 普通加法：分配新内存
before = id(Y)
Y = Y + X
print(f"Address changed: {id(Y) != before}")  # True，Y 指向了新的内存地址

# 2. 使用切片赋值：重用原内存
Z = torch.zeros_like(Y)
before = id(Z)
Z[:] = X + Y
print(f"Address retained (slice): {id(Z) == before}")  # True，Z 仍在原地址

# 3. 使用原地运算符 +=：重用原内存
before = id(Z)
Z += X
print(f"Address retained (+=): {id(Z) == before}")  # True，更加简洁
```

**总结：**
- 使用 `Z[:] = ...` 可以将结果写入预分配的内存中。
- 使用 `+=`, `*=` 等运算符也可以实现原地修改，避免创建新的临时对象。
- `id()` 函数可以用来验证对象在内存中的地址是否改变。
