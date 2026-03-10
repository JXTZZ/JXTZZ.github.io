# Softmax 回归的简洁实现 (PyTorch)

在学习了线性回归预测连续数值之后，我们迎来了深度学习中最重要的概念之一——分类问题。**大模型（如 ChatGPT）本质上就是一个超级巨大版的 Softmax 回归模型**。它们的文本生成，其实就是“多分类问题”：根据上文，在几十万个词表中预测下一个词的概率。

本文将演示如何使用 PyTorch 高级 API 简洁地实现 Softmax 回归，并揭示其工程实践中的底层陷阱。

## 1. Softmax 与交叉熵核心原理解析

### 1.1 Softmax：概率转换器
线性回归输出的是没有范围限制的实数（术语叫 *Logits*）。而在分类问题中，我们需要输出概率（严格分布在 0 到 1 之间，且总和为 1）。
Softmax 函数通过以下三步实现“标准化”：
1. **指数化 ($e^x$)**：将所有分数变成正数，并拉大差距。
2. **求和**：计算所有指数值的总和作为分母。
3. **算百分比**：将每个指数值除以总和。

### 1.2 交叉熵损失 (Cross-Entropy)：衡量“打脸程度”
分类问题中，我们不看均方差，只看模型对“正确答案”给出的概率有多高。
交叉熵的本质是计算：$-log(正确选项的预测概率)$。
* 预测概率越高（如 0.9），$-log(0.9) \approx 0.1$，损失很小。
* 预测概率越低（如 0.0001），$-log(0.0001) \approx 9.2$，损失爆炸（必须重罚！）。

## 2. 代码实现

我们继续使用 `d2l` 库加载 Fashion-MNIST 图像分类数据集。

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

### 2.1 定义模型并初始化参数

Softmax 回归的输出层也是一个全连接层。因为我们处理的是 28x28 的图像数据，首先需要用 `nn.Flatten()` 将其展平为 784 维的一维向量。由于有 10 个类别，输出特征维度为 10。

```python
# PyTorch不会隐式地调整输入的形状
# nn.Flatten() 将输入的二维图像展平为一维向量
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# 应用初始化函数
net.apply(init_weights)
```

### 2.2 定义损失函数（🚨 工程排坑：LogSumExp 技巧）

**面试极爱考！为什么不先算 `Softmax`，再算 `交叉熵`？**

因为 **数值溢出 (Overflow)**！如果某个 Logit 这个值达到 100，经过 $e^{100}$ 计算后，计算机的 float32 会直接爆表变成 `inf` (无穷大)。接着算百分比的时候就会出现 `inf / inf = NaN`。一旦出现 `NaN`，训练瞬间崩溃。

因此，**绝对不要自己分开写 Softmax 和对数计算**。PyTorch 的 `nn.CrossEntropyLoss()` 把这两个步骤在底层融合成了一个算子，它内置了 **LogSumExp 技巧**（自动减去矩阵中的最大值来保证数值稳定性），完美避开了溢出问题。

总结：永远直接把**没有经过处理的原生态打分 (Logits)** 扔给 `nn.CrossEntropyLoss()`。

```python
# 'none' 表示返回每个样本的单独损失，而不是计算平均值
loss = nn.CrossEntropyLoss(reduction='none')
```

### 2.3 定义优化算法与训练

使用与线性回归相同的随机梯度下降（SGD）优化器，体现了深度学习优化算法的普适性。

```python
# 小批量随机梯度下降，学习率设置为 0.1
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型 10 个 Epoch
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 3. 张量切片实战补充 (消灭 for 循环)
在底层自研或调试损失函数时，常常需要提取真实标签对应的概率值。如果不使用高级 API，而是操作张量，不要用 `for` 循环！使用切片是非常优雅且高效的做法：

```python
# 假设 y_hat 是模型对 2 个样本在 3 个类别上的预测概率矩阵
y_hat = torch.tensor([[0.1, 0.3, 0.6],
                      [0.3, 0.2, 0.5]])

# y 是真实标签（第0个样本是第0类，第1个样本是第2类）
y = torch.tensor([0, 2])

# 高级索引提取：生成行号范围 + 列号标签
# 相当于提取 y_hat[0, 0] 和 y_hat[1, 2]
correct_probs = y_hat[range(len(y_hat)), y] 
print(correct_probs) # 输出 tensor([0.1000, 0.5000])

# 计算交叉熵
cross_entropy_loss = -torch.log(correct_probs)
```

## 4. 小结
1. **大模型的基石**：文本生成就是一个巨大的 Softmax 多分类器任务（Next-token prediction）。
2. **严防计算陷阱**：在工程实践中，必须直接传递未经 Softmax 处理的 Logits 给 `nn.CrossEntropyLoss` 以保证数值稳定性。
3. **高级特征**：运用张量的高级切片能力来替代低效的循环操作。