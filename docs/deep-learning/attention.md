# Attention：从加权平均到多头自注意力

注意力机制做的事很朴素：给定一个查询（query），在一组键值对（key-value pairs）里计算相关性，再按相关性对 value 做加权求和。Transformer 把这套操作放到序列内部，于是每个 token 都能按当前需要读取其他 token 的信息。

这篇笔记先把单个 query 讲清楚，再走到 scaled dot-product attention、multi-head attention 和完整的 Transformer block。公式默认使用 batch-first 的张量布局。

## 1. 从注意力池化开始

已有键值对

$$
(\mathbf{k}_1, \mathbf{v}_1),\ldots,(\mathbf{k}_n, \mathbf{v}_n),
$$

以及查询 $\mathbf{q}$。先用打分函数 $s$ 衡量 query 与每个 key 的匹配程度，再经 softmax 得到权重：

$$
\alpha_i
=
\frac{\exp(s(\mathbf{q},\mathbf{k}_i))}
{\sum_{j=1}^{n}\exp(s(\mathbf{q},\mathbf{k}_j))}.
$$

最终输出是 value 的加权和：

$$
\operatorname{Attention}(\mathbf{q},K,V)
=
\sum_{i=1}^{n}\alpha_i\mathbf{v}_i.
$$

这里有三个容易混淆的角色：

- query 描述“当前要找什么”；
- key 用于和 query 匹配，决定每个位置分到多少权重；
- value 是真正被汇总的内容。

key 和 value 往往来自同一个输入位置，但用途不同。模型可以学习到：用一组特征判断“该不该看”，再从另一组特征里读取内容。

### 1.1 与核回归的关系

Nadaraya-Watson 核回归也可以写成加权平均：

$$
\hat y(x)=
\frac{\sum_i K(x-x_i)y_i}{\sum_j K(x-x_j)}.
$$

若使用高斯核，权重可改写为对距离分数做 softmax。它和现代 attention 共享“根据相似度加权汇总”的骨架。区别在于，Transformer 中的 query、key、value 通常由可学习的线性投影得到，打分函数也随训练一起调整。

## 2. Scaled dot-product attention

一次处理多个 query 时，把它们堆成矩阵：

- $Q\in\mathbb{R}^{L_q\times d_k}$；
- $K\in\mathbb{R}^{L_k\times d_k}$；
- $V\in\mathbb{R}^{L_k\times d_v}$。

标准公式是：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

各张量的形状沿着计算过程变化如下：

|步骤|形状|含义|
|---|---|---|
|$QK^\top$|$L_q\times L_k$|每个 query 对每个 key 的分数|
|softmax 后|$L_q\times L_k$|每行是一组归一化权重|
|乘 $V$|$L_q\times d_v$|每个 query 汇总后的表示|

softmax 沿 key 这一维计算。换句话说，每个 query 都有自己的一行注意力分布。

### 2.1 为什么使用点积

点积可以批量写成矩阵乘法，GPU 很擅长处理。若向量经过归一化，点积与余弦相似度直接相关；未归一化时，它同时受方向和模长影响。这里更准确的说法是：点积是模型可学习投影空间中的匹配分数，不必把它机械理解成语义相似度。

### 2.2 为什么除以 $\sqrt{d_k}$

假设 query 和 key 的各维独立、均值为 0、方差为 1，则它们的点积是 $d_k$ 项乘积之和，方差约为 $d_k$。维度增大后，分数的绝对值也容易变大，softmax 会变得很尖，梯度集中到少数位置。

除以 $\sqrt{d_k}$ 后，分数的方差大致回到常数量级。它不能保证训练永远稳定，但能减少维度变化给 softmax 带来的尺度偏移。

### 2.3 mask 放在哪里

mask 加在 softmax 之前。被屏蔽位置的分数设成 $-\infty$（工程中也常用当前 dtype 能表示的很小值），softmax 后对应权重就是 0。

常见 mask 有两种：

- padding mask：不读取补齐出来的 token；
- causal mask：第 $t$ 个位置只能读取自己和过去，不能看到未来 token。

causal mask 的分数矩阵是下三角可见：

$$
M_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

如果某一整行全被 mask，softmax 可能产生 `NaN`。构造 batch mask 时要保证每个有效 query 至少能看到一个 key。

## 3. Self-attention 与 cross-attention

设输入序列为 $X\in\mathbb{R}^{B\times L\times d_{model}}$，通过三个线性层得到：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

Q、K、V 都来自同一个序列时，称为 self-attention。这里的“self”表示信息源相同，并不表示一个 token 只关注自己。

cross-attention 使用不同的信息源。例如解码器状态产生 $Q$，编码器输出产生 $K$ 和 $V$：

$$
Q=YW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

这样，解码器的每个位置都能按需读取输入序列。

## 4. Multi-head attention

单头 attention 只在一个投影空间中计算匹配。多头注意力把通道分成 $h$ 份，每一头有自己的投影：

$$
\operatorname{head}_i
=
\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V),
$$

$$
\operatorname{MHA}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W_O.
$$

通常取 $d_{head}=d_{model}/h$。多头并不会凭公式保证每一头学出“语法”“位置”之类固定功能，只是给模型提供多个独立的匹配子空间。某些头会出现可解释模式，也有一些头的作用高度重叠。

### 4.1 形状怎么变

以 $B=2$、$L=4$、$d_{model}=8$、$h=2$ 为例：

```text
X                 (2, 4, 8)
Q, K, V           (2, 4, 8)
拆成 2 个头        (2, 2, 4, 4)   # B, H, L, D_head
attention scores  (2, 2, 4, 4)   # B, H, L_q, L_k
每头输出           (2, 2, 4, 4)
拼回通道           (2, 4, 8)
```

最常见的 bug 就在转置这里：分头后要把 head 维移到序列维之前，否则矩阵乘法会在错误的维度上进行。

## 5. 一份可运行的 PyTorch 实现

下面的实现保留了 padding mask、causal mask 和 attention dropout。输入、输出形状均为 `(batch, seq_len, d_model)`。

```python
import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape

        # (B, L, 3D) -> (3, B, H, L, D_head)
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if padding_mask is not None:
            # padding_mask: (B, L)，True 表示 padding 位置
            scores = scores.masked_fill(
                padding_mask[:, None, None, :], float("-inf")
            )

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        context = weights @ v

        # (B, H, L, D_head) -> (B, L, D)
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, d_model)
        return self.out_proj(context), weights
```

调试时可以先检查两件事：

```python
x = torch.randn(2, 4, 8)
layer = MultiHeadSelfAttention(d_model=8, num_heads=2)
output, weights = layer(x, causal=True)

assert output.shape == (2, 4, 8)
assert weights.shape == (2, 2, 4, 4)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones_like(weights.sum(dim=-1)),
)
```

开启 dropout 后，训练模式中的权重行和不再严格等于 1，因为 dropout 会对 softmax 权重做随机置零和缩放。检查归一化性质时，应把模块切到 `eval()`，或把 dropout 设为 0。

实际项目优先使用 `torch.nn.functional.scaled_dot_product_attention` 或 `nn.MultiheadAttention`。框架可以按设备和 dtype 选择更省显存的内核；手写版本更适合核对公式和张量形状。

## 6. Attention 放进 Transformer 后还缺什么

attention 自身没有序列顺序概念。若不加入位置编码，同一组 token 只要同时打乱，输出也会按相同方式打乱。Transformer 因此需要绝对位置编码、相对位置偏置或 RoPE 等位置信息。

一个常见的 pre-norm block 可以写成：

$$
X' = X + \operatorname{MHA}(\operatorname{LN}(X)),
$$

$$
Y = X' + \operatorname{FFN}(\operatorname{LN}(X')).
$$

这里还有三块不能忽略：

- residual connection 保留原表示，也给梯度提供较短路径；
- layer normalization 控制每个 token 的特征尺度；
- FFN 对每个位置独立做通道混合，attention 负责位置之间的信息交换。

不同模型会调整归一化位置、激活函数、门控结构和 residual 缩放，不能只凭一张 Transformer 示意图推断实现细节。

## 7. 计算量与长序列问题

标准 self-attention 的分数矩阵大小为 $L\times L$。只看 attention 部分，时间和显存复杂度通常记为 $O(L^2)$；投影层还包含约 $O(Ld_{model}^2)$ 的计算。序列很长时，分数矩阵往往先成为瓶颈。

常见改进思路包括：

- 使用优化后的精确 attention 内核，避免显式保存完整分数矩阵；
- 局部或滑动窗口 attention，只读取邻域；
- 稀疏、分块或低秩近似；
- 自回归推理时缓存过去 token 的 K、V，避免每一步重复投影。

KV cache 减少的是重复计算，不会消除上下文长度带来的缓存增长。多查询注意力（MQA）和分组查询注意力（GQA）通过让多个 query heads 共享较少的 K/V heads，进一步减少推理缓存。

## 8. 容易写错的地方

1. **缩放维度写错。** 分母是 $\sqrt{d_{head}}$，不是总的 $\sqrt{d_{model}}$，除非实现里只有一个头。
2. **softmax 维度写错。** 应沿 key 维，也就是分数矩阵的最后一维。
3. **mask 时机写错。** mask 应作用于 softmax 前的 logits；softmax 后再乘 0 会破坏归一化。
4. **把 attention 权重当作完整解释。** 权重能显示一次信息混合的分配，但模型还有 value 投影、输出投影、残差和后续层，不能仅凭一张热力图解释最终预测。
5. **忽略位置编码。** self-attention 本身只看内容匹配，不知道 token 的先后顺序。
6. **混淆 `eval()` 与禁用梯度。** `eval()` 关闭 dropout 等训练行为，`torch.no_grad()` 或 `torch.inference_mode()` 才停止构建反向图。

## 9. 我目前的理解

可以把 attention 记成两步：先用 $QK^\top$ 决定“从哪里读”，再用权重乘 $V$ 决定“读出什么”。多头、mask、位置编码和 residual 都是在这两步之外补表达能力或约束。遇到新变体时，先核对 Q、K、V 的来源、分数矩阵的可见范围和输出如何回到 residual stream，通常就能看懂大半。
