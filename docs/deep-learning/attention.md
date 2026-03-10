# 深入理解自注意力机制 (Self-Attention)

## 1. 直觉理解：什么是 Q、K、V？
* **Query (Q)**：代表“我想要什么”（比如我正在查询的字）。
* **Key (K)**：代表“别人有什么特征”（比如上下文中的其他字）。
* **Value (V)**：代表“别人实际的内容/信息”。
* **计算核心**：Q 和 K 计算相似度，谁的相似度高，谁的 V 的权重就大。

## 2. 核心数学公式推导
注意力机制的灵魂就是下面这个公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**面试常考点记录：**
* **为什么要做 $QK^T$（点积）？** 
  *写你的理解：点积在几何上代表两个向量的相似度...*
* **为什么要除以 $\sqrt{d_k}$（缩放因子 Scaling）？** 
  *写你的理解（非常重要！）：防止维度 $d_k$ 过大导致点积结果极大，从而让 Softmax 进入梯度消失的饱和区...*

## 3. PyTorch 代码手撕 (面试必考)
*(提示：作为软工学生，贴上代码能极大增加你的工程说服力。你可以参考以下代码自己默写一遍)*

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        # 定义生成 Q, K, V 的线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 1. 计算 Q * K^T
        # K.transpose(-2, -1) 是把最后两个维度转置
        scores = torch.matmul(Q, K.transpose(-2, -1)) 
        
        # 2. 缩放 (Scale)
        scores = scores / math.sqrt(self.d_model)
        
        # 3. Softmax 归一化得到注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 4. 乘以 V 得到最终输出
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
```