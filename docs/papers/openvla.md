# OpenVLA：让视觉语言模型直接控制机器人

> 论文：Kim 等，CoRL 2024。
>
> 先记住三句话：
>
> 1. OpenVLA 的输入是一张 RGB 图和一句任务指令，输出一个 7 维机器人动作。
> 2. 它把连续动作离散成 7 个 token，因此可以像语言模型生成单词一样生成动作。
> 3. 它的优势是语义理解、跨任务预训练和完全开源；短板是单图、单步、约 6 Hz，精细连续控制仍不如 action-chunk 策略。

## 一、论文对应的 Task 是什么？

### 1. 任务定义

OpenVLA 研究的是语言条件下的通用机器人操作：机器人看到当前场景，收到一句自然语言命令，然后直接产生低层控制动作。

例如：

- “把茄子放进锅里”；
- “拿起白色胶带”；
- “把蓝色杯子叠到粉色杯子上”；
- “用毛巾盖住指定物体”；
- “把桌上的碎屑扫进簸箕”。

这不是视觉问答。模型不能只回答“应该拿茄子”，而要真正控制机械臂把任务做完。

### 2. 实际任务的输入与输出

部署时输入：

- 当前的一张第三人称 RGB 图像；
- 一句自然语言任务指令。

OpenVLA 不输入：

- 机器人关节角或末端位姿；
- 历史图像；
- 深度、点云或腕部相机；
- 未来动作历史。

模型每次输出一个 7 维相对末端动作：

```text
[Δx, Δy, Δz, Δrot1, Δrot2, Δrot3, gripper]
```

即 3 维平移、3 维旋转和 1 维夹爪控制。执行后再拍一张新图，继续预测下一步（Figure 2，p. 4；Section 6，p. 11）。

旋转的具体参数化、动作单位和安全裁剪范围，论文中未明确说明。

### 3. 为什么值得研究？

传统机器人策略通常只会少数任务，而且需要为新机器人重新采集大量数据。机器人数据中对“Taylor Swift 照片”“AAA 电池”“紫葡萄”等开放世界概念的覆盖也很少。

视觉语言模型已经从互联网学到了大量物体和语言知识。OpenVLA 想回答：能否把这些知识迁移到机器人控制，并把模型、代码、训练方法全部开放？

**一句话 Task：给定一张当前 RGB 图像和自然语言指令，持续预测 7 维相对动作，使机器人完成跨场景、跨任务并可少量数据适配的操作。**

## 二、这个任务有哪些数据集，本文用了什么数据？

### 1. OpenVLA 预训练数据

核心训练数据来自 Open X-Embodiment（OpenX/OXE）。这是一个汇集许多真实机器人数据集的大型集合。

OpenVLA 从中筛选约 97 万条真实机器人轨迹。主要来源包括 BridgeData V2、Fractal/RT-1、Kuka、DROID 等（Section 3.3，pp. 5–6；Appendix Table 3，p. 21）。

这些数据差异很大：机器人不同、相机不同、任务不同、动作范围也不同。作者做了三件重要的数据工程：

1. 只保留至少有一个第三人称相机、能够转换成单臂末端控制的数据。
2. 采用类似 Octo 的采样权重，避免低多样性的大数据集支配训练。
3. 删除 Bridge 每条轨迹的第一个全零动作，否则单步模型很容易学会一直不动。

DROID 最初占 10% 权重，但 action-token accuracy 长期较低，因此最后三分之一训练中被移除。这个事实也说明，跨机器人数据还没有真正做到“无缝混合”。

### 2. 新机器人适配数据

作者还在 Franka 上采集少量新任务演示，数量约 10–150 条，包括：

- 把胡萝卜放进碗；
- 把玉米倒入锅；
- 按语言选择物体并移动到盘子；
- 用毛巾盖住指定物体；
- 擦桌子并处理干扰物。

采集者、遥操作设备、失败演示处理和验证集划分，论文中未明确说明。

### 3. LIBERO 仿真数据

LIBERO 包含 4 个 suite，每个 suite 有 10 个任务，每任务 50 条人类示范。作者重新渲染为 `256×256`，删除 no-op 和无法重放成功的演示，并把图像旋转 180°以匹配硬件设置（Appendix E，pp. 36–37）。

因此，本文 LIBERO 结果不能直接和未做这些处理的旧论文数字简单比较。

### 4. 一条数据究竟长什么样？

以 Bridge 的“Put red bottle into pot”为例，一条原始轨迹可以概括为：

```text
episode = {
  instruction: "put red bottle into pot",
  images:       [I_1, I_2, ..., I_T],
  actions:      [a_1, a_2, ..., a_T]
}
```

每个 `a_t` 是与 `I_t` 对齐的单步专家动作。

训练时不会把整条轨迹一起输入，而是取其中一个时刻：

```text
sample_t = {
  image:       当前一张224×224 RGB,
  instruction: 任务语言,
  target:      当前一个7维动作
}
```

必须分清：

- 原始 dataset 是变长 episode；
- 训练 sample 是一个时刻的图像—语言—动作；
- 7 个 action tokens 表示同一时刻动作的 7 个维度，不是未来 7 步。

## 三、已有工作怎样解决这个 Task？

### 1. RT-1-X

RT-1-X 是约 35M 参数的通用策略，主要从机器人数据学习，没有完整 Internet VLM 先验。

当场景中有多个候选物体、背景改变或出现新类别时，它容易抓错目标或无目的移动。Bridge 总成功率为 18.5%，language grounding 为 30%；OpenVLA 分别达到 70.6% 和 90%（Figure 3，p. 7；Table 4，p. 26）。

实际问题是：机器人数据很难覆盖所有物体—词语组合，仅靠这些数据容易记住位置和背景捷径。

### 2. Octo

Octo 是 93M 参数的开源通用机器人策略，也使用 OpenX 数据。它具有跨机器人动作先验，但控制模块并不是把完整 VLM 端到端变成动作生成器。

Octo 在部分新机器人任务上很强，例如用毛巾覆盖物体；但 Bridge semantic generalization 为 0%，说明它在开放语言指代上仍有明显困难（Figure 3，p. 7）。

### 3. RT-2-X

RT-2-X 是 55B 参数的闭源 VLA，也把动作当 token。它具有很强的语义能力，但无法下载、重新训练或在新机器人上微调。

它在 Bridge 上还受到首步 no-op 数据污染影响；研究者只能在推理时绕过最高概率的停止动作。OpenVLA 可以直接清洗数据后重新训练，这是“开放”带来的实际价值（Appendix C，pp. 32–33）。

但 RT-2-X 并非全面落后。Bridge semantic generalization 为 38.8%，略高于 OpenVLA 的 36.3%；Google Robot OOD 两者同为 82.9%。

### 4. Diffusion Policy

Diffusion Policy 从目标任务小数据训练，使用历史图像和 proprioception，并生成未来动作块。它在窄而精细的任务上动作更平滑：例如把玉米倒进锅可达到 100%。

但当三件物体同时出现、语言决定操作哪一个时，它缺少开放世界语义先验。`Move <object> onto Plate` 的 ID/OOD 只有 25.0%/8.3%，OpenVLA 为 75.0%/58.3%（Table 7，p. 32）。

两者解决的是不同问题：Diffusion 更擅长“怎样连续、精确地动”，OpenVLA 更擅长“应该对哪个物体做什么”。

## 四、本文方案有什么特点和创新？

| Baseline 的问题 | OpenVLA 的设计 | 为什么有帮助 | 证据 |
| --- | --- | --- | --- |
| 小策略语义和 OOD 泛化弱 | 从 7B Prismatic VLM 初始化并端到端训练 | 复用 Internet 视觉语言知识 | Fig. 3–4 |
| 单一视觉特征难兼顾语义和空间 | SigLIP + DINOv2 双视觉编码器 | 前者偏语义，后者补局部空间信息 | Table 9–10 |
| LLM 不能直接输出连续数值 | 7 维动作分别量化成 256 个 token | 可以直接使用 next-token prediction | Section 3.2 |
| OpenX 数据异构、含无效动作 | 筛选、重采样、删除 Bridge no-op | 减少数据污染和大数据源支配 | Table 9、Appendix C |
| 7B 微调和部署昂贵 | LoRA 与 4-bit 量化 | 降低可训练参数和显存 | Table 1–2 |

OpenVLA 最重要的贡献是完整开放配方，不是某个全新的数学公式：

**开放 VLM + 97 万机器人轨迹 + action tokenization + 高效微调/部署工具。**

## 五、本文的具体方案

### 1. 整体流程

```text
一张224×224 RGB图
↓ SigLIP 和 DINOv2
融合视觉 patch tokens
↓ 两层 MLP 投影
Llama可读取的视觉 tokens

语言指令
↓ tokenizer
文本 tokens

视觉 tokens + 文本 tokens
↓ Llama 2 7B 自回归生成
7个动作 tokens
↓ 反量化
一个7维连续动作
↓ 执行并重新拍图
```

它使用离线行为克隆，不使用 RL。预训练后可以直接部署，也可以用新机器人小数据做 full fine-tuning 或 LoRA。

### 2. 双视觉编码器在做什么？

同一张图分别进入：

- SigLIP：擅长图像和语言语义对齐，例如识别“茄子”“电池”；
- DINOv2：补充局部形状和空间结构，例如目标与夹爪的位置关系。

两套特征在同一个图像 patch 的通道维拼接，再由两层 MLP 投影到 Llama embedding 空间（Section 3.1，p. 4）。

视觉 token 数、patch size 和投影层隐藏维度，论文中未明确说明。

### 3. 连续动作怎样变成 token？

对动作的每一个维度，作者先统计训练数据的 1% 和 99% 分位数，再把中间范围均匀分成 256 个 bins。

例如某一维动作落在第 42 个 bin，就映射成一个特殊 token。7 个动作维度最终得到 7 个 token：

```text
[x-token, y-token, z-token,
 rot1-token, rot2-token, rot3-token,
 gripper-token]
```

为什么不用最小值和最大值？极端异常动作会把范围拉得很宽，使常见动作的量化变粗。1%–99% 分位数可以把 256 个格子集中在常见范围（Section 3.2，p. 5）。

论文没有比较 128、256、512 bins，也没有与连续动作头做直接消融，因此不能说 256 一定最优。

### 4. 训练过程

最终预训练 batch size 是 2048。一个样本包含：

```text
当前224×224 RGB图
任务指令
专家7维动作 → 7个action token IDs
```

训练步骤：

1. 双视觉编码器提取 patch 特征。
2. 指令套入类似 `What should the robot do to {task}? A:` 的 prompt。
3. Llama 依次预测 7 个动作 token。
4. 用真实前序 token 做 teacher forcing。
5. 只在动作 token 位置计算交叉熵。

所以它预测的是 256 类 token 分布，不是连续 MSE，也不是 diffusion noise。

训练从 Prismatic-7B 开始，视觉编码器、projector 和 Llama 全部更新；固定学习率 `2×10^-5`，训练 27 epochs。作者使用 64 张 A100 训练 14 天，约 21,500 A100-hours（Section 3.4–3.5，p. 6）。

### 5. 推理过程

以“把红色瓶子放入锅里”为例：

1. 拍摄当前一张 RGB 图。
2. 与语言指令一起编码。
3. Llama 自回归生成 7 个 action tokens。
4. 将 token 反量化为 7 维连续动作。
5. 机器人执行该动作。
6. 重新拍图并重复。

模型约 6 Hz，因此每秒只能进行约 6 次这样的闭环。论文没有明确说明控制器如何在更高频率下插值或保持动作。

### 6. LoRA 和量化

LoRA 不更新 7B 全部参数，而是在各线性层加入低秩增量。`r=32` 只训练约 97.6M 参数，占 1.4%，成功率 68.2%；全参数微调为 69.7%（Table 1，p. 10）。

4-bit 推理显存约 7.0 GB，bfloat16 为 16.8 GB，而成功率在误差范围内接近（Table 2，p. 10）。

## 六、本文效果如何？

### 1. 评估设置

- Bridge：17 个真实任务，每个方法每任务 10 次。
- Google Robot：12 个任务，每个方法每任务 5 次。
- Franka：新任务小数据微调，共 129 次 rollout。
- LIBERO：4 个 suite；每个方法、每个 suite、每个 seed 500 次，3 个 seeds。

主要指标是 Success Rate。部分困难任务允许 0.5 分，表示部分完成。Google 每任务只有 5 次，所以一次成败就会改变 20 个百分点；不能过度解释很小的差异。

### 2. Bridge 与 Google Robot

| 平台/类别 | RT-1-X | Octo | RT-2-X | OpenVLA |
| --- | ---: | ---: | ---: | ---: |
| Bridge Overall | 18.5 | 20.0 | 50.6 | **70.6** |
| Bridge Visual | 8.0 | 29.0 | 52.0 | **87.0** |
| Bridge Semantic | 26.3 | 0.0 | **38.8** | 36.3 |
| Bridge Language Grounding | 30.0 | 40.0 | 85.0 | **90.0** |
| Google Overall | 33.3 | 26.7 | 78.3 | **85.0** |
| Google OOD | 34.3 | 14.3 | **82.9** | **82.9** |

OpenVLA 对 RT-1-X 和 Octo 有明显优势；Bridge 上也显著高于 RT-2-X。Google 上 OpenVLA 与 RT-2-X 的误差条重叠，不能说显著获胜（Figures 3–4，pp. 7–8；Tables 4/6，pp. 26/28）。

### 3. 新机器人适配

| 平台 | Diffusion | Octo FT | OpenVLA scratch | OpenVLA FT |
| --- | ---: | ---: | ---: | ---: |
| Franka-Tabletop | 48.5 | 43.4 | 43.4 | **67.2** |
| Franka-DROID | 35.0 | 38.3 | 21.7 | **58.3** |

OpenVLA 微调后总体最好，说明 OpenX 预训练确实可以迁移到新机器人。但具体任务上并非总赢：窄而精确的单指令任务，Diffusion Policy 可达到 90–100%；Octo 在 Cover with Towel 上也更强（Table 7，p. 32）。

### 4. LIBERO

| 方法 | 平均成功率 |
| --- | ---: |
| Diffusion Policy scratch | 72.4 |
| Octo fine-tuned | 75.1 |
| OpenVLA fine-tuned | **76.5** |

OpenVLA 只比 Octo 高 1.4 个百分点，而且在 Object/Goal suite 不是第一。作者认为原因是 OpenVLA 预训练都来自真实机器人，存在 sim-to-real domain gap；但论文没有直接消融这一解释（Table 12，p. 37）。

### 5. 关键消融

| 改动 | 结果 | 能说明什么 |
| --- | ---: | --- |
| 完整 OpenX 预训练 | 76.3 | 基准 |
| 只用 Bridge 训练 | 45.6 | 跨机器人数据是主要收益来源之一 |
| Bridge-only 去掉 DINOv2 | 40.6 | 双视觉编码器有帮助，但增益较小 |
| 微调视觉 encoder | 80.0 | 控制任务需要适配视觉特征 |
| 冻结视觉 encoder | 46.7 | 只冻结通用视觉特征效果较差 |
| LoRA r=32 | 68.2 | 接近全参数微调 69.7 |

需要注意：论文没有干净比较离散 token head 与 continuous/diffusion head，也没有比较不同 bin 数。因此实验支持“完整 OpenVLA 配方有效”，不能把所有提升都归给 action tokenization。

## 七、论文的不足和改进方向

### 1. 作者明确指出的不足

1. 只看当前一张图，没有历史和机器人本体状态。
2. 一次只输出一个动作，不支持 action chunk。
3. 约 6 Hz 推理，难以直接控制 50 Hz 双臂机器人。
4. 典型成功率仍低于 90%，还达不到高可靠部署。
5. 精细窄任务的动作平滑性不如 Diffusion Policy。

### 2. 我们还应注意什么？

- “开放”主要指模型、代码和机器人数据配方；SigLIP、DINOv2、Llama 2 的原始预训练数据并非全部开放。
- Bridge 清除了 no-op，而 RT-2-X 无法重训，主表混入了数据清洗差异。
- 真实测试次数较少，多项差距没有显著性检验。
- 7B 对每次只生成 7 个动作 token 而言很重，计算效率不理想。
- action quantile 如何跨不同机器人和控制频率迁移，没有完整说明。
- 没有碰撞率、不确定性、安全停止和失败恢复指标。

### 3. 可以怎样改进？

| 现有问题 | 改进思路 | 怎样验证 |
| --- | --- | --- |
| 语义强但动作不够平滑 | VLM 提供条件，接 diffusion/flow action-chunk head | 同输入下比较单步 token 与连续 chunk |
| 单图缺少速度和遮挡信息 | 加短视频、多相机和 proprioception，并压缩成少量控制 token | 测遮挡、快速移动和接触任务 |
| 6 Hz 延迟高 | 并行动作头或蒸馏到高频小模型 | 注入不同延迟，画成功率—延迟曲线 |
| OpenX 混合靠经验 | 在固定算力下学习各数据源权重 | 在完全未见机器人上评估正/负迁移 |
| 缺少安全与恢复 | 输出置信度，加入本地安全控制器和拒绝机制 | 统计碰撞、急停和恢复成功率 |

## 汇报时最值得讲的 3 个点

1. **语言模型如何输出动作。** 7 维连续动作分别量化为 256 bins，再复用 Llama 词表中的 256 个低频 token。
2. **数据多样性比小模块更关键。** 去掉 OpenX 预训练下降 30.7 点；去掉 DINOv2 只下降约 5 点。
3. **OpenVLA 和 Diffusion Policy 是互补的。** 前者更懂“做什么、对谁做”，后者更擅长“连续而精确地做”。

## 常见问题

### Q1：OpenVLA 输入机器人关节状态吗？

不输入。标准输入只有当前一张第三人称 RGB 图和语言指令。

### Q2：7 个 token 是未来 7 步吗？

不是。它们是同一时刻 7 个动作维度。

### Q3：为什么用 256 bins？

256 对应 8-bit 动作符号，也方便复用 256 个低频词表 token。但论文没有证明 256 是最优值。

### Q4：双视觉编码器是主要提升来源吗？

不是。消融中它贡献约 5 点，而跨机器人 OpenX 预训练贡献约 30.7 点。

### Q5：OpenVLA 全面超过 RT-2-X 吗？

没有。Bridge 总体更强，但 Bridge semantic 略低；Google OOD 两者相同。

### Q6：LoRA 是否意味着普通显卡就能轻松训练？

LoRA 大幅减少可训练参数，但论文中 batch 16 仍报告约 59.7 GB 显存。它降低了门槛，不等于任何消费级显卡都能直接复现。

---

## 附录：术语表

- VLA（Vision-Language-Action）：根据图像和语言直接输出机器人动作的模型。
- VLM（Vision-Language Model）：共同理解图像和文字的模型。
- Action Tokenization：把连续动作量化成离散 token。
- Autoregressive：按顺序生成 token，后一个依赖前面已经生成的 token。
- Proprioception：机器人对自身关节、位姿等状态的感知。
- Language Grounding：把语言中的对象和动作对应到当前真实场景。
- OOD（Out of Distribution）：测试条件与训练分布不同。
- LoRA：只训练低秩增量参数的高效微调方法。
- Quantization：用更低位数表示模型权重，以节省显存和计算。
- No-op：不产生运动的零动作。

## 阅读说明

- Python 环境可在 PowerShell 中用 `conda activate robotic` 激活。
- ARS `pdf_read_preflight.py` 结构预检为 **PASS**，37/37 页，warnings 为空。
- 本报告中的页码、表格和结论来自论文原文；【我的分析】和【推断】不是作者明确结论。


---

[阅读论文 PDF](../assets/papers/openVLA.pdf) · [返回论文阅读](index.md)
