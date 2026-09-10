# Octo：可灵活适配新机器人的开源通用策略

> 论文：Octo Model Team，arXiv:2405.12213v2，2024。
>
> 先记住三句话：
>
> 1. Octo 用 80 万条多机器人轨迹预训练，希望以后学习新机器人时不必从零开始。
> 2. 它把语言、目标图像、多相机观察都变成 token，再由 Transformer 统一处理。
> 3. Transformer 只运行一次，后面的多步 diffusion 全部在小型 action head 中完成，因此比“每次去噪都跑大模型”省计算。

## 一、论文对应的 Task 是什么？

### 1. 任务定义

论文研究的是 Generalist Robot Policy，也就是通用机器人策略：先在许多机器人和任务的数据上预训练一个基础策略，再直接使用或用少量新数据微调到新任务。

它想解决的不是一个固定任务，而是一类机器人操作问题：

| 场景 | 机器人要完成什么 |
| --- | --- |
| WidowX BridgeV2 | 把胡萝卜放到盘子、把茄子放入锅中 |
| UR5 Tabletop | 从一个碗拿出玩具老虎并放到另一个碗、用布擦桌面 |
| Stanford Coffee | 选择咖啡胶囊并精确放进咖啡机 |
| CMU Baking | 拿起面包、放进烤面包机、关上机器 |
| Berkeley Insertion | 根据视觉和力/力矩，把插销插进只有 1.5 mm 余量的孔 |
| Berkeley Bimanual | 双臂拿起记号笔并拔掉笔帽 |

这些任务的机器人、相机、动作空间和控制频率并不相同。Octo 的重点是：**同一个预训练模型能否适应这些不同接口？**（Figure 1；Figure 4，p. 6）

### 2. 实际任务的输入

Octo 支持比较灵活的输入：

- 一张或多张 RGB 图像，例如第三人称相机和腕部相机；
- 最近两帧观察历史；
- 两种任务描述任选其一：自然语言或目标图像；
- 微调时还可加入新的传感器，例如机器人状态、末端速度、力/力矩。

目标图像表示“任务做完后应该是什么样”。例如语言是“把面包放到盘子”，目标图像可以直接展示面包已经位于盘子上的状态。

预训练中并非所有数据都有语言或腕部相机：只有 56% 数据有语言标注，27% 有腕部相机（Discussion，p. 8）。缺失的输入会被 mask 或补零。

### 3. 实际任务的输出

Octo 输出的是一段连续机器人动作，即 action chunk。

预训练和 zero-shot 测试使用 delta end-effector control，可理解为末端执行器相对移动、旋转和夹爪控制。论文没有在正文中完整列出标准动作的确切维度和旋转表示，因此不应自行写死为某一种 7D 定义。

微调时可以替换 action head，适配新的动作空间。例如：

- Berkeley Pick-Up 使用关节位置控制；
- Berkeley Insertion 输出末端 twist；
- Berkeley Bimanual 输出 14 维动作，即两条 6-DoF 手臂和两个夹爪（Appendix F，pp. 16–17）。

论文明确使用 receding-horizon control：预测一段动作，只执行前一部分，再根据新观察重新规划。标准预训练 action chunk 的具体长度和每次执行多少步，论文中未明确说明；双臂任务单独使用长度 64，并执行前 12 步。

### 4. 为什么值得研究？

传统做法通常是：每换一个机器人或任务，就重新采数据、设计输入输出并从头训练。这会重复消耗人力，而且少量数据训练出的策略泛化很窄。

已有通用策略又常有两个限制：

1. 输入输出接口固定，只能用预训练时那一种相机或动作空间；
2. 大模型或训练代码不公开，研究者很难微调和复现。

Octo 希望提供一个可下载、可修改的基础策略，让新任务只需约 100 条示范和数小时微调。

**一句话 Task：给定语言或目标图像、最近的多相机观察以及可选机器人状态，生成连续动作块，并能通过少量微调适应新的传感器、动作空间和机器人。**

## 二、这个任务有哪些数据集，本文用了什么数据？

### 1. Open X-Embodiment

Open X-Embodiment（OXE/OpenX）汇集了多个机构、机器人和任务的数据。论文报告当时整个集合约有 150 万条 episode，Octo 从中筛选 80 万条用于训练（Related Work，p. 3）。

作者最终选取 25 个子数据集，最大的三项是：

| 数据集 | 训练采样权重 |
| --- | ---: |
| Fractal | 17.0% |
| Kuka | 17.0% |
| Bridge | 17.0% |
| BC-Z | 9.1% |
| Stanford Hydra | 6.0% |
| Language Table | 5.9% |

完整 25 项见 Appendix Table III（p. 14）。这些是训练采样权重，不一定等于原始数据量比例。

### 2. 数据如何筛选和统一？

作者先删除：

- 没有图像的数据集；
- 不能转换为 delta end-effector control 的数据集；
- 过度重复、分辨率太低或任务过于狭窄的数据集。

然后把数据粗略分为“更丰富”和“较少样”，将更丰富数据的权重加倍，并降低重复 episode 很多的数据集权重。这个分类和权重主要由人工经验决定（Section III-B，pp. 4–5）。

跨数据集还进行了两个统一处理：

- 缺失相机通道补零并 mask；
- 夹爪动作统一为 `+1=打开，0=闭合`。

不同数据的时间频率、坐标系和动作转换细节，论文中没有完整列出。

### 3. 一条原始数据包含什么？

一条原始 episode 可以概括为：

```text
episode = {
  primary_images: [I_1, ..., I_T],
  wrist_images:   可选,
  language:       可选的任务指令,
  actions:        [a_1, ..., a_T],
  robot_state:    某些数据集可提供，但不是统一必需字段
}
```

原始 episode 是完整轨迹，不是一段固定长度的 action chunk。

训练时从中构造 sample：

```text
sample_t = {
  observation_history: 最近2帧图像,
  task:                language 或 future goal image,
  target:              从当前时刻开始的专家动作块
}
```

目标图像不一定是人工标注。作者使用 hindsight goal relabeling，从同一轨迹未来随机选一个状态的图像作为 goal（Section III-D，p. 5）。

每条轨迹最多随机抽 100 个时刻进入数据加载器，以免超长 episode 塞满 shuffle buffer（Appendix E，p. 15）。

### 4. 训练、验证和测试怎样划分？

- 预训练：80 万条筛选后的 OXE episode。
- 验证集：论文中未明确说明固定 validation split 和 checkpoint 选择规则。
- Zero-shot 测试：来自预训练机器人和任务，但改变物体位置、光照、背景和干扰物。
- 微调测试：6 个新设置，每项约 100 条目标演示，真实机器人测试通常 20 次。

所以论文所谓 zero-shot 不是“从未见过的新技能”。它主要是预训练任务上的直接调用和场景变化泛化。

## 三、已有工作怎样解决这个 Task？

### 1. RT-1-X

RT-1-X 是约 35M 参数的开源通用策略，也在 OpenX 上训练，但只使用约 35 万条 episode，输入输出接口更固定。

具体局限：换相机组合或动作空间时，原架构的大模块可能需要重新初始化。Figure 5 中，Octo 在 WidowX、UR5 和 RT-1 Robot 上平均比 RT-1-X 高 29 个百分点（pp. 6–7）。

但比较同时改变了模型结构、模型大小和数据量，因此不能把差距全部归因于 Octo 的某一个模块。

### 2. RT-2-X

RT-2-X 是 55B 视觉语言动作模型，也在约 35 万条 OpenX episode 上训练。它有很强的视觉语言能力，但权重和完整训练流程不公开，也没有展示怎样灵活增加新传感器或动作空间。

Octo 在 WidowX 和 RT-1 Robot 的 zero-shot 任务上与 RT-2-X 表现相近（Figure 5，p. 6）。其中 WidowX 的 RT-2-X 数字来自另一篇论文，不是作者在完全相同实验中重新运行；这一比较应谨慎解读。

### 3. 从头训练的 ResNet + Transformer

这个 28M baseline 使用 ResNet、FiLM 语言条件和小型 Transformer diffusion decoder，在每个新任务的约 100 条示范上从头训练。

问题是小数据覆盖窄，也不能复用其它机器人学到的技能。Table I 中，六个新设置平均成功率只有 20%，Octo 微调后为 72%（p. 7）。

作者还发现，如果把 Octo 的大型 Transformer 从头训练，它会很快过拟合；因此从头 baseline 反而使用更适合小数据的 ResNet 架构。这个选择合理，但意味着“Octo pretrained vs same architecture scratch”并非主表中的直接对照。

### 4. VC-1 视觉预训练 baseline

VC-1 使用在 4,000 小时第一人称视频和 ImageNet 上预训练的 ViT-B，再接 MLP 以 MSE 回归动作。

它提供视觉特征，却没有预训练完整机器人策略，也没有跨机器人动作先验。六项微调平均只有 15%，说明“好视觉 encoder”不等于“好策略初始化”（Table I，p. 7）。

### 5. MSE 与离散动作头

Appendix E（p. 15）给出很具体的失败现象：

- MSE policy 会“hedging”，动作很慢，甚至不愿旋转夹爪；因为多种合理动作被平均。
- 256-bin 离散动作更果断，但精度不足，经常错过抓取位置。
- Diffusion 保留连续精度，又能表示多种动作模式。

## 四、本文方案有什么特点和创新？

| Baseline 的问题 | Octo 的设计 | 为什么有帮助 | 证据 |
| --- | --- | --- | --- |
| 相机、任务条件和动作接口被架构写死 | 模块化 tokenizer + blockwise attention + readout token | 可新增输入或替换小型输出头，不必重建 backbone | Fig. 2；Table I |
| 语言标注不足 | 同时支持语言和 goal image，并用 hindsight relabeling | 没有语言的数据也能从未来图像获得任务监督 | Section III-D |
| MSE 平均化、离散动作不精确 | 小型 diffusion action head 生成连续 action chunk | 同时保留多模态性、精度和时间连贯性 | Table II；Appendix E |
| 多机器人数据异构且失衡 | 筛选 25 个 OXE 数据集、统一相机/夹爪、人工重权 | 建立可共享的训练接口并减少重复数据支配 | Fig. 3；Table III |
| 新机器人需要从零训练 | 保留 Transformer，仅增加新 tokenizer/head 后全量微调 | 复用已学到的视觉、任务和控制特征 | Table I |
| 小模型可能容量不足 | 10M、27M、93M 三档 scaling | 大模型更能处理场景变化和抓取时机 | Fig. 6 |

论文也很坦率地说明：Transformer、goal conditioning、action chunk 和 diffusion 都不是单独首次提出。贡献主要是把它们组合成一个**开放、可扩展、可微调的通用策略框架**（Introduction，p. 2）。

## 五、本文的具体方案

### 1. 整体流程

```text
语言或目标图像
↓ task tokenizer
Task tokens

最近2帧第三人称/腕部RGB
↓ shallow CNN + patchify
Observation tokens

Task + Observation tokens
↓ blockwise-masked Octo Transformer
Readout embedding
↓ 小型 diffusion action head，20步去噪
连续 action chunk
↓ 执行前一部分
获取新观察并重新规划
```

训练是离线模仿学习，不是 RL。微调继续使用同样的 diffusion objective，并更新全部模型参数。

### 2. 输入怎样变成 token？

#### 语言

语言先经过冻结的 `t5-base`，得到 16 个 language embedding tokens。T5-base 有 111M 参数，但不计入 Octo Transformer 的 27M/93M 参数数字（Section III-A，p. 4；Appendix D/E，p. 15）。

作者尝试过更大 T5 和微调最后两层，都没有提高控制表现。原因可能是机器人数据中的语言不够丰富，无法有效训练语言模型。

#### 图像

- 第三人称图像：随机 crop、resize 到 `256×256`、color jitter，像素归一化到 `[-1,1]`。
- 腕部图像：不随机 crop，resize 到 `128×128`。
- 图像先过浅层 CNN，再按 `16×16` patch 切分。
- 第三人称图像得到 256 tokens；腕部图像得到 64 tokens（Appendix D，pp. 14–15）。

#### 目标图像

目标图像用相同的图像 tokenizer 转成 task tokens。训练时随机屏蔽语言或 goal image，使同一模型能接受任一种任务定义；没有语言的数据始终用 goal image。

### 3. Transformer 怎样组织这些 token？

序列可以简化为：

```text
[Task tokens]
[第1帧 Observation tokens] [第1个 Readout token]
[第2帧 Observation tokens] [第2个 Readout token]
```

Observation token 只能关注 task 和当前/更早的观察，不能看未来观察。不存在的输入被完全 mask。

Readout token 类似 BERT 的 `[CLS]`：

- 它可以读取前面的 task 和 observation；
- observation 不能反过来读取它；
- action head 只接收 readout embedding。

因此以后增加一种传感器，只需增加新的 tokenizer 和位置 embedding；改变动作空间，则增加新的 readout/action head。预训练 Transformer 主体结构不必重做（Figure 2，pp. 3–4）。

### 4. 模型大小

| 模型 | 层数 | Hidden | MLP | Heads | 参数量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Octo-Small | 12 | 384 | 1536 | 6 | 27M |
| Octo-Base | 12 | 768 | 3072 | 12 | 93M |

另外还有实验用 Octo-Tiny，约 10M。Figure 6 显示模型越大，WidowX 和 UR5 的 zero-shot 成功率越高（Appendix Table V，p. 15）。

### 5. Diffusion action head

action head 是一个 3 层 MLP，hidden dimension 为 256，带 residual connection 和 LayerNorm。它使用 20 个 diffusion steps 和 cosine noise schedule（Appendix D，p. 15）。

训练直觉：

```text
真实专家action chunk
↓ 加随机高斯噪声
带噪action chunk
↓ action head读取readout embedding、噪声步k和带噪动作
预测应去掉的噪声
↓ DDPM loss
更新模型
```

论文用 `ε_θ(x_k,e,k)` 表示 denoising network。其输入是当前带噪动作 `x_k`、Transformer readout `e` 和 diffusion step `k`；输出用于把 `x_k` 更新为更干净的 `x_{k-1}`（Equation 1，p. 5）。

模型为什么不在每个去噪步骤都运行 Transformer？因为视觉、语言和观察没有改变。Octo 先运行一次 93M Transformer 得到 `e`，之后 20 步只运行很小的 MLP action head。这是它的重要效率设计。

### 6. 一个 batch 怎样训练？

逻辑上，一个样本包含：

```text
最近2帧 observation
language 或 future goal image
expert action chunk
输入缺失 mask
```

一个 batch 混合来自 25 个数据集的样本。训练步骤：

1. 图像、语言或 goal 分别 tokenization。
2. 按时间排列 token，加入位置 embedding。
3. Octo Transformer 输出 readout embedding。
4. 给专家 action chunk 加随机强度噪声。
5. diffusion head 预测噪声/去噪方向。
6. 计算标准 DDPM objective 并反向传播。

Octo-Base 训练配置：

- AdamW；
- learning rate `3×10^-4`；
- warmup 2,000 steps；
- inverse square-root decay；
- weight decay 0.1；
- gradient clipping 1.0；
- batch size 2,048；
- 训练 300k steps；
- TPU v4-128，约 14 小时（Section III-D，p. 5；Appendix Table IV，p. 14）。

### 7. 微调和推理

微调时，如果新机器人多了一种传感器或换了动作空间：

1. 添加新的输入 tokenizer 或 action head。
2. 其余预训练参数保留。
3. 用约 100 条目标演示训练 50k steps。
4. 使用 linear warmup + cosine learning-rate decay。
5. 更新整个模型，而不是只冻结 backbone。

一张 24 GB NVIDIA A5000 完成一次微调约需 5 小时（Section III-C/D，p. 5）。

部署时：

```text
当前两帧观察 + task
↓ Transformer只运行一次
随机高斯action chunk
↓ 小action head做20步去噪
连续动作块
↓ 执行前一部分并重新观察
```

论文没有统一报告 Octo-Base 的端到端推理延迟、各平台标准 action horizon 或执行长度。

### 8. 用一条 Berkeley Bimanual 数据走完整流程

任务是 ALOHA 双臂拔记号笔笔帽。Octo 预训练只有单臂数据，因此为它新建 14 维 action head。

```text
原始episode：
左腕RGB + 右腕RGB + 双臂14维专家动作序列
↓ 切训练sample
最近2帧双腕图像 + 未来64步动作
↓ 图像预处理
每张腕图128×128 → 64个tokens
↓ Octo Transformer
readout embedding
↓ 20步diffusion训练/采样
64×14 action chunk
↓ 部署时执行前12步
重新观察并规划
```

该任务只用新的双臂数据微调，最终测试成功率为 80%；scratch baseline 为 20%，VC-1 为 50%（Table I，p. 7；Appendix F，p. 17）。

## 六、本文效果如何？

### 1. 评估设置和指标

论文在 4 个机构的 9 种真实机器人设置上评估（Figure 4，p. 6）。

主要指标是 Success Rate：成功完成任务的 rollout 数除以总 rollout 数。

- Zero-shot：3 种机器人，每种选 2 个语言任务，每任务 10 次。
- 微调：6 个设置，主表称每个设置 20 次。
- Appendix F 又说 Berkeley Bimanual 只测试 10 次，与 Table I caption 的“每个 domain 20 trials”不一致，应视为论文报告细节冲突。
- 论文大多只给点估计，没有置信区间或多随机种子。

### 2. Zero-shot 结果

Figure 5（p. 6）显示：

- Octo 在 WidowX、UR5 和 RT-1 Robot 上均高于 RT-1-X；
- 三个平台平均高约 29 个百分点；
- 在 WidowX 和 RT-1 Robot 上，Octo 与 55B RT-2-X 表现相近；
- goal-image conditioning 在 WidowX 上比 language conditioning 高 25 个百分点（Section IV-A，pp. 6–7）。

但任务来自预训练分布，只改变物体位置、背景、光照和干扰物。因此更准确的说法是“对预训练机器人和技能的开箱调用”，不是全新技能 zero-shot。

Appendix Table VII（p. 17）进一步揭示边界：

| 泛化类型 | 平均成功率 |
| --- | ---: |
| 训练分布内任务 | 85% |
| 新物体 | 80% |
| 新环境 | 40% |
| 新技能 | 5% |

Octo 能把已学技能用到新物体，但几乎不会凭空获得“翻杯子、精确插槽”等新技能。

### 3. 新任务微调结果

| 设置 | Scratch | VC-1 | Octo |
| --- | ---: | ---: | ---: |
| Berkeley Insertion | 10% | 5% | **70%** |
| Stanford Coffee | 45% | 0% | **75%** |
| CMU Baking | 25% | 30% | **50%** |
| Berkeley Pick-Up | 0% | 0% | **60%** |
| Berkeley Coke | 20% | 10% | **100%** |
| Berkeley Bimanual | 20% | 50% | **80%** |
| 平均 | 20% | 15% | **72%** |

Octo 比次优 baseline 平均高 52 个百分点，而且同一微调 recipe 能适配：

- 新传感器：力/力矩；
- 新动作：关节位置、末端 twist；
- 新机器人：ViperX 和 ALOHA 双臂（Table I，p. 7）。

这是论文最有说服力的实验，因为它验证的不是“预训练任务记得好不好”，而是基础策略能否迁移到新接口。

### 4. 关键消融

Table II/VI（pp. 7/17）全部在 WidowX、4 个任务、40 次 trial 上评估：

| 改动 | 平均成功率 | 说明 |
| --- | ---: | --- |
| Octo-Small 完整配置 | **83%** | 基准 |
| 数据缩为 RT-X 的 11-dataset mixture | 60% | 数据广度很重要 |
| 只用单机器人 Bridge | 43% | 跨机器人数据提高泛化 |
| 离散 256-bin action | 18% | 动作果断但精度不足 |
| 连续 MSE action | 35% | 多种策略被平均，动作犹豫 |
| ResNet-50 + Transformer | 70% | 大数据规模下 ViT-first 更好 |

这组消融建立了比较清楚的逻辑：

```text
更宽的数据混合
+ Transformer-first 架构
+ Diffusion action head
= 最好的通用策略表现
```

但各消融只测试 40 次，缺少误差条；“数据集数量”也同时改变了数据量和数据分布，不能严格区分二者。

### 5. 模型规模

Figure 6（p. 8）比较约 10M、27M 和 93M 三种模型。WidowX 和 UR5 的 zero-shot 成功率随参数增加而提高。作者观察到 Base 更能等待合适抓取时机，也更能适应初始场景变化。

这支持 scaling trend，但只测试两个任务、每个任务 10 次，尚不足以建立可靠的机器人 scaling law。

## 七、论文的不足和改进方向

### 1. 作者明确指出的不足

1. 腕部相机利用不好，有时只用第三人称相机反而更强。
2. 只有 27% 预训练数据含腕部相机，导致该模态学习不足。
3. 只有 56% 数据含语言，语言条件明显弱于 goal image。
4. 训练数据主要是成功/最优演示，没有学习次优动作、失败和在线纠错。
5. 只训练和评估单臂/双臂操作，没有导航和移动操作。

### 2. 我们还应注意什么？

- “zero-shot”任务来自预训练技能，真正的新技能成功率只有 5%。
- 25 个数据集的筛选、丰富度分类和权重调整较主观，没有系统优化。
- 论文没有明确 validation split、标准 action chunk 长度和端到端推理延迟。
- 实验通常只有 10–20 次，缺少置信区间；Table I 与 Appendix F 的 trial 数还有冲突。
- 9 个设置跨 4 个机构，但仍由论文合作团队执行，不等于独立第三方复现。
- proprioception 在预实验中反而降低性能，说明模型可能把当前关节状态错误当成未来动作捷径；这个问题没有解决。
- action chunk 内仍是开环，突然碰撞或物体滑落时无法立刻反应。
- 没有报告碰撞率、力峰值、急停、人工接管和失败恢复。

### 3. 可以怎样改进？

| 现有问题 | 改进思路 | 怎样验证 |
| --- | --- | --- |
| 腕部相机数据少、利用差 | 做 camera-balanced sampling，并随机遮挡第三人称视角 | 比较遮挡、细小物体和精密插入表现 |
| 语言只覆盖 56% | 为轨迹生成可核验的分阶段语言，并训练 language-goal 对齐 | 测同义指令、组合指令和未见对象 |
| 新技能 zero-shot 只有 5% | 预训练可组合的 3D subgoal/contact skill，而不只模仿整段动作 | 严格留出动作原语组合做测试 |
| 数据权重靠人工 | 用 held-out 迁移收益或梯度相似度学习 mixture | 固定数据量和算力比较不同采样方法 |
| 20 步 diffusion 有延迟 | consistency/flow distillation 到 1–4 步 | 报告成功率、延迟、控制频率和 jerk |
| 缺失败恢复和安全 | 加扰动演示、在线纠正和本地 safety shield | 测滑落、碰撞、错误抓取后的恢复率 |

## 汇报时最值得讲的 3 个点

1. **为什么 Octo 容易换输入输出。** 所有输入先变成独立 token block，动作由被动读取信息的 readout token 接小 head；新增传感器不必改 Transformer 主干。
2. **Diffusion 为什么放在小 head。** Transformer 对观察只编码一次，20 次去噪只运行 3 层 MLP，兼顾动作多模态与计算效率。
3. **最强证据是微调，不是 zero-shot。** 新机器人、新力觉、新关节动作空间上平均 72%，而 scratch 20%；但真正未见技能 zero-shot 只有 5%。

## 常见问题

### Q1：Octo 是 VLA 吗？

它能接受语言并输出动作，可以归入广义 VLA；但不像 OpenVLA 那样让大语言模型自回归生成 action tokens。Octo 用冻结 T5 编码语言，用自己的 Transformer 和 diffusion head 生成连续动作。

### Q2：Octo 与 OpenVLA 最大差别是什么？

Octo 是 93M 模块化策略，支持多相机、两帧历史、goal image 和连续 action chunk；OpenVLA 是 7B VLM，用单图和语言自回归生成单步离散动作 token。前者更强调接口灵活与连续控制，后者更强调互联网语义。

### Q3：一条原始数据和训练 sample 有什么区别？

原始数据是一整条长度 `T` 的 episode；训练 sample 只取最近两帧、一个 task condition 和当前开始的动作块。

### Q4：目标图像从哪里来？

使用 hindsight goal relabeling，从同一演示未来随机选一帧作为目标。因此没有语言标注的数据也能训练 goal-conditioned policy。

### Q5：Diffusion 每一步都要运行 93M Transformer 吗？

不用。Transformer 只运行一次，20 次去噪都在隐藏维度 256 的小型 action head 中完成。

### Q6：为什么 MSE baseline 动作很慢？

同一状态可能有多个合理动作，MSE 倾向取平均。平均结果可能表现为犹豫、移动幅度小或不愿旋转。

### Q7：Octo 能直接控制完全没见过的机器人吗？

论文最强结果是用约 100 条演示微调到新机器人，而不是完全零数据。换动作空间时还需要新增 action head。

### Q8：为什么 goal image 比语言好？

目标图像同时提供对象、位置和最终空间布局，而且全部轨迹都能自动生成；语言只覆盖 56% 数据，内容也不够丰富。

---

## 附录：术语表

- Generalist Robot Policy（GRP）：在多个机器人和任务上预训练、可继续适配的通用策略。
- Goal Image：展示任务目标状态的图像。
- Hindsight Goal Relabeling：从轨迹未来选择一帧，反过来作为当前样本的目标。
- Tokenizer：把图像、语言或其它传感器转换成 Transformer token 的模块。
- Blockwise Attention：按输入类型和时间规定哪些 token 可以互相读取。
- Readout Token：只读取观察信息、再交给输出 head 的汇总 token。
- Action Head：把 readout embedding 转换成机器人动作的轻量模块。
- Diffusion Policy：从随机动作开始，多次去噪得到连续动作序列。
- Action Chunk：一次预测多个未来动作。
- Receding Horizon：只执行动作块前一部分，然后重新观察和规划。
- Delta End-Effector Control：控制末端执行器相对当前姿态移动，而非直接指定整条关节轨迹。
- Zero-shot：不使用当前评估任务的额外微调数据；本文任务技能仍可能已存在于预训练中。

## 阅读说明

- Python 环境可在 PowerShell 中用 `conda activate robotic` 激活。
- ARS `pdf_read_preflight.py` 结构预检为 **PASS**：17/17 页，warnings 为空；SHA-256 为 `73bff297cfafe523319162124e6b7f96919c0930e0a380f307255c4c7464ac93`。
- 已检查正文、Discussion、Appendix C–F、Figure 2/4/5/6/7 和 Table I–VII。
- 本报告由 AI 辅助完成全文抽取、图表核对和结构化改写。结论来自作者论文；【我的分析】或机制解释不等于独立复现。论文实验由作者团队报告，且 arXiv 版本不能自动视为第三方验证。


---

[阅读论文 PDF](../assets/papers/octo.pdf) · [返回论文阅读](index.md)
