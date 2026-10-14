---
title: 第 11 章 · 强化学习
description: MIT《机器人操作》中文译稿。连接随机优化、策略梯度、基于价值与模型的方法，理解操作任务中的强化学习。
tags:
- 机器人操作
- MIT 讲义
layout: article
updated: '2026-10-13'
updated_source: git:c0a5c26daf5d9a0bb25c5620f4c002497f69f678
---

<div class="reader-kicker">ROBOTIC MANIPULATION / 中文译稿 / 11</div>

# 第 11 章　强化学习

*Robotic Manipulation: Perception, Planning, and Control*

<div class="translation-meta" markdown>
**原著作者** [Russ Tedrake](http://people.csail.mit.edu/russt/)

**原文最后修改日期** 2026 年 10 月 11 日

**原文版权** © Russ Tedrake, 2020–2026

**对照阅读** [原文 PDF](pdfs/reinforcement-learning-original.pdf) · [在线原文](https://manipulation.mit.edu/rl.html)
</div>

## 本章目录

- [11.1　强化学习软件](#section1)
- [11.2　策略梯度方法](#section2)
    - [11.2.1　黑箱优化](#section2.1)
    - [11.2.2　随机最优控制](#section2.2)
    - [11.2.3　使用策略的梯度，而不使用环境的梯度](#section2.3)
    - [11.2.4　REINFORCE、PPO 与 TRPO](#section2.4)
    - [11.2.5　操作任务的控制本应容易](#section2.5)
- [11.3　基于价值的方法](#section3)
- [11.4　基于模型的强化学习](#section4)
- [11.5　练习](#section5)
- [参考文献](#references)

---

如今，强化学习（RL）备受关注，相关文献也非常丰富，被归入强化学习的算法范围也扩大了不少。Sutton 和 Barto 的经典著作已经更新，仍是我认为最好的入门书。[参考文献 1](#Sutton18)如果希望从更理论的角度学习，可以读[参考文献 2](#Agarwal20b)、[参考文献 3](#Szepesvari10)。此外，还有一些很好的在线课程：[Stanford CS234](http://web.stanford.edu/class/cs234/)、[Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/)和[DeepMind 与 UCL 的课程](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver)。

本章先介绍必要的基础，让大家有共同的认识，再重点讨论与机器人操作有关的强化学习思路和示例。操作任务既需要丰富的感知，又要适应多样环境，往往还涉及复杂接触，因此很适合研究强化学习。许多重要的应用研究，也正是由操作问题推动，并在操作任务中验证的。

<a id="section1"></a>

## 11.1　强化学习软件

网上已经有大量强化学习工具箱，质量和完善程度差别很大。不过，连接仿真器与不同强化学习算法时，已有一种接口成为事实上的标准：[Gymnasium](https://gymnasium.farama.org/)，人们也常把它简称为 Gym。

值得比较一下 Gymnasium 的[环境](https://gymnasium.farama.org/api/env/)接口，即 `gymnasium.Env`，与 Drake 的 System 接口有什么不同。在 Drake 中，我希望用丰富而清晰的接口表达和优化动力系统，充分展示并利用支配方程的结构。Gym 的目标则是暴露尽可能少的细节，让机器人、Atari 游戏，甚至[编译器](https://github.com/facebookresearch/CompilerGym)都能封装进同一种接口。几乎任何 Drake 系统都可以封装成 Gym 环境。

<a id="drake_gym"></a>

<a id="example1"></a>

### 示例 11.1　把 Drake 仿真封装成 Gym 环境

Gym 环境只是仿真器外面的一层简单封装，提供[基本接口](https://github.com/openai/gym/blob/6a04d49722724677610e36c1f92908e72f51da0c/gym/core.py#L35)，主要包括 `reset()`、`step()` 和 `render()`。其中，`step()` 方法返回当前观测、单步奖励和一些终止条件。

使用下面的导入语句，就能把任意 Drake 仿真封装成 OpenAI Gym 环境：

```python
from pydrake.gym import DrakeGymEnv
```


 [`DrakeGymEnv` 构造函数](https://drake.mit.edu/pydrake/pydrake.gym.html?highlight=drakegym#pydrake.gym.DrakeGymEnv)接收一个 `Simulator`，以及与动作对应的输入端口、与观测对应的输出端口等信息。奖励既可以写成依赖 `Simulator` 的 `Context` 的简单函数，也可以通过另一个输出端口给出。

`DrakeGymEnv`接收的是 `Simulator`，而不只是 `System`；也可以接收一个生成随机 `Simulator` 的函数。这样，你就能控制积分器参数，或在每次 rollout 中为同一机器人生成不同环境，甚至改变物体数量。底层 `System` 的状态和端口数量也可能随之变化。能够生成仿真器的函数称为 `SimulatorFactory`，是 Drake [随机系统建模框架](https://drake.mit.edu/doxygen_cxx/group__stochastic__systems.html)的核心概念。

反过来，也可以把任意 Gym 环境接入 Drake，只是某些更高级的算法就无法使用了。我当然也建议做强化学习时使用 Drake，很多人已经这样做：它提供经过严格编写和测试的动力系统库，以及擅长处理接触的物理引擎，还方便把强化学习方法与基于模型的方法直接比较。我承认自己可能有些偏爱，但本讲义确实会沿用这条路线。

有人[可能认为](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)，模型越细致，隐含假设就越多，也越容易受到仿真到现实差距的影响。我不这样认为。认真建模也包括建立不确定性模型，让它覆盖我们希望研究的较窄或较广的一类系统；不确定性模型本身有结构时，还能带来更多好处。我认为，强化学习与控制交汇处的一项根本挑战，是找到足够丰富的模型，既能描述操作任务的多样性，比如以 RGB 相机为输入，又保留可供更强算法利用的结构。这些模型还应随着数据不断扩展和改进。

 

Gymnasium 提供环境接口，并不实现强化学习算法。算法本身也有很多热门代码库。在撰写讲义时，我推荐 [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/index.html)，它提供了一组基于 PyTorch 的实现，设计和文档都很用心。

还有一类与强化学习密切相关、但并非专为它设计的算法：黑箱优化。我很喜欢 [Nevergrad](https://facebookresearch.github.io/nevergrad/)，本章也会使用它。

<a id="section2"></a>

## 11.2　策略梯度方法

<a id="section2.1"></a>

### 11.2.1　黑箱优化

<a id="nevergrad"></a>

<a id="example2"></a>

#### 示例 11.2　黑箱优化

**运行示例**：[在 Colab 中打开](https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/book/rl/black_box.ipynb)

<a id="section2.2"></a>

### 11.2.2　随机最优控制

<a id="section2.3"></a>

### 11.2.3　使用策略的梯度，而不使用环境的梯度

有关这些算法的推导与基本分析，可以在[这里](http://underactuated.csail.mit.edu/rl_policy_search.html)找到更多说明。

<a id="section2.4"></a>

### 11.2.4　REINFORCE、PPO 与 TRPO

<a id="box_flipup"></a>

<a id="example3"></a>

#### 示例 11.3　翻起箱体

**运行示例**：[在 Colab 中打开](https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/book/rl/box_flipup.ipynb)

<a id="section2.5"></a>

### 11.2.5　操作任务的控制本应容易

如今，理论强化学习与控制的交叉研究正处在很好的时期，两个领域都在吸收对方的新技术和认识。一个简单例子是，我们逐渐理解：许多经典控制问题，比如线性二次调节器，其代价函数对常用策略参数并不凸，但[梯度下降仍然有效](http://underactuated.csail.mit.edu/policy_search.html)，因为不存在局部极小值。能够得到这类结论的问题和参数化方式，也在迅速增加。

<a id="section3"></a>

## 11.3　基于价值的方法

<a id="section4"></a>

## 11.4　基于模型的强化学习

<a id="section5"></a>

## 11.5　练习

<a id="exercise-1"></a>

<a id="exercise1"></a>

### 练习 11.1　随机优化

本练习需要实现一种不依赖精确解析梯度的随机优化方法。所有操作都在[对应 notebook](https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/book/rl/exercises/stochastic_optimization.ipynb)中完成，具体步骤如下：

1. 使用精确的解析梯度实现梯度下降。

2. 实现带有正确方差缩放的高斯零阶梯度估计，并比较平滑作用与采样噪声。

3. 证明估计量的目标是高斯平滑后的损失，并证明加入基线不会改变其期望。

4. 实现带基线的随机梯度下降。

<a id="exercise-2"></a>

<a id="exercise2"></a>

### 练习 11.2　REINFORCE

本练习在推箱任务上实现最基本的 REINFORCE 算法。所有操作都在[对应 notebook](https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/book/rl/exercises/policy_gradient.ipynb)中完成，具体步骤如下：

1. 实现策略损失函数。

2. 实现价值损失函数。

3. 实现优势函数。

<a id="rl-box-flipup"></a>

<a id="exercise3"></a>

### 练习 11.3　分析强化学习翻箱策略

本练习分析一个训练用来翻箱的 [PPO](https://arxiv.org/abs/1707.06347) 策略。与 REINFORCE 一样，PPO 是直接优化策略参数的策略梯度方法，目标是最大化价值函数。为便于分析，我们采用第 8 章的[翻箱示例](manipulator-control.md#force_flip_up)：机器人只是一个点状手指，目标是翻起箱体。训练代码可以在[这里](https://github.com/RussTedrake/manipulation/blob/master/book/rl/train_boxflipup.py)查看。

1. 查看生成环境所用的[代码](https://github.com/RussTedrake/manipulation/blob/master/manipulation/envs/box_flipup.py)。令 $\theta$ 表示箱体相对竖直方向的角度，$\omega$ 为角速度，$q_f$ 为观测到的手指位置，$v_f$ 为手指速度，$u_f$ 为手指的位置指令。这里训练策略采用了什么奖励函数？请写出数学表达式，用取模运算处理角度环绕。奖励中的各项分别代表什么，为什么这样设计合理？

2. 这里不展开 PPO 的全部细节。它与 REINFORCE 相似，但利用学习到的价值函数降低方差，还采用近似目标，并通过裁剪逐样本损失施加信赖域约束，避免每一步更新过大。请简要说明：为什么 PPO 可能比 REINFORCE 更稳定、样本效率更高？翻箱任务中，如果裁剪范围太小或太大，你预期会有什么表现？

3. 我们用 PPO 训练了 3,000,000 步，[这里](https://youtube.com/playlist?list=PLOZK7fx6sI6lvDalINByA_kbYfBxn76by&si=XUndO5UDHMJcywr2)提供了不同训练阶段的策略视频。随着训练步数增加，策略有什么变化？请定性描述它随时间的变化，以及各阶段奖励函数中哪些项影响最大。

请留意，即使是二维点状手指翻箱这样简单的操作任务，而且已有稠密奖励，训练出可用策略仍需要不少时间。更困难的抓取与放置任务，直接采用强化学习可能尤其难以训练，特别是只有在物体被拿起或正确放置时才给奖励的稀疏奖励任务。另一方面，强化学习又能在接触丰富的场景中表现很好，翻箱就是一例；[用强化学习让单手解魔方的项目](https://www.youtube.com/watch?v=x4O8pojMF0w)也展示了这样的能力，不过它还大量依赖域随机化、课程学习和大规模计算。相比之下，运动控制领域的情况似乎很不一样，也许因为更容易设计稠密奖励，并在仿真中自动重置。

<a id="references"></a>

## 参考文献

<div class="references" markdown="1">

1. <a id="Sutton18"></a> Richard S. Sutton and Andrew G. Barto , "Reinforcement Learning: An Introduction" , MIT Press
    , 2018 .

2. <a id="Agarwal20b"></a> Alekh Agarwal and Nan Jiang and Sham M. Kakade and Wen Sun , "Reinforcement Learning: Theory and Algorithms" , Online Draft
    , 2020 .

3. <a id="Szepesvari10"></a> Csaba Szepesvari , "Algorithms for Reinforcement Learning" , Morgan and Claypool Publishers
    , 2010 .

</div>

---

**继续阅读**：[上一章：面向操作任务的深度感知（中文）](deep-perception.md) · [全书目录与前言（中文）](preface.md) · [下一章：软体机器人与触觉感知（中文）](soft-robots.md)  
[无障碍访问](https://accessibility.mit.edu/) · © Russ Tedrake, 2026
