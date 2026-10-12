---
title: 全书目录与前言
description: MIT《机器人操作：感知、规划与控制》全书目录与前言中文翻译。
---

<div class="reader-kicker">ROBOTIC MANIPULATION / 中文译稿 / 00</div>

# 全书目录与前言

*Robotic Manipulation: Perception, Planning, and Control*

<div class="translation-meta" markdown>
**原著作者** [Russ Tedrake](http://people.csail.mit.edu/russt/)

**原文最后修改日期** 2026 年 9 月 13 日

**原文版权** © Russ Tedrake, 2020–2026

**对照阅读** [原文 PDF](pdfs/preface-original.pdf) · [课程讲义首页](https://manipulation.mit.edu/)
</div>

> **说明**：这份讲义用于 [MIT 正在开设的一门课程](http://manipulation.csail.mit.edu/Fall2026/)（MIT 6.4210/6.4212），内容还在不断完善，并将在整个 2026 年秋季学期持续更新。

[如何引用讲义、使用批注和提供反馈](https://manipulation.mit.edu/misc.html)

---

## 全书目录 { #all-chapters }

**[前言](#preface)**

### 第 1 章　引言

[在线原文](https://manipulation.mit.edu/intro.html) · [中文译文](introduction.md)

- [操作不止于抓取与放置](https://manipulation.mit.edu/intro.html#section1)
- [开放世界中的操作](https://manipulation.mit.edu/intro.html#section2)
- [仿真](https://manipulation.mit.edu/intro.html#section3)
- [这份讲义是交互式的](https://manipulation.mit.edu/intro.html#section4)
- [基于模型的设计与分析](https://manipulation.mit.edu/intro.html#section5)
- [讲义的组织方式](https://manipulation.mit.edu/intro.html#section6)
- [练习](https://manipulation.mit.edu/intro.html#section7)

### 第 2 章　先给你准备一个机器人

[在线原文](https://manipulation.mit.edu/robot.html)

- [机器人描述文件](https://manipulation.mit.edu/robot.html#section1)
- [机械臂](https://manipulation.mit.edu/robot.html#section2)
  - 位置控制机器人
    - 位置控制
    - 补充：带传动机构的连杆动力学
  - 力矩控制机器人
  - 日益丰富的硬件
  - Kuka iiwa 仿真
- [机械手](https://manipulation.mit.edu/robot.html#section3)
  - 灵巧手
  - 简单夹爪
  - 柔性／欠驱动机械手
  - 其他末端执行器
  - 如果你还没见过……
- [传感器](https://manipulation.mit.edu/robot.html#section4)
- [将各部分整合起来](https://manipulation.mit.edu/robot.html#hardware_station)
  - `HardwareStation`
  - `HardwareStationInterface`
  - `HardwareStation` 独立仿真
- [更多 `HardwareStation` 示例](https://manipulation.mit.edu/robot.html#section6)
- [练习](https://manipulation.mit.edu/robot.html#section7)

### 第 3 章　基本的抓取与放置

[在线原文](https://manipulation.mit.edu/pick.html)

- [Monogram 记号法](https://manipulation.mit.edu/pick.html#monogram)
- [通过空间变换实现抓取与放置](https://manipulation.mit.edu/pick.html#section2)
- [空间代数](https://manipulation.mit.edu/pick.html#spatial_algebra)
  - 三维旋转的表示方法
- [正运动学](https://manipulation.mit.edu/pick.html#kinematics)
  - 运动学树
  - 抓取与放置中的正运动学
- [微分运动学（雅可比矩阵）](https://manipulation.mit.edu/pick.html#jacobian)
- [微分逆运动学](https://manipulation.mit.edu/pick.html#section6)
  - 雅可比矩阵的伪逆
  - 雅可比矩阵的可逆性
- [定义抓取位姿与预抓取位姿](https://manipulation.mit.edu/pick.html#section7)
- [抓取与放置轨迹](https://manipulation.mit.edu/pick.html#pick_and_place_trajectory)
- [将各部分整合起来](https://manipulation.mit.edu/pick.html#section9)
- [带约束的微分逆运动学](https://manipulation.mit.edu/pick.html#diff_ik_w_constraints)
  - 将伪逆求解表述为优化问题
  - 添加速度约束
  - 添加位置与加速度约束
  - 关节居中
  - 跟踪期望位姿
  - 其他问题表述方式
- [练习](https://manipulation.mit.edu/pick.html#section11)

### 第 4 章　基于几何的位姿估计

[在线原文](https://manipulation.mit.edu/pose.html)

- [相机与深度传感器](https://manipulation.mit.edu/pose.html#section1)
  - 深度传感器
  - 仿真
- [几何的表示方法](https://manipulation.mit.edu/pose.html#section2)
- [已知对应关系的点云配准](https://manipulation.mit.edu/pose.html#registration)
- [迭代最近点算法（ICP）](https://manipulation.mit.edu/pose.html#icp)
- [处理局部视图与离群点](https://manipulation.mit.edu/pose.html#section5)
  - 检测离群点
  - 点云分割
  - 对应关系的推广
  - 软对应关系
  - 非线性优化
    - 预计算距离函数
  - 全局优化
- [非穿透约束与“自由空间”约束](https://manipulation.mit.edu/pose.html#section6)
  - 将自由空间约束表述为非穿透约束
- [跟踪](https://manipulation.mit.edu/pose.html#section7)
- [将各部分整合起来](https://manipulation.mit.edu/pose.html#section8)
- [展望](https://manipulation.mit.edu/pose.html#section9)
- [练习](https://manipulation.mit.edu/pose.html#exercises)

### 第 5 章　料箱抓取

[在线原文](https://manipulation.mit.edu/clutter.html)

- [生成随机杂乱场景](https://manipulation.mit.edu/clutter.html#section1)
  - 下落的物体
- [带摩擦接触的静力平衡](https://manipulation.mit.edu/clutter.html#section2)
  - 空间力
  - 碰撞几何
  - 碰撞物体之间的接触力
  - 接触坐标系
  - （库仑）摩擦锥
  - 将静力平衡表述为优化问题
- [接触仿真](https://manipulation.mit.edu/clutter.html#section3)
- [基于模型的抓取选择](https://manipulation.mit.edu/clutter.html#section4)
  - 接触力旋量锥（contact wrench cone）
  - 共线对跖抓取（colinear antipodal grasps）
- [从点云中选择抓取](https://manipulation.mit.edu/clutter.html#section5)
  - 点云预处理
  - 估计法向量与局部曲率
  - 评估候选抓取
  - 生成候选抓取
- [特殊边界情况](https://manipulation.mit.edu/clutter.html#section6)
- [任务层编程](https://manipulation.mit.edu/clutter.html#task)
  - 状态机与行为树
  - 任务规划
  - 大语言模型
  - 用于“清理杂乱物体”的简单状态机
- [将各部分整合起来](https://manipulation.mit.edu/clutter.html#section8)
- [练习](https://manipulation.mit.edu/clutter.html#exercises)

### 第 6 章　运动规划

[在线原文](https://manipulation.mit.edu/trajectories.html)

- [逆运动学](https://manipulation.mit.edu/trajectories.html#section1)
  - 从末端执行器位姿到关节角
  - 将逆运动学表述为带约束的优化问题
  - 全局逆运动学
  - 逆运动学与微分逆运动学的比较
  - 使用逆运动学进行抓取规划
- [运动学轨迹优化](https://manipulation.mit.edu/trajectories.html#section2)
  - 轨迹参数化
  - 优化算法
- [基于采样的运动规划](https://manipulation.mit.edu/trajectories.html#section3)
  - 快速探索随机树（RRT）
  - 概率路线图（PRM）
  - 后处理
  - 基于采样的规划在实践中的应用
- [基于凸集图（GCS）的运动规划](https://manipulation.mit.edu/trajectories.html#gcs)
  - 凸集图
  - GCS（运动学）轨迹优化
  - （无碰撞）构型空间的凸分解
  - `GcsTrajOpt` 示例
  - 变体与扩展
- [时间最优的路径参数化](https://manipulation.mit.edu/trajectories.html#topp)
- [练习](https://manipulation.mit.edu/trajectories.html#section6)

### 第 7 章　移动操作

[在线原文](https://manipulation.mit.edu/mobile.html)

- [新登场的机器人](https://manipulation.mit.edu/mobile.html#section1)
- [感知有什么不同？](https://manipulation.mit.edu/mobile.html#section2)
  - 局部视图／主动感知
  - 未知环境（可能是动态的）
  - 机器人状态估计
- [运动规划有什么不同？](https://manipulation.mit.edu/mobile.html#section3)
  - 轮式机器人
    - 完整约束驱动（holonomic drives）
    - 非完整约束驱动（nonholonomic drives）
  - 腿式机器人
- [仿真有什么不同？](https://manipulation.mit.edu/mobile.html#section4)
- [导航](https://manipulation.mit.edu/mobile.html#section5)
  - 建图（除定位之外）
  - 识别可通行地形
- [练习](https://manipulation.mit.edu/mobile.html#section6)

### 第 8 章　机械臂控制

[在线原文](https://manipulation.mit.edu/force.html)

- [机械臂控制工具箱](https://manipulation.mit.edu/force.html#toolbox)
- [假设机器人是一个质点](https://manipulation.mit.edu/force.html#section2)
  - 轨迹跟踪
  - （直接）力控制
  - 间接力控制
  - 位置／力混合控制
- [一般情况（使用机械臂动力学方程）](https://manipulation.mit.edu/force.html#section3)
  - 轨迹跟踪
  - 关节刚度控制
  - 笛卡尔空间刚度控制与操作空间控制
  - iiwa 上的一些实现细节
- [将各部分整合起来](https://manipulation.mit.edu/force.html#section4)
- [轴孔装配](https://manipulation.mit.edu/force.html#section5)
- [练习](https://manipulation.mit.edu/force.html#section6)

### 第 9 章　物体检测与分割

[在线原文](https://manipulation.mit.edu/segmentation.html)

- [迈向大数据](https://manipulation.mit.edu/segmentation.html#section1)
  - 众包标注数据集
  - 通过微调分割新类别
  - 面向操作任务的标注工具
  - 合成数据集
  - 自监督学习
  - 更大规模的数据集
- [物体检测与分割](https://manipulation.mit.edu/segmentation.html#section2)
- [将各部分整合起来](https://manipulation.mit.edu/segmentation.html#section3)
- [变体与扩展](https://manipulation.mit.edu/segmentation.html#section4)
  - 使用自监督学习进行预训练
  - 利用大规模模型
- [练习](https://manipulation.mit.edu/segmentation.html#section5)

### 第 10 章　面向操作任务的深度感知

[在线原文](https://manipulation.mit.edu/deep_perception.html)

- [位姿估计](https://manipulation.mit.edu/deep_perception.html#section1)
  - 位姿表示
  - 损失函数
  - 位姿估计基准
  - 局限性
- [抓取选择](https://manipulation.mit.edu/deep_perception.html#section2)
- [（语义）关键点](https://manipulation.mit.edu/deep_perception.html#section3)
- [稠密对应关系](https://manipulation.mit.edu/deep_perception.html#section4)
- [场景流](https://manipulation.mit.edu/deep_perception.html#section5)
- [任务层状态](https://manipulation.mit.edu/deep_perception.html#section6)
- [其他感知任务／表示方法](https://manipulation.mit.edu/deep_perception.html#section7)
- [练习](https://manipulation.mit.edu/deep_perception.html#exercises)

### 第 11 章　强化学习

[在线原文](https://manipulation.mit.edu/rl.html)

- [强化学习软件](https://manipulation.mit.edu/rl.html#section1)
- [策略梯度方法](https://manipulation.mit.edu/rl.html#section2)
  - 黑箱优化
  - 随机最优控制
  - 使用策略的梯度，而不使用环境的梯度
  - REINFORCE、PPO、TRPO
  - 操作任务的控制本应容易
- [基于价值的方法](https://manipulation.mit.edu/rl.html#section3)
- [基于模型的强化学习](https://manipulation.mit.edu/rl.html#section4)
- [练习](https://manipulation.mit.edu/rl.html#section5)

### 第 12 章　软体机器人与触觉感知

[在线原文](https://manipulation.mit.edu/tactile.html)

- [为什么要采用软体结构？](https://manipulation.mit.edu/tactile.html#section1)
- [软体机器人硬件](https://manipulation.mit.edu/tactile.html#section2)
- [软体仿真](https://manipulation.mit.edu/tactile.html#section3)
- [触觉感知](https://manipulation.mit.edu/tactile.html#section4)
  - 我们希望／需要获得哪些信息？
  - 视触觉感知
  - 全身感知
  - 触觉传感器仿真
- [利用触觉传感器进行感知](https://manipulation.mit.edu/tactile.html#section5)
- [利用触觉传感器进行控制](https://manipulation.mit.edu/tactile.html#section6)

### 附录 A　空间代数

[在线原文](https://manipulation.mit.edu/spatial.html)

- [位置、旋转与位姿](https://manipulation.mit.edu/spatial.html#section1)
- [空间速度](https://manipulation.mit.edu/spatial.html#section2)
- [空间力](https://manipulation.mit.edu/spatial.html#section3)

### 附录 B　Drake

[在线原文](https://manipulation.mit.edu/drake.html)

- [Pydrake](https://manipulation.mit.edu/drake.html#section1)
- [在线 Jupyter Notebook](https://manipulation.mit.edu/drake.html#notebooks)
  - 在 Google Colab 上运行
  - 启用需要许可证的求解器
- [在自己的计算机上运行](https://manipulation.mit.edu/drake.html#section3)
- [获取帮助](https://manipulation.mit.edu/drake.html#section4)

### 附录 C　DrakeGym 环境

[在线原文](https://manipulation.mit.edu/environments.html)

- [翻起箱体（Box Flipup）](https://manipulation.mit.edu/environments.html#section1)

### 附录 D　搭建自己的“机器人操作工作站”

[在线原文](https://manipulation.mit.edu/station.html)

- [消息传递](https://manipulation.mit.edu/station.html#section1)
- [Kuka LBR iiwa + Schunk WSG 夹爪](https://manipulation.mit.edu/station.html#section2)
- [Franka Panda](https://manipulation.mit.edu/station.html#section3)
- [Intel RealSense D415 深度相机](https://manipulation.mit.edu/station.html#section4)

### 附录 E　其他事项

[在线原文](https://manipulation.mit.edu/misc.html)

- [如何引用这些讲义](https://manipulation.mit.edu/misc.html#cite)
- [批注工具的使用礼仪](https://manipulation.mit.edu/misc.html#annotation)
- [一些优秀的期末项目](https://manipulation.mit.edu/misc.html#projects)
- [欢迎向我提供反馈！](https://manipulation.mit.edu/misc.html#feedback)

讲义配套代码的文档可以在[这里](https://manipulation.mit.edu/python/index.html)查阅。

---

<a id="preface"></a>

## 前言

我一直喜欢机器人学，只是最近才把目光转向机器人操作。尤其吸引我的，是这样一个挑战：让机器人掌握物理规律，获得像人类或动物那样的灵巧性与敏捷性。最早让我认识到动力学有多重要的，是[被动动态行走机器人](http://underactuated.mit.edu/intro.html)及其背后优美的分析。从那时起，无论是理解世界，还是研究机器人，我都把动力学放在核心位置。

此后，我又迷上了实验流体动力学。鸟类靠翅膀的灵活运动来“操作”空气，因而能够飞得既高效又敏捷。这个想法让我十分着迷。而人形机器人，以及在杂乱环境中高速穿行的飞行器，又促使我更深入地思考感知在动力学与控制中的作用。

现在，我愈发相信，感知与动力学之间的相互作用，是理解机器人操作的关键。尤其有意思的是，一些看似“简单”的任务，比如扣好衬衫上的纽扣，就能把这个问题展现得十分清楚。

给机器人编程时，我一直以计算和算法为主要手段。最初，我主要用机器学习，尤其是强化学习，为简单的步行机器人设计控制系统。随着机器人和任务越来越复杂，我逐渐转向更成熟的基于模型的规划方法，以及基于优化的控制方法。

在我看来，没有哪个学科对动力学的研究比控制理论更深入。最好的基于模型的控制算法，无论是算法效率，还是对性能与鲁棒性的保证，都远远超过了目前学习控制所能达到的水平。

遗憾的是，数学理论的成熟，也让控制领域在研究假设和问题表述上相对保守，而机器人操作恰恰会打破这些假设。比如，鲁棒控制通常假定动力学是光滑或近乎光滑的，不确定性可以用简单的分布或集合来表示。但在操作任务中，接触过程的力学并不光滑；光照的变化、物体数量的不同，以及未知的几何形状和动力学特性，也都会带来不确定性。

据我所知，即使是目前最先进的机器人操作系统，也还没有用严格的控制理论来设计决定机器人何时与物体接触、何时分离的底层反馈控制。这份讲义的一个明确目标，就是尝试改变这一状况。

过去几年，深度学习显著推动了机器人感知的发展，使机器人在实验室和工厂之外执行操作任务时，突破了一些最棘手的障碍。后文会介绍物体识别、分割、位姿与关键点估计、形状补全等任务中用到的深度学习工具。

与此同时，一些较早的学习控制方法也重新受到广泛关注。这在一定程度上得益于强大的计算能力，以及日益普及的机器人硬件和仿真器。不过，与感知领域的学习方法相比，学习控制算法距离成熟应用还有很长的路要走。一些看起来非常出色的结果，至今仍难以理解和复现。

尽管如此，近期的研究也确实让我们看到了控制领域保守态度的局限。学习领域的研究者正在大胆尝试，为机器人操作提出比以往更有挑战性、也更令人兴奋的问题。有些任务，我们尝试之后才发现其实相当容易；另一些任务，则仍然存在根本性的困难。

最后，我觉得，从物流到家用机器人，机器人操作真正开始改变现实世界的时机已经成熟。过去几年，无人机从学术研究对象变成了消费产品。不久前，自动驾驶也从学术研究走向产业，至少从资金投入来看是如此。机器人操作似乎将成为下一个从研究走向实际应用的重要领域。

对风险投资者而言，现在投资这项技术仍有风险。但一旦技术成熟，市场会有多大，大家并不怀疑。能有机会参与这样的转变，我们实在很幸运。

这就是这份讲义的起点。如今，学习、控制与机器人技术正在交汇，我们既有机会直接推动工业和消费领域的应用，也可能为系统理论与控制开辟一个新的时代。我希望自己能跟上这一进程，也享受这段旅程。

<a id="toolbox"></a>

## 机器人操作工具箱

这份讲义还有一个明确目标：为研究机器人操作时最有用的工具，提供高质量的软件实现。

如果数学表述的清晰程度与运行性能难以兼顾，我会优先保证表述清楚。在条件允许时，我也会尽量给出性能较好的实现，或介绍其他可选方案。机器人操作研究进展很快，这份讲义也会随之不断更新。我希望 Drake 和讲义中的软件组件，能直接帮到你自己的工作。

如果你想搭建讲义中用到的部分或全部硬件，可以查阅[附录](https://manipulation.mit.edu/station.html)中的资料和说明。

使用这些代码时，也欢迎[为项目作出贡献](https://drake.mit.edu/getting_help.html)，尤其是帮助完善 Drake 中已经成熟的代码。提出问题、报告 bug，同样可能是重要的贡献。如果对讲义有疑问，或发现了问题，可以在[这里提交](https://github.com/RussTedrake/manipulation/issues)。

---

**继续阅读**：[第 1 章——引言（中文译文）](introduction.md) · [第 1 章在线原文](https://manipulation.mit.edu/intro.html)
