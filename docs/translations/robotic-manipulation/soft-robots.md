---
title: 第 12 章 · 软体机器人与触觉感知
description: MIT《机器人操作：感知、规划与控制》中文翻译：第 12 章 · 软体机器人与触觉感知。
---

<div class="reader-kicker">ROBOTIC MANIPULATION / 中文译稿 / 12</div>

# 第 12 章　软体机器人与触觉感知

*Robotic Manipulation: Perception, Planning, and Control*

<div class="translation-meta" markdown>
**原著作者** [Russ Tedrake](http://people.csail.mit.edu/russt/)

**原文最后修改日期** 2026 年 10 月 11 日

**原文版权** © Russ Tedrake, 2020–2026

**对照阅读** [原文 PDF](pdfs/soft-robots-original.pdf) · [在线原文](https://manipulation.mit.edu/tactile.html)
</div>

## 本章目录

- [12.1　为什么要采用软体结构？](#section1)
- [12.2　软体机器人硬件](#section2)
- [12.3　软体仿真](#section3)
- [12.4　触觉感知](#section4)
    - [12.4.1　我们希望或需要获得哪些信息？](#section4.1)
    - [12.4.2　视触觉感知](#section4.2)
    - [12.4.3　全身感知](#section4.3)
    - [12.4.4　触觉传感器仿真](#section4.4)
- [12.5　利用触觉传感器进行感知](#section5)
- [12.6　利用触觉传感器进行控制](#section6)
- [参考文献](#references)

---

到目前为止，我们主要依靠相机和本体感知，例如关节传感器，获取外界信息。还有一种显而易见的感知方式不能忽略：触觉。本章的第一个目标，是介绍触觉传感硬件的近期进展，以及如何利用这些信息的计算方法。对人类操作而言，听觉可能是接下来最重要的感知方式，但这一方向目前还研究得不多。

操作研究的另一个重要趋势，是设计和制造本身就柔软的机器人。操作需要与环境频繁接触，而前面讨论的机器人多数相当刚硬，只在指尖装有橡胶垫之类的柔软材料，因为传统上，我们只希望指尖接触环境。本章的第二个目标，是介绍软体机器人硬件和相应计算框架的进展。

这两个主题看起来可以分开，为什么放在同一章？因为柔软的结构能够支持触觉感知，甚至可以说，丰富的触觉感知离不开它。反过来，对覆盖柔软皮肤的机器人而言，触觉又是本体感知很自然的延伸。因此，这两个主题其实密切相关。

<a id="section1"></a>

## 12.1　为什么要采用软体结构？

<a id="section2"></a>

## 12.2　软体机器人硬件

<a id="section3"></a>

## 12.3　软体仿真

有限元法（FEM）、物质点法（MPM）等。

这段[论文配套视频](https://www.youtube.com/watch?v=yugIJDkWR6M)介绍了 Drake 中高性能、可靠的有限元软体仿真的近期进展，包括软体与刚体的相互作用。[参考文献 1](#Han23)解说中有一句很值得留意：每个时间步都会把接触问题求解到收敛。这与游戏引擎级物理仿真为了速度而采用大量近似的做法，形成了鲜明对比。令人惊讶的是，如今只用一个 CPU 核心，就能以实时速度完成这些计算。

<a id="section4"></a>

## 12.4　触觉感知

<a id="section4.1"></a>

### 12.4.1　我们希望或需要获得哪些信息？

<a id="section4.2"></a>

### 12.4.2　视触觉感知

<a id="section4.3"></a>

### 12.4.3　全身感知

接触估计是支持全身触觉皮肤的一项有力理由。一系列研究已经让我们较好地理解，如何通过关节力矩传感器估计机械臂上发生接触的位置。但这个问题并不适定，尤其在多个接触点同时存在时，仅靠关节力矩测量显然难以获得足够的信息。[参考文献 2](#Pang20a)

<a id="section4.4"></a>

### 12.4.4　触觉传感器仿真

<a id="section5"></a>

## 12.5　利用触觉传感器进行感知

<a id="section6"></a>

## 12.6　利用触觉传感器进行控制

<a id="references"></a>

## 参考文献

<a id="Han23"></a>

1. Xuchen Han and Joseph Masterjohn and Alejandro Castro , "A Convex Formulation of Frictional Contact between Rigid and Deformable Bodies" , arXiv preprint arXiv:2303.08912 , 2023 .

<a id="Pang20a"></a>

2. Tao Pang and Jack Umenberger and Russ Tedrake , "Identifying External Contacts from Joint Torque Measurements on Serial Robotic Arms and Its Limitations" , Under Review. , May, 2021 .
[文献链接](http://groups.csail.mit.edu/robotics-center/public_papers/Pang20a.pdf)

---

**继续阅读**：[上一节（中文）](reinforcement-learning.md) · [全书目录与前言（中文）](preface.md) · [下一节（中文）](spatial-algebra.md)
