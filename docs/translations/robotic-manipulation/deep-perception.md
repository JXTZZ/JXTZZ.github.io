---
title: 第 10 章 · 面向操作任务的深度感知
description: MIT《机器人操作》中文译稿。讨论面向操作任务的深度感知，包括位姿、抓取、三维表示与视觉特征学习。
tags:
- 机器人操作
- MIT 讲义
layout: article
updated: '2026-10-13'
updated_source: git:c0a5c26daf5d9a0bb25c5620f4c002497f69f678
---

<div class="reader-kicker">ROBOTIC MANIPULATION / 中文译稿 / 10</div>

# 第 10 章　面向操作任务的深度感知

*Robotic Manipulation: Perception, Planning, and Control*

<div class="translation-meta" markdown>
**原著作者** [Russ Tedrake](http://people.csail.mit.edu/russt/)

**原文最后修改日期** 2026 年 10 月 11 日

**原文版权** © Russ Tedrake, 2020–2026

**对照阅读** [原文 PDF](pdfs/deep-perception-original.pdf) · [在线原文](https://manipulation.mit.edu/deep_perception.html)
</div>

## 本章目录

- [10.1　位姿估计](#section1)
    - [10.1.1　位姿表示](#section1.1)
    - [10.1.2　损失函数](#section1.2)
    - [10.1.3　位姿估计基准](#section1.3)
    - [10.1.4　局限性](#section1.4)
- [10.2　抓取选择](#section2)
- [10.3　语义关键点](#section3)
- [10.4　稠密对应关系](#section4)
- [10.5　场景流](#section5)
- [10.6　任务层状态](#section6)
- [10.7　其他感知任务与表示方法](#section7)
- [10.8　练习](#exercises)
- [参考文献](#references)

---

上一章介绍了基于深度学习的物体检测和实例分割。这些都是计算机视觉中处理 RGB 图像的通用任务。将检测、分割与几何感知结合，就可以只在分割出的点云上估计已知物体的位姿，而不必处理整个场景；也可以只对目标物体的点云运行抓取选择算法，把感兴趣的物体拿起来。

深度学习用于感知时，一个很强的优势是：可以先在 ImageNet、COCO 等其他数据集上，甚至在其他任务上预训练，再用自己领域的数据或任务微调。不过，机器人操作究竟需要哪些感知任务？检测和分割是很好的起点，但要真正操作物体，我们往往还需要了解更多。本章就来讨论这个问题。

一种可能的答案是直接学习端到端的视觉运动策略，也常被形象地称为“从像素到力矩”。这个方向留到后面的章节。这里先考虑，怎样把基于深度学习的感知系统，与前面建立起来的基于模型的规划、控制工具结合。

我们先从一个熟悉任务的深度学习版本入手：物体位姿估计。

<a id="section1"></a>

## 10.1　位姿估计

在[几何感知一章](pose-estimation.md)中，我们已经较详细地讨论过位姿估计，也得到了一些认识。最重要的是，几何方法很难充分利用 RGB 信息，而颜色往往对确定物体位姿非常有用，几何本身不能说明全部情况。另一个细节是，ICP 的损失虽然概念清楚，却不能充分体现非穿透、自由空间等更丰富的约束。随着二维视觉的一些核心问题逐渐显得已经“解决”，计算机视觉领域对三维感知的兴趣迅速增加，这对机器人研究是好事。

最简单的问题是：从单张 RGB 图像估计一个已知物体的位姿。例如，怎样训练一个专门识别芥末瓶的网络，让它输入 RGB 图像，输出位姿估计？如果能做到，自然也可以对检测或实例分割系统输出的包围框进行裁剪，再估计其中物体的位姿。

<a id="section1.1"></a>

### 10.1.1　位姿表示

我们又要面对位姿如何表示的问题。早期不少网络把位姿空间离散成若干区间，将位姿估计视为分类；后来则逐渐转向位姿回归。[参考文献 1](#Mahendran17)、[参考文献 2](#Xiang17)用三个数回归 x、y、z 平移很直观，但三维朝向可以用欧拉角、单位四元数、旋转矩阵等多种方式表示，选择不同，学习与泛化的效果也会不同。

对于输出单个位姿的网络，[参考文献 3](#Zhou19)和[参考文献 4](#Levinson20)指出，许多旋转参数化方式存在不连续性。他们建议用六个数表示旋转，再通过 Gram–Schmidt 正交化投影到 SO(3)；或者直接输出旋转矩阵的九个数，再通过[SVD 正交化](pose-estimation.md#registration)投影回 SO(3)。

更根本的问题是，许多研究认为只输出一个“正确”位姿并不够。[参考文献 5](#Hashimoto20)、[参考文献 6](#Deng22)物体可能具有旋转对称性，也可能被严重遮挡；这时，输出整个位姿分布更加合适。离散表示很容易配合类别分布，但连续位姿的分布该怎样表示？一种优美的选择是 Bingham 分布，它相当于把高斯分布的思想推广到单位四元数。[参考文献 7](#Glover14)、[参考文献 8](#Peretroukhin20)、[参考文献 6](#Deng22)

<a id="section1.2"></a>

### 10.1.2　损失函数

不管网络输出和真值标签采用哪种位姿表示，都需要合适的损失函数。基于四元数的损失可以计算两个朝向之间的测地距离，通常比欧拉角上的最小二乘误差更合适。另一种计算成本更高、但可能更合理的办法，是用重建误差定义损失，避免网络因无法消除的物体对称性而受到不合理惩罚。[参考文献 9](#Hodan20)

如果网络输出整个位姿分布，损失的选择又会带来新的问题。既可以根据标有真实位姿的数据统计来训练，比如选择最大似然损失而非均方误差；也可以利用物体的对称性提供更直接的监督。例如，[参考文献 5](#Hashimoto20)利用图像差异，高效但带有启发式地估计每个样本的真值协方差。

<a id="section1.3"></a>

### 10.1.3　位姿估计基准

六维物体位姿估计基准（BOP）。[参考文献 9](#Hodan20)

<a id="section1.4"></a>

### 10.1.4　局限性

位姿估计很自然，估计结果也容易接入许多机器人流程，但我一直认为，它往往不是连接感知与规划、控制的最佳方式。虽然已有研究尝试把位姿推广到整个物体类别，[参考文献 10](#Wang19)位姿估计仍然很依赖已知物体，适用范围因此受到限制。准确估计位姿既困难，对许多操作任务又并非必需。

<a id="section2"></a>

## 10.2　抓取选择

J. J. Gibson 在[参考文献 11](#Gibson77)、[参考文献 12](#Gibson79)中提出了影响深远的可供性（affordance）理论，强调感知应当服务于行为，行为又受感知支配。对机器人而言，可供性关心的不是物体或环境“是什么”、处于什么位姿，而是它提供了哪些行动机会。重要的是，我们可能不必显式估计位姿，就能估计这些机会。

较早且成功的两个例子是[参考文献 13](#tenPas17)、[参考文献 14](#tenPas18)和[参考文献 15](#Mahler17)、[参考文献 16](#Mahler17a)，它们尝试直接从原始 RGB-D 输入学习抓取可供性。[参考文献 13](#tenPas17)采用的抓取选择策略，与前面介绍的[抓取采样器](bin-picking.md#grasp_sampler)很相似，后者也部分受它启发。不过，除了几何启发式，[参考文献 13](#tenPas17)还训练了一个小型 CNN 来预测抓取能否成功：输入是抓取附近像素及局部几何的体素化摘要，输出是成功或失败的二分类结果。类似地，[参考文献 15](#Mahler17)学习了一个抓取质量 CNN，输入抓取坐标系中的深度图，以及相机坐标系中的抓取深度，输出成功概率。

Transporter 网络。[参考文献 17](#Zeng21)

<a id="section3"></a>

## 10.3　语义关键点

面向类别级机器人操作的关键点可供性方法 kPAM。[参考文献 18](#Manuelli19)、[参考文献 19](#Gao20b)

<a id="section4"></a>

## 10.4　稠密对应关系

[参考文献 20](#Florence18a)、DynoV2 等。

<a id="section5"></a>

## 10.5　场景流

<a id="section6"></a>

## 10.6　任务层状态

<a id="section7"></a>

## 10.7　其他感知任务与表示方法

前面的介绍不可能面面俱到，而且这个领域进展很快。这里也简要列出一些与操作密切相关的思路。

后续还会补充更多内容。

<a id="exercises"></a>

## 10.8　练习

<a id="don_contrastive"></a>

<a id="exercise1"></a>

### 练习 10.1　稠密物体网络与对比损失

本题进一步探索课堂中介绍的稠密物体网络（Dense Object Nets）。它能很快学到一致的像素级表示，用于视觉理解和操作。这些表示同时适用于刚性和非刚性物体，还能泛化到同类新物体，并通过自监督学习训练。请在[对应 notebook](https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/book/deep_perception/exercises/contrastive.ipynb)中，先实现训练所用的损失函数，再利用训练好的网络预测图像之间的对应关系。

<a id="references"></a>

## 参考文献

<div class="references" markdown="1">

1. <a id="Mahendran17"></a> Siddharth Mahendran and Haider Ali and René Vidal , "3d pose regression using convolutional neural networks" , Proceedings of the IEEE International Conference on Computer Vision Workshops , pp. 2174--2182, 2017 .

2. <a id="Xiang17"></a> Yu Xiang and Tanner Schmidt and Venkatraman Narayanan and Dieter Fox , "Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes" , arXiv preprint arXiv:1711.00199 , 2017 .

3. <a id="Zhou19"></a> Yi Zhou and Connelly Barnes and Jingwan Lu and Jimei Yang and Hao Li , "On the continuity of rotation representations in neural networks" , Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 5745--5753, 2019 .

4. <a id="Levinson20"></a> Jake Levinson and Carlos Esteves and Kefan Chen and Noah Snavely and Angjoo Kanazawa and Afshin Rostamizadeh and Ameesh Makadia , "An analysis of svd for deep rotation estimation" , Advances in Neural Information Processing Systems , vol. 33, pp. 22554--22565, 2020 .

5. <a id="Hashimoto20"></a> Kunimatsu Hashimoto* and Duy-Nguyen Ta* and Eric Cousineau and Russ Tedrake , "KOSNet: A Unified Keypoint, Orientation and Scale Network for Probabilistic 6D Pose Estimation" , Under review , 2020 .
    [文献链接](http://groups.csail.mit.edu/robotics-center/public_papers/Hashimoto20.pdf)

6. <a id="Deng22"></a> Haowen Deng and Mai Bui and Nassir Navab and Leonidas Guibas and Slobodan Ilic and Tolga Birdal , "Deep bingham networks: Dealing with uncertainty and ambiguity in pose estimation" , International Journal of Computer Vision , vol. 130, no. 7, pp. 1627--1654, 2022 .

7. <a id="Glover14"></a> Jared Marshall Glover , "The Quaternion Bingham Distribution, 3D Object Detection, and Dynamic Manipulation" , 
    PhD thesis, Massachusetts Institute of Technology, May, 2014 .

8. <a id="Peretroukhin20"></a> Valentin Peretroukhin and Matthew Giamou and David M Rosen and W Nicholas Greene and Nicholas Roy and Jonathan Kelly , "A smooth representation of belief over so (3) for deep rotation learning with uncertainty" , arXiv preprint arXiv:2006.01031 , 2020 .

9. <a id="Hodan20"></a> Tomás Hodan and Martin Sundermeyer and Bertram Drost and Yann Labbé and Eric Brachmann and Frank Michel and Carsten Rother and Jirí Matas , "BOP Challenge 2020 on 6D Object Localization" , European Conference on Computer Vision Workshops (ECCVW) , 2020 .

10. <a id="Wang19"></a> He Wang and Srinath Sridhar and Jingwei Huang and Julien Valentin and Shuran Song and Leonidas J Guibas , "Normalized object coordinate space for category-level 6d object pose and size estimation" , Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 2642--2651, 2019 .

11. <a id="Gibson77"></a> James J Gibson , "The theory of affordances" , Hilldale, USA , vol. 1, no. 2, pp. 67--82, 1977 .

12. <a id="Gibson79"></a> James J Gibson , "The ecological approach to visual perception" , Psychology press
    , 1979 .

13. <a id="tenPas17"></a> Andreas ten Pas and Marcus Gualtieri and Kate Saenko and Robert Platt , "Grasp pose detection in point clouds" , The International Journal of Robotics Research , vol. 36, no. 13-14, pp. 1455--1473, 2017 .

14. <a id="tenPas18"></a> Andreas Ten Pas and Robert Platt , "Using geometry to detect grasp poses in 3d point clouds" , Robotics Research: Volume 1 , pp. 307--324, 2018 .

15. <a id="Mahler17"></a> Jeffrey Mahler and Jacky Liang and Sherdil Niyaz and Michael Laskey and Richard Doan and Xinyu Liu and Juan Aparicio Ojea and Ken Goldberg , "Dex-net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics" , arXiv preprint arXiv:1703.09312 , 2017 .

16. <a id="Mahler17a"></a> Jeffrey Mahler and Ken Goldberg , "Learning deep policies for robot bin picking by simulating robust grasping sequences" , Conference on robot learning , pp. 515--524, 2017 .

17. <a id="Zeng21"></a> Andy Zeng and Pete Florence and Jonathan Tompson and Stefan Welker and Jonathan Chien and Maria Attarian and Travis Armstrong and Ivan Krasin and Dan Duong and Vikas Sindhwani and others , "Transporter networks: Rearranging the visual world for robotic manipulation" , Conference on Robot Learning , pp. 726--747, 2021 .

18. <a id="Manuelli19"></a> Lucas Manuelli* and Wei Gao* and Peter Florence and Russ Tedrake , "kPAM: KeyPoint Affordances for Category-Level Robotic Manipulation" , arXiv e-prints , pp. arXiv:1903.06684, Mar, 2019 .
    [文献链接](https://sites.google.com/view/kpam)

19. <a id="Gao20b"></a> Wei Gao and Russ Tedrake , "kPAM 2.0: Feedback control for generalizable manipulation" , IEEE Robotics and Automation Letters , 2020 .
    [文献链接](http://groups.csail.mit.edu/robotics-center/public_papers/Gao20b.pdf)

20. <a id="Florence18a"></a> Peter R. Florence* and Lucas Manuelli* and Russ Tedrake , "Dense Object Nets: Learning Dense Visual Object Descriptors By and For Robotic Manipulation" , Conference on Robot Learning (CoRL) , October, 2018 .
    [文献链接](http://groups.csail.mit.edu/robotics-center/public_papers/Florence18a.pdf)

</div>

---

**继续阅读**：[上一章：物体检测与分割（中文）](object-detection.md) · [全书目录与前言（中文）](preface.md) · [下一章：强化学习（中文）](reinforcement-learning.md)  
[无障碍访问](https://accessibility.mit.edu/) · © Russ Tedrake, 2026
