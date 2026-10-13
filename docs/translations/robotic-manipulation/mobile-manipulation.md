---
title: 第 7 章 · 移动操作
description: MIT《机器人操作：感知、规划与控制》中文翻译：第 7 章 · 移动操作。
---

<div class="reader-kicker">ROBOTIC MANIPULATION / 中文译稿 / 07</div>

# 第 7 章　移动操作

*Robotic Manipulation: Perception, Planning, and Control*

<div class="translation-meta" markdown>
**原著作者** [Russ Tedrake](http://people.csail.mit.edu/russt/)

**原文最后修改日期** 2026 年 10 月 11 日

**原文版权** © Russ Tedrake, 2020–2026

**对照阅读** [原文 PDF](pdfs/mobile-manipulation-original.pdf) · [在线原文](https://manipulation.mit.edu/mobile.html)
</div>

## 本章目录

- [7.1　新登场的机器人](#section1)
- [7.2　感知有什么不同？](#section2)
    - [7.2.1　局部视图与主动感知](#section2.1)
    - [7.2.2　未知且可能变化的环境](#section2.2)
    - [7.2.3　机器人状态估计](#section2.3)
- [7.3　运动规划有什么不同？](#section3)
    - [7.3.1　轮式机器人](#section3.1)
        - [7.3.1.1　完整约束驱动](#section3.1.1)
        - [7.3.1.2　非完整约束驱动](#section3.1.2)
    - [7.3.2　腿式机器人](#section3.2)
- [7.4　仿真有什么不同？](#section4)
- [7.5　导航](#section5)
    - [7.5.1　建图，除了定位之外](#section5.1)
    - [7.5.2　识别可通行地形](#section5.2)
- [7.6　练习](#section6)
- [参考文献](#references)

---

前面主要讨论桌面操作。即使机器人固定在桌旁，只要把各种物体和挑战放到它面前，就已经足够我们研究很多章。不过，在继续深入之前，我想先问一个问题：如果机器人还装上轮子，甚至有了腿，会怎样？

移动操作中的许多技术难题，与桌面操作是相同的。但也会出现新的挑战，或者某些原有问题在移动场景中变得更加迫切。本章主要介绍这些挑战，以及一些解决思路。

更重要的是，给机械臂加上移动能力，往往也会拓展我们对机器人能做什么的想象。现在，可以让它在家中四处移动，完成各种任务。该让我们的作品走出去，[探索甚至操作这个世界](https://www.youtube.com/watch?v=WnTKllDbu5o)了。

<a id="section1"></a>

## 7.1　新登场的机器人

<a id="section2"></a>

## 7.2　感知有什么不同？

<a id="section2.1"></a>

### 7.2.1　局部视图与主动感知

[几何感知一章](pose-estimation.md#partial_views)已经讨论过，只能看到物体一部分时会遇到哪些困难。桌面操作通常可以从桌子周围的多个方向布置相机，但杂乱物体仍可能造成遮挡。

移动机器人通常不能事先把整个环境都配满传感器。我们往往只使用安装在机器人上的传感器，例如头部相机。如果只看到物体的一侧，前面的[对跖抓取策略](bin-picking.md#antipodal)就不能直接使用了。

解决这个问题的主要办法，是机器学习。只看到半个芥末瓶时，瞬时传感器读数并不足以规划抓取，我们还得推测物体背面的形状，也就是形状补全，尽管未必需要显式完成它。之所以能猜到背面是什么样，是因为过去接触过类似瓶子。那些物体的统计规律，补上了当前测量缺少的信息。因此，学习在这里并不只是方便，而是根本性的需求。后面很快会讨论基于学习的感知。

移动操作机器人通常还有另一种办法。既然相机和其他传感器都能移动，我们就可以主动改变它们的位置，收集信息、降低不确定性。主动感知和不确定性下规划已经有丰富的研究，后续还会补充。

<a id="section2.2"></a>

### 7.2.2　未知且可能变化的环境

讨论桌面操作时，我们先假设物体已知，再介绍对跖抓取等适用于未知物体的方法。但整个过程中，一直假设桌子和料箱的几何形状与位置已知。例如，会直接裁掉已知环境对应的点云，只留下物体上的点。机器人固定在桌旁时，这样做很合理；换成移动机器人，面对更多陌生环境，就必须放宽这个假设。

一种办法是为环境中的所有东西建模：先检测桌子，再估计位姿，必要时也估计形状，把参数化桌子模型拟合到观测上。但在广阔、多样的场景中，这条路很难走。有没有像物体对跖抓取那样的方法，能减少对环境显式建模的依赖？

我们希望环境表示不要求具体的物体模型，能支持运动规划所需的快速碰撞检测和最小距离计算，也能根据原始传感器数据高效更新，并扩展到大场景。合并后的原始点云容易更新，也不需要模型，但不太适合快速执行规划查询。

体素网格几乎满足这些要求，前面高效下采样点云时其实已经用过。它把有限范围的三维空间划分为固定大小的立方体，比如每个体素体积为 1 cm$^3.$，并标记是否被占据，更一般地也可以保存占据概率。无碰撞规划时，把占据体素视为需要避开的碰撞几何。许多体素碰撞查询很快，也容易并行，球体与体素的碰撞查询尤其高效，这也是 Drake 模型常用密集球体近似碰撞几何的原因之一。用点云更新体素网格同样方便，而合适的数据结构还能兼顾精细分辨率与大范围场景。

<a id="example-1"></a>

<a id="example1"></a>

#### 示例 7.1　OctoMap

OctoMap 是面向机器人、大规模且高效的体素网格算法和软件实现。[参考文献 1](#Hornung13)……

如今，把视觉场景分成物体和环境，通常最好借助机器学习。后面会实现相应的分割流程。

<a id="section2.3"></a>

### 7.2.3　机器人状态估计

移动操作在感知上的最大区别，是不仅要估计环境状态，还要估计机器人自身的状态。此前处理点云时，默认已知相机在世界中的位姿 ${}^WX^C$。我们可以利用已知形状和图案，通过精细标定离线估计相机外参，也默认机器人运动学已知。机器人固定在桌旁，世界到末端执行器的变换由高精度编码器直接测得时，这样假设很合理；一旦有了移动底座，就不再成立。

移动机器人状态估计已有很长的历史，[参考文献 2](#Thrun05)是经典参考。算法通常基于近似递归贝叶斯滤波，或与之密切相关的平滑方法。[参考文献 3](#Kaess08)移动底座除了轮式编码器和惯性测量单元（IMU），通常还装有测距传感器或相机。状态估计把这些带噪声的测量融合成一致的估计。一个重要经验是，只靠轮式里程计几乎总是不够：真实轮子会打滑，而视觉和测距信息提供了独立且必要的测量。

自[参考文献 2](#Thrun05)问世以来，很多事情已经改变。激光雷达和深度相机的量程、精度、帧率都有显著进步，使这些方法更加有效。

<a id="example-2"></a>

<a id="example2"></a>

#### 示例 7.2　在 Drake 中仿真测距传感器

更大的变化也许是，状态估计从高度依赖测距和深度相机，逐渐转向越来越强的纯 RGB 方法。这得益于视觉惯性里程计[参考文献 4](#Scaramuzza11)、单目深度估计[参考文献 5](#Ming21)、稠密三维重建与运动恢复结构[参考文献 6](#Mildenhall21)、[参考文献 7](#Kerbl23)等技术的进展。

<a id="section3"></a>

## 7.3　运动规划有什么不同？

先用现有工具箱，搭建一个尽可能简单的移动操作机器人。

<a id="example-3"></a>

<a id="example3"></a>

### 示例 7.3　给 iiwa 加上移动关节底座

给 iiwa 的底座增加几个自由度：不再把连杆 0 焊接到世界，而是插入三个移动关节，分别对应 $x$、$y$ 和 $z$ 方向，中间用不可见、无质量的连杆连接。$x$、$y$ 不设范围限制，$z$ 则设置适度限位。iiwa 的第一个关节本来就绕底座 $z$ 轴旋转，再去掉它的限位，允许连续转动。改动很简单，却已经具备移动机器人的基本特点。

**运行示例**：[在 Colab 中打开](https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/book/mobile/mobile.ipynb)

现在，原来的运动规划工具需要改多少，才能适用于这台 iiwa？答案是几乎不用改。它们本来就相当通用，基本适用于任意运动学构型。新底座只是在运动学树上多了几个关节，仍然符合原有框架。

一个需要考虑的变化，是自由度从 7 个增加到了 10 个。轨迹优化围绕局部轨迹求解，容易扩展到高维，所以问题不大。采样规划更容易受到维数影响，不过 RRT 在十维空间通常仍可工作。PRM 也能处理十维，但可能需要高效实现，而且已开始接近适用极限。

另一个新问题是连续转动关节，比如绕 $z$ 轴旋转没有限位。规划时必须留意：采样规划通常要调整距离度量和扩展操作，允许连接边绕过 $2\pi$ 的角度边界；非凸轨迹优化本身不用改算法，但可能遇到局部极小值。对于[凸集图轨迹优化](motion-planning.md#gcs)，应让 IRIS 区域在局部坐标系中工作，并使任何环绕关节上的区域宽度不超过 $\pi$。[参考文献 8](#Cohn23)处理不当，就可能规划出不合理的绕远路径。[这个示例](https://ggcs-anonymous-submission.github.io/meshcat/1CL-4CR/trajopt.html)展示了运动学轨迹优化在穿过货架时得到次优解的情形，示例来自[参考文献 8](#Cohn23)。

如果不用移动关节，而用轮子或腿来实现底座，会怎样？有些轮式机器人和大多数腿式机器人，可以直接提供这种移动关节抽象；另一些则会引入规划必须考虑的额外约束。

<a id="section3.1"></a>

### 7.3.1　轮式机器人

虽然真实轮子会打滑，运动规划时通常仍假设不打滑，再用反馈控制跟踪规划路径。[参考文献 9](#Siciliano16)（第 Ch. 49 节）

<a id="section3.1.1"></a>

#### 7.3.1.1　完整约束驱动

<a id="example-4"></a>

<a id="example4"></a>

##### 示例 7.4　麦克纳姆轮的运动学

<a id="example-5"></a>

<a id="example5"></a>

##### 示例 7.5　在 Drake 中仿真麦克纳姆轮

<a id="section3.1.2"></a>

#### 7.3.1.2　非完整约束驱动

差速驱动、Dubins 车、Reeds–Shepp 车。

<a id="example-6"></a>

<a id="example6"></a>

##### 示例 7.6　差速驱动的运动学

<a id="section3.2"></a>

### 7.3.2　腿式机器人

如果你负责机器人的腿部规划和控制，有[大量文献](https://underactuated.mit.edu)可供研究。不过，若用 [Spot](https://bostondynamics.com/products/spot/) 做移动操作，采用 Boston Dynamics 的接口，无论你觉得幸运还是遗憾，都不必深入腿部的底层控制。Spot 的 API 大体把腿式平台当作完整约束底座。当然，还有一些次要因素：机器人的质心受到动力学约束，为保持平衡，实际运动可能偏离你请求的轨迹。

<a id="example-7"></a>

<a id="example7"></a>

#### 示例 7.7　用 Spot 做移动操作

如果要在崎岖或不连续的地形上操作，问题当然更有意思，而这正是腿式平台擅长的地方。

<a id="section4"></a>

## 7.4　仿真有什么不同？

移动操作不仅要求扩展自主算法，也促使我们扩大仿真范围。原来只模拟桌上的几个物体，现在则要把机器人放进更广阔的环境。

此前说到仿真，主要指物理引擎、渲染引擎等基础技术。机器人领域常见的主流仿真器包括 NVIDIA Isaac／Omniverse、MuJoCo 和 Drake；PyBullet 曾经很流行，但已不再积极维护。还有一类也称为仿真器的工具，建立在这些引擎或 Unity、Unreal 等游戏引擎之上，提供更高层的抽象。我更倾向于把它们看作内容整合平台，它们同样很重要。

<a id="example-8"></a>

<a id="example8"></a>

### 示例 7.8　Sapien

<a id="example-9"></a>

<a id="example9"></a>

### 示例 7.9　Behavior 1K

<a id="example-10"></a>

<a id="example10"></a>

### 示例 7.10　Habitat 3.0

我希望能方便地把这些工具的问题定义和资产导入 Drake，既使用我们的仿真引擎，也使用感知、规划与控制工具解决任务。为此，我们正在逐步增强对其他格式的加载和转换支持。

<a id="section5"></a>

## 7.5　导航

到目前为止，本章多数思路都可以看作对原有操作工具箱的适度扩展。不过，移动机器人的导航还有一些桌面操作中基本不会遇到的重要问题。

<a id="section5.1"></a>

### 7.5.1　建图，除了定位之外

<a id="section5.2"></a>

### 7.5.2　识别可通行地形

<a id="section6"></a>

## 7.6　练习

<a id="exercise-1"></a>

<a id="exercise1"></a>

### 练习 7.1　搭建仿真

本练习通过加载物体，搭建一个适合操作的场景。所有操作都在[对应 notebook](https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/book/mobile/exercises/simulation_set_up.ipynb)中完成，具体步骤如下：

1. 分析给定 SDF 文件中的碰撞几何。

2. 搭建自己的仿真。

<a id="exercise-2"></a>

<a id="exercise2"></a>

### 练习 7.2　移动底座的逆运动学

本练习为固定底座和移动底座的机械臂分别实现逆运动学优化。去掉底座的位置约束后，优化问题会更容易求解。所有操作都在[对应 notebook](https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/book/mobile/exercises/mobile_base_ik.ipynb)中完成。

<a id="references"></a>

## 参考文献

<div class="references" markdown="1">

1. <a id="Hornung13"></a> Armin Hornung and Kai M. Wurm and Maren Bennewitz and Cyrill  Stachniss and Wolfram Burgard , "OctoMap: An Efficient Probabilistic 3D Mapping Framework Based   on Octrees" , Autonomous Robots , 2013 .

2. <a id="Thrun05"></a> S. Thrun and W. Burgard and D. Fox , "Probabilistic Robotics" , MIT Press
    , 2005 .

3. <a id="Kaess08"></a> M. Kaess and A. Ranganathan and F. Dellaert , "iSAM: Incremental Smoothing and Mapping" , IEEE Transactions on Robotics , vol. 24, no. 6, pp. 1365-1378, 2008 .

4. <a id="Scaramuzza11"></a> Davide Scaramuzza and Friedrich Fraundorfer , "Visual odometry [tutorial]" , Robotics & Automation Magazine, IEEE , vol. 18, no. 4, pp. 80--92, 2011 .

5. <a id="Ming21"></a> Yue Ming and Xuyang Meng and Chunxiao Fan and Hui Yu , "Deep learning for monocular depth estimation: A review" , Neurocomputing , vol. 438, pp. 14--33, 2021 .

6. <a id="Mildenhall21"></a> Ben Mildenhall and Pratul P Srinivasan and Matthew Tancik and Jonathan T Barron and Ravi Ramamoorthi and Ren Ng , "Nerf: Representing scenes as neural radiance fields for view synthesis" , Communications of the ACM , vol. 65, no. 1, pp. 99--106, 2021 .

7. <a id="Kerbl23"></a> Bernhard Kerbl and Georgios Kopanas and Thomas Leimkuhler and George Drettakis , "3d gaussian splatting for real-time radiance field rendering" , ACM Transactions on Graphics (ToG) , vol. 42, no. 4, pp. 1--14, 2023 .

8. <a id="Cohn23"></a> Thomas Cohn and Mark Petersen and Max Simchowitz and Russ Tedrake , "Non-Euclidean Motion Planning with Graphs of Geodesically-Convex Sets" , Robotics: Science and Systems , 2023 .
    [文献链接](http://groups.csail.mit.edu/robotics-center/public_papers/Cohn23.pdf)

9. <a id="Siciliano16"></a> Bruno Siciliano and Oussama Khatib , "Springer handbook of robotics" , Springer
    , 2016 .

</div>

---

**继续阅读**：[上一章：运动规划（中文）](motion-planning.md) · [全书目录与前言（中文）](preface.md) · [下一章：机械臂控制（中文）](manipulator-control.md)  
[无障碍访问](https://accessibility.mit.edu/) · © Russ Tedrake, 2026
