---
title: 附录 D · 搭建自己的机器人操作工作站
description: MIT《机器人操作：感知、规划与控制》中文翻译：附录 D · 搭建自己的机器人操作工作站。
---

<div class="reader-kicker">ROBOTIC MANIPULATION / 中文译稿 / 附录 D</div>

# 附录 D　搭建自己的机器人操作工作站

*Robotic Manipulation: Perception, Planning, and Control*

<div class="translation-meta" markdown>
**原著作者** [Russ Tedrake](http://people.csail.mit.edu/russt/)

**原文最后修改日期** 2026 年 10 月 11 日

**原文版权** © Russ Tedrake, 2020–2026

**对照阅读** [原文 PDF](pdfs/manipulation-station-original.pdf) · [在线原文](https://manipulation.mit.edu/station.html)
</div>

## 本章目录

- [D.1　消息传递](#section1)
- [D.2　Kuka LBR iiwa + Schunk WSG 夹爪](#section2)
- [D.3　Franka Panda](#section3)
- [D.4　Intel RealSense D415 深度相机](#section4)

---

仿真是一种非常强大的工具。能够免费提供仿真，也让更多人获得了平等参与研究的机会。就在几年前，人们还不相信：在仿真中开发的操作系统，到了真实世界也能工作。如今，我们不断看到证据表明，仿真的保真度已经相当高；只要在仿真中表现足够鲁棒，就有理由期待它能迁移到现实。

当然，看着自己的想法和代码让一台真实机器人动起来，仍是无可替代的体验。本附录希望尽可能降低搭建机器人硬件平台的门槛，提供物料清单、底层驱动和基本配置说明，帮助你拥有自己的操作工作站。

<a id="section1"></a>

## D.1　消息传递

Drake 支持多种消息传递系统，包括 ROS1。大多数情况下，我并不坚持你一定要选择哪一种。这门课采用 [LCM](https://lcm-proj.github.io/)，主要有两个原因：其一，依赖很轻量，把 ROS 设为 Drake 的必需依赖会带来不少负担和限制；其二，对于机器人最底层的通信，我们更倾向于使用多播 UDP，而不是 TCP/IP。

因此，下面列出和提供的主要驱动都基于 LCM。在条件允许时，我也会指出对应的 ROS 方案。如果你已有成熟的 ROS 工作流程，又想使用我们的驱动，可以通过简单的 lcm2ros／ros2lcm 桥接程序，把 LCM 消息发布到 ROS topic 上，用于记录日志等。

<a id="section2"></a>

## D.2　Kuka LBR iiwa + Schunk WSG 夹爪

我们大量使用过 iiwa7 R800 和 iiwa14 R820。Drake 提供这两种型号的 URDF，驱动也应当同时支持它们。课程实验选用 iiwa7，因为它稍小一些，而且实验不需要更高的末端负载能力。

- [IIWA LCM 驱动](https://github.com/RobotLocomotion/drake-iiwa-driver)

Schunk WSG 50 是一款可靠、精确的双指夹爪，我们使用下来很满意。

- [Schunk LCM 驱动](https://github.com/RobotLocomotion/drake-schunk-driver)

补充说明：我们正在评估 Franka Panda 作为替代平台，也可能很快在这里提供相应驱动。

<a id="section3"></a>

## D.3　Franka Panda

Franka Panda 系列是很受欢迎的 iiwa 替代方案。我们使用的驱动通过 LCM 与机器人直接通信。

- [Franka Panda 驱动](http://github.com/RobotLocomotion/drake-franka-driver)

当然，也可以[直接使用 ROS 驱动](https://github.com/RobotLocomotion/drake-ros)。

<a id="section4"></a>

## D.4　Intel RealSense D415 深度相机

我们选择 D415，是因为它的最小工作距离较合适，而且多台相机同时观察同一场景时不会彼此干扰。

- [LCM 驱动](https://github.com/RobotLocomotion/realsense2-lcm-driver)

- [相机标定工具箱](https://github.com/ToyotaResearchInstitute/handical).


---

**继续阅读**：[上一章：附录 C · DrakeGym 环境（中文）](drakegym.md) · [全书目录与前言（中文）](preface.md) · [下一章：附录 E · 其他事项（中文）](miscellaneous.md)  
[无障碍访问](https://accessibility.mit.edu/) · © Russ Tedrake, 2026
