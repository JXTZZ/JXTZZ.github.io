---
title: 附录 A · 空间代数
description: MIT《机器人操作》中文译稿。集中查阅位置、旋转、位姿、空间速度和空间力的 Monogram 记号与运算规则。
tags:
- 机器人操作
- MIT 讲义
layout: article
updated: '2026-10-13'
updated_source: git:c0a5c26daf5d9a0bb25c5620f4c002497f69f678
---

<div class="reader-kicker">ROBOTIC MANIPULATION / 中文译稿 / 附录 A</div>

# 附录 A　空间代数

*Robotic Manipulation: Perception, Planning, and Control*

<div class="translation-meta" markdown>
**原著作者** [Russ Tedrake](http://people.csail.mit.edu/russt/)

**原文最后修改日期** 2026 年 10 月 11 日

**原文版权** © Russ Tedrake, 2020–2026

**对照阅读** [原文 PDF](pdfs/spatial-algebra-original.pdf) · [在线原文](https://manipulation.mit.edu/spatial.html)
</div>

## 本章目录

- [A.1　位置、旋转与位姿](#section1)
- [A.2　空间速度](#section2)
- [A.3　空间力](#section3)

---

本讲义陆续介绍了空间代数的基本规则，我自己也经常回头查阅。为便于参考，这里把它们集中整理一下。这些规则会大量使用前文的[Monogram 记号法](pick-and-place.md#monogram)。

Drake 文档也很好地总结了[多体系统中的物理量](https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html)，并说明它们与 Monogram 记号法之间的对应关系。

<a id="section1"></a>

## A.1　位置、旋转与位姿

按照[前文](pick-and-place.md#spatial_algebra)的约定，${}^Bp^A_C$ 表示点或坐标系 $A$ 相对于点或坐标系 $B$ 的位置，分量在坐标系 $C$ 中表达。${}^BR^A$ 表示从坐标系 $B$ 测得的坐标系 $A$ 的朝向。与向量不同，纯旋转不需要再指定一个“表达坐标系”。同样，${}^BX^A$ 表示从坐标系 $B$ 测得的坐标系 $A$ 的位姿或变换。位姿也不使用表达坐标系下标，因为我们总是把它表达在参考坐标系中。

空间代数的基本规则如下：

- 位置向量在同一个坐标系中表达时，只要参考点与目标点的标记首尾衔接，就可以相加：
  
    $$
    {}^Ap^B_F + {}^Bp^C_F
          = {}^Ap^C_F.
    $$
  
    加法满足交换律，也可以明确地定义相反向量：
  
    $$
    {}^Ap^B_F = -
          {}^Bp^A_F.
    $$
  
    这些关系应该很直观，不过还是请你亲自确认一下。

- 乘上旋转矩阵，就能改变向量的表达坐标系：
  
    $$
    {}^Ap^B_G = {}^GR^F {}^Ap^B_F.
    $$
  
    你也许会惊讶，只用旋转就能完成这个变换，但确实如此。表达坐标系的位置，并不会改变两个点之间的相对位置。

- 旋转的参考坐标系与目标坐标系首尾衔接时，可以相乘：
  
    $$
    {}^AR^B \: {}^BR^C = {}^AR^C.
    $$
  
    逆运算也很简单：
  
    $$
    \left[{}^AR^B\right]^{-1} = {}^BR^A.
    $$
  
    如果用旋转矩阵表示，这就是通常的矩阵求逆。旋转矩阵还是正交矩阵，所以有 $R^{-1}=R^T.$。

- 当位置以某个坐标系为参考，而且分量也在同一个坐标系中表达时，变换可以把这些运算统一成一种方便的记号：
  
    $$
    {}^Gp^A = {}^GX^F {}^Fp^A = {}^Gp^F + {}^Fp^A_G
          = {}^Gp^F + {}^GR^F {}^Fp^A.
    $$
  
  

- 变换可以复合：
  
    $$
    {}^AX^B {}^BX^C =
          {}^AX^C,
    $$
  
    也存在逆变换：
  
    $$
    \left[{}^AX^B\right]^{-1} = {}^BX^A.
    $$
  
    请注意，变换的逆虽然形式也很简单，但一般并不是 $X^T,$，即 $X^{-1}$ 通常不等于转置。

<a id="section2"></a>

## A.2　空间速度

正如[前文](pick-and-place.md#jacobian)所介绍的，我们用一个六维向量表示位姿的变化率，称为[空间速度](https://drake.mit.edu/doxygen_cxx/group__multibody__spatial__vectors.html)：

$$
{}^AV^B_C = \begin{bmatrix} {}^A\omega^B_C \\
    {}^A\text{v}^B_C \end{bmatrix}.
$$

其中，${}^AV^B_C$ 是坐标系 $B$ 相对于坐标系 $A$、在坐标系 $C$ 中表达的空间速度，也称为速度旋量（twist）。${}^A\omega^B_C \in \mathbb{R}^3$ 是角速度，描述坐标系 $B$ 相对于 $A$ 的转动，并在 $C$ 中表达；${}^A\text{v}^B_C \in \mathbb{R}^3$ 是平移速度，其简写约定与位置相同。空间速度也能自然地纳入空间代数：

- 速度在同一个坐标系中表达时可以相加：
  
    $$
    \begin{gathered}
            {}^A\text{v}^B_F + {}^B\text{v}^C_F = {}^A\text{v}^C_F, \qquad {}^A\omega^B_F +
            {}^B\omega^C_F = {}^A\omega^C_F,\end{gathered}
    $$
  
    相反向量满足 ${}^AV^C_F = - {}^CV^A_F,$。

- 旋转可以改变速度的表达坐标系：
  
    $$
    {}^A\text{v}^B_G = {}^GR^F {}^A\text{v}^B_F,
            \qquad {}^A\omega^B_G = {}^GR^F {}^A\omega^B_F.
    $$
  
  

- 跨坐标系合成平移速度时，有：
  
    $$
    {}^A\text{v}^C_F = {}^A\text{v}^B_F + {}^B\text{v}^C_F
              + {}^A\omega^B_F \times {}^Bp^C_F.
    $$
  
  

- 由此可见，求平移速度的相反向量，不能只交换参考坐标系和被测坐标系，还要考虑额外的一项：
  
    $$
    -{}^A\text{v}^B_F =
            {}^B\text{v}^A_F + {}^A\omega^B_F \times {}^Bp^A_F.
    $$
  
  

<a id="section3"></a>

## A.3　空间力

正如[前文](bin-picking.md#spatial_force)所介绍的，我们用六维向量定义[空间力](https://drake.mit.edu/doxygen_cxx/group__multibody__spatial__vectors.html)，并采用 Monogram 记号：

$$
F^{B_p}_{{\text{name}},C} = \begin{bmatrix}
    \tau^{B_p}_{\text{name},C} \\ f^{B_p}_{\text{name},C} \end{bmatrix} \quad \text{ or,
    if you prefer } \quad \left[F^{B_p}_{\text{name}}\right]_C = \begin{bmatrix}
    \left[\tau^{B_p}_{\text{name}}\right]_C \\ \left[f^{B_p}_{\text{name}}\right]_C
    \end{bmatrix}.
$$

其中，$F^{B_p}_{\text{name},C}$ 表示作用在点或坐标系原点 $B_p$、在坐标系 $C$ 中表达的某个具名空间力。名称可以省略；如果没有指定表达坐标系，就默认采用世界坐标系。对力而言，建议在作用点 $B_p$ 的记号中明确写出受力刚体 $B$，因为我们经常需要同时处理大小相等、方向相反的力。

<a id="spatial_force_algebra"></a>

空间力同样符合空间代数的规则：

- 作用于同一刚体、同一作用点，并在同一坐标系中表达的空间力，可以相加。例如：
  
    $$
    F^{B_p}_{\text{total},C}
            = \sum_i F^{B_p}_{i,C} .
    $$
  
  

- 把空间力从作用点 $B_p$ 移到 $B_q$，需要使用叉积：
  
    <a id="equation-eq-spatial_force_shift"></a>

    $$
    f^{B_q}_C
            = f^{B_p}_C, \qquad \tau^{B_q}_C = \tau^{B_p}_C + {}^{B_q}p^{B_p}_C
            \times f^{B_p}_C.
       \tag{A.1}
    $$
  
  

- 与其他空间向量一样，旋转可以改变力的表达坐标系：
  
    $$
    f^{B_p}_D = {}^DR^C
            f^{B_p}_C, \qquad \tau^{B_p}_D = {}^DR^C
            \tau^{B_p}_C.
    $$
  
  


---

**继续阅读**：[上一章：软体机器人与触觉感知（中文）](soft-robots.md) · [全书目录与前言（中文）](preface.md) · [下一章：附录 B · Drake（中文）](drake.md)  
[无障碍访问](https://accessibility.mit.edu/) · © Russ Tedrake, 2026
