---
title: 附录 B · Drake
description: MIT《机器人操作：感知、规划与控制》中文翻译：附录 B · Drake。
---

<div class="reader-kicker">ROBOTIC MANIPULATION / 中文译稿 / 附录 B</div>

# 附录 B　Drake

*Robotic Manipulation: Perception, Planning, and Control*

<div class="translation-meta" markdown>
**原著作者** [Russ Tedrake](http://people.csail.mit.edu/russt/)

**原文最后修改日期** 2026 年 10 月 11 日

**原文版权** © Russ Tedrake, 2020–2026

**对照阅读** [原文 PDF](pdfs/drake-original.pdf) · [在线原文](https://manipulation.mit.edu/drake.html)
</div>

## 本章目录

- [B.1　Pydrake](#section1)
- [B.2　在线 Jupyter Notebook](#notebooks)
    - [B.2.1　在 Google Colab 上运行](#section2.1)
    - [B.2.2　启用需要许可证的求解器](#section2.2)
- [B.3　在自己的计算机上运行](#section3)
- [B.4　获取帮助](#section4)

---

Drake 是本讲义使用的主要软件工具箱，它的开发很大程度上源于 MIT 的 [《欠驱动机器人学》](https://underactuated.mit.edu)课程。有关 Drake 的信息和文档，以官网为主要入口。本附录则补充一些实用说明，帮助你运行讲义中的示例和练习。

<a id="section1"></a>

## B.1　Pydrake

Drake 的主体是 C++ 库，采用严格的编码规范，其成熟度也以支持工业界的专业应用为目标。为了便于入门和快速验证想法，本讲义全部使用 Python，通过 Drake 的 Python 绑定 pydrake 调用底层功能。这些绑定还没有 C++ 后端那么成熟，不过正在快速完善。欢迎提出反馈，也欢迎贡献代码。

特别需要注意，C++ API 文档已经很完善，而自动生成的 Python 文档还在改进。我目前建议先到 [C++ 文档](https://drake.mit.edu/doxygen_cxx/)中查找所需功能；只有想确认某个类或方法在 pydrake 中怎样调用时，再查阅 [Python 文档](https://drake.mit.edu/pydrake/)。

Drake 也提供了不少[教程](https://github.com/RobotLocomotion/drake/tree/master/tutorials)，可以帮助你入门。

<a id="notebooks"></a>

## B.2　在线 Jupyter Notebook

几乎所有示例和练习都会以 [Jupyter Notebook](http://jupyter.org/) 的形式提供，这样我们就能利用近年来出现的免费云端计算资源。

<a id="section2.1"></a>

### B.2.1　在 Google Colab 上运行

点击示例旁的“Open in Colab”链接，依次运行各个单元格。第一个代码单元格会安装所需的软件包。如果想保留修改，请在 Google Drive 中保存一份副本。

许多 notebook 用 Meshcat 做交互式可视化。运行 `StartMeshcat()` 后，点击“Open Meshcat in a separate window”，使用查看器时保持 notebook 打开。如果希望把查看器嵌入当前单元格旁边，可以运行：

```python
from manipulation.meshcat_utils import DisplayMeshcat
DisplayMeshcat(meshcat)
```


<a id="section2.2"></a>

### B.2.2　启用需要许可证的求解器

如果持有相应许可证，就可以为 `MathematicalProgram` 启用更强大的求解器，其中多数对学术用户免费。具体方法请参阅 [MOSEK 文档](https://drake.mit.edu/doxygen_cxx/classdrake_1_1solvers_1_1_mosek_solver.html) 和 [Gurobi 文档](https://drake.mit.edu/doxygen_cxx/classdrake_1_1solvers_1_1_gurobi_solver.html)。

<a id="section3"></a>

## B.3　在自己的计算机上运行

学习深入之后，你很可能会希望在自己的计算机上运行这些示例，甚至进一步扩展它们。请使用 [Drake 支持的平台和配置](https://drake.mit.edu/installation.html)，也就是最近两个版本的 macOS 和 Ubuntu，以及系统默认的 Python 版本。最简单的安装方式是通过 `pip`；我通常建议先创建虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install "manipulation[all]" jupyterlab
```


接着，从 [GitHub 的 book 目录](https://github.com/RussTedrake/manipulation/tree/master/book)下载 notebook，用 `jupyter lab` 运行，也可以编写自己的 notebook。

Drake 官网还有其他[安装方式](https://drake.mit.edu/installation.html)，包括预编译程序和 Docker 镜像。如果选择从源码构建，那很好，但别忘了同时构建 Python 绑定。最后，还要确保 Drake 的安装路径包含在 `PYTHONPATH` 中。

如果想一次下载全部 notebook，可以用 

```python
git
    clone
```

 获取[课程讲义仓库](https://github.com/RussTedrake/manipulation)。通常最好从仓库根目录开始操作，然后用下面的命令启动 notebook：

```bash
jupyter notebook
```


各章的示例保存在 .ipynb 文件中，与对应章节的 HTML 文件放在一起；notebook 练习则位于 `exercises` 子目录。

<a id="section4"></a>

## B.4　获取帮助

如果使用 Drake 时遇到问题，请参考[帮助页面](https://drake.mit.edu/getting_help.html)的建议。如果问题出在 manipulation 仓库，可以到[项目 issue 页面](https://github.com/RussTedrake/manipulation/issues)查找已知问题，也可以提交新的 issue。


---

**继续阅读**：[上一章：附录 A · 空间代数（中文）](spatial-algebra.md) · [全书目录与前言（中文）](preface.md) · [下一章：附录 C · DrakeGym 环境（中文）](drakegym.md)  
[无障碍访问](https://accessibility.mit.edu/) · © Russ Tedrake, 2026
