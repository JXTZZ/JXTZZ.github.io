---
title: CS146S · 课程说明与环境准备
description: CS146S 中文课程作业说明：第 1–5 周阅读导航，以及 Python、Conda 与 Poetry 环境准备。
tags:
- CS146S
- 软件开发
- 编程 Agent
layout: article
date: '2026-10-27'
---

<div class="reader-kicker">CS146S / 中文译稿 / README</div>

# CS146S：课程说明与环境准备

<div class="translation-meta" markdown>
**课程** [CS146S · The Modern Software Developer](https://themodernsoftware.dev)

**原文版本** [对应版本](https://github.com/mihail911/modern-software-dev-assignments/tree/df18839a8174ca57edad89c03948bd295459c9c0)

**对照阅读** [英文 README](https://github.com/mihail911/modern-software-dev-assignments/blob/df18839a8174ca57edad89c03948bd295459c9c0/README.md)

**中文整理** LoTusY · 个人译稿
</div>

这里收录斯坦福大学 2026 年秋季课程 [CS146S：现代软件开发者（The Modern Software Developer）](https://themodernsoftware.dev)的作业。

各页提供对应版本的英文原文链接。命令、文件名、API 字段和评分要求保留原意；作答模板里的 `TODO` 是留给学习者填写的内容。正式提交时仍按作业要求填写各周的 `writeup.md`。

## 中文作业导航

| 周次 | 作业主题 | 文档 |
| --- | --- | --- |
| 第 1 周 | 剖析一次真实 Claude Code 会话的调用轨迹 | [作业说明](week1/assignment.md) · [作答模板](week1/writeup.md) |
| 第 2 周 | 为第三方 API 构建 MCP 服务器 | [作业说明](week2/assignment.md) · [作答模板](week2/writeup.md) |
| 第 3 周 | 编写一个 Skill | [作业说明](week3/assignment.md) · [作答模板](week3/writeup.md) |
| 第 4 周 | 让一个缺乏清晰指引的仓库适合 Agent 使用 | [作业说明](week4/assignment.md) · [作答模板](week4/writeup.md) |
| 第 5 周 | 评估代码仓库对 Agent 的支持程度 | [作业说明](week5/assignment.md) · [作答模板](week5/writeup.md) |

## 配置开发环境

请先获取[原课程作业仓库](https://github.com/mihail911/modern-software-dev-assignments)。下面的命令在课程作业仓库中执行。

以下步骤适用于 Python 3.12。

1. **安装 Anaconda。**

   - 下载并安装 [Anaconda 个人版](https://www.anaconda.com/download)。
   - 新开一个终端，确保 `conda` 已加入 `PATH`。

2. **创建并激活 Conda 环境，使用 Python 3.12。**

   ```bash
   conda create -n cs146s python=3.12 -y
   conda activate cs146s
   ```

3. **安装 Poetry。**

   ```bash
   curl -sSL https://install.python-poetry.org | python -
   ```

4. **在已激活的 Conda 环境中，用 Poetry 安装项目依赖。**

   在仓库根目录运行：

   ```bash
   poetry install --no-interaction
   ```

---

**继续阅读**：[第 1 周作业：调用轨迹分析](week1/assignment.md) · [课程目录](index.md)
