---
title: 第 4 周 · 让仓库适合 Agent 使用
description: CS146S 第 4 周作业说明中文译稿。探索陌生代码库，编写针对仓库的 AGENTS.md 或 CLAUDE.md，并实际试用。
tags:
- CS146S
- 软件开发
- 编程 Agent
layout: article
date: '2026-10-27'
---

<div class="reader-kicker">CS146S / 中文译稿 / WEEK 4 / 作业说明</div>

# 第 4 周：让一个缺乏清晰指引的仓库适合 Agent 使用

<div class="translation-meta" markdown>
**课程** [CS146S · The Modern Software Developer](https://themodernsoftware.dev)

**原文版本** [对应版本](https://github.com/mihail911/modern-software-dev-assignments/tree/df18839a8174ca57edad89c03948bd295459c9c0)

**对照阅读** [英文作业说明](https://github.com/mihail911/modern-software-dev-assignments/blob/df18839a8174ca57edad89c03948bd295459c9c0/week4/assignment.md)

**中文整理** LoTusY · 个人译稿
</div>

[本周作答模板](writeup.md) · [课程说明](../overview.md) · [课程目录](../index.md)

## 作业概述

选择一个不熟悉的开源仓库，为它补上一份缺失的 Agent 配置文件：`AGENTS.md` 或 `CLAUDE.md`，任选其一。重点是练习如何组织一份好文件的**结构与内容**：什么该写、什么不该写，以及要具体到什么程度，才能真正派上用场。

### 学习目标

- **熟悉**陌生代码库，达到能准确描述它的程度。
- **整理**这些知识，形成可长期复用、能直接指导操作的仓库上下文。
- **判断**哪些信息适合放入 Agent 配置文件，哪些只是干扰。

## 参考资料

- **[AGENTS.md 格式](https://agents.md/)**
- **[如何写好 CLAUDE.md](https://www.hlyr.dev/blog/writing-a-good-claude-md)**
- **[Claude Code 最佳实践](https://code.claude.com/docs/en/best-practices)**

## 仓库示例

| 仓库 |
| --- |
| [bigskysoftware/htmx](https://github.com/bigskysoftware/htmx) |
| [iamkun/dayjs](https://github.com/iamkun/dayjs) |
| [slidevjs/slidev](https://github.com/slidevjs/slidev) |
| [pixijs/pixijs](https://github.com/pixijs/pixijs) |
| [yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) |
| [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) |
| [vuejs/vue](https://github.com/vuejs/vue) |
| [react/react](https://github.com/react/react) |

也可以自行选择，但必须是**并非由你编写**的仓库。

## 第一部分：探索仓库（20 分）

先 fork 仓库，再弄清它实际如何工作：

- 怎样构建、测试、进行 lint 检查和运行？找出**准确命令**。
- 主要目录和入口在哪里？
- 代码遵循了哪些没有明文说明的惯例？
- 哪些内容是生成文件、随仓库附带的第三方代码，或出于其他原因不应手动修改？

可以让 Agent 帮忙探索，这是最快的方式。

## 第二部分：编写配置文件（55 分）

在你 fork 出的仓库根目录中，编写 `AGENTS.md` 或 `CLAUDE.md`，至少覆盖：

- **项目是什么**：用几句话交代背景，让 Agent 知道项目所属领域。
- **架构**：关键目录、入口，以及各部分的关系。
- **命令**：构建、测试、lint、运行。准确到可以直接复制执行，例如 `pixi run pytest tests/`，而不是“运行测试”。
- **惯例**：仓库实际采用的测试写法、代码风格和命名模式。
- **操作边界**：哪些内容不能改；有哪些看似合理却会用错的命令或目录，值得提前提醒。

质量要求是：每一行都应当**针对这个仓库**。称职模型本就会遵循的通用建议，只是在凑字数；重复 README 也一样。文件每次都会加载进上下文，长度本身就是成本，尽量控制在一页左右。

## 第三部分：实际试用（25 分）

放好配置文件后，让 Agent 在仓库中完成两个小任务，例如运行测试套件，以及做一处小修改。

如实报告观察结果：它是否使用你给出的命令，是否遵循仓库惯例？再根据实际使用情况，指出**你会对配置文件做出的一项修改**。

## 提交内容

在 `week4/` 中提交 `AGENTS.md` 或 `CLAUDE.md`、完成的 `writeup.md`，以及你 fork 出的仓库链接。

## 评分标准（共 100 分）

| 部分 | 分值 | 满分要求 |
| --- | --- | --- |
| 一、探索 | 20 | 准确说明构建、测试、lint 命令，以及结构与惯例 |
| 二、配置文件 | 55 | 覆盖上述五方面；针对仓库；命令准确；内容简洁 |
| 三、试用 | 25 | 实际完成两个任务，如实报告，并提出一项有观察依据的改进 |

## 提交步骤

1. 将所有更改推送到远程仓库，以供评分。
2. **确认已将 `mihail911`、`isaackann` 和 `vdaita` 添加为作业仓库的协作者。**
3. 通过 Gradescope 提交。

---

**继续阅读**：[本周作答模板](writeup.md) · [课程目录](../index.md) · [下一周：评估 Agent 就绪度](../week5/assignment.md)
