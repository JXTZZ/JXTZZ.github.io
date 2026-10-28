---
title: 第 3 周 · 编写一个 Skill
description: CS146S 第 3 周作业说明中文译稿。选择可复用流程，编写 Skill，并验证触发条件与端到端执行。
tags:
- CS146S
- 软件开发
- 编程 Agent
layout: article
date: '2026-10-27'
---

<div class="reader-kicker">CS146S / 中文译稿 / WEEK 3 / 作业说明</div>

# 第 3 周：编写一个 Skill

<div class="translation-meta" markdown>
**课程** [CS146S · The Modern Software Developer](https://themodernsoftware.dev)

**原文版本** [对应版本](https://github.com/mihail911/modern-software-dev-assignments/tree/df18839a8174ca57edad89c03948bd295459c9c0)

**对照阅读** [英文作业说明](https://github.com/mihail911/modern-software-dev-assignments/blob/df18839a8174ca57edad89c03948bd295459c9c0/week3/assignment.md)

**中文整理** LoTusY · 个人译稿
</div>

[本周作答模板](writeup.md) · [课程说明](../overview.md) · [课程目录](../index.md)

## 作业概述

Skill 用来保存可重复执行的流程，也就是那些你原本每次都要粘贴到聊天中的操作说明。本周，你要编写一个 Skill，让它用于真实的开源仓库，并检验：该触发时能否触发，执行起来是否可靠。

### 学习目标

- **找出**值得固化为流程、而非反复口头解释的工作。
- **编写**能稳定触发的 Skill，并在正文中体现判断方法，而不只是罗列步骤。
- **验证**它是否在合适时机触发，实际执行是否经得起检验。

## 参考资料

- **[agentskills.io](https://agentskills.io)**：目前多数 Agent 都已支持的开放标准。
- **[Claude Code Skill 文档](https://code.claude.com/docs/en/skills)**：一种具体实现，可参考其文件布局和 frontmatter，也就是文档开头的元数据。
- **[Skill 示例](https://github.com/mattpocock/skills)**：动手前先读几个。

## 目标仓库

从下列仓库中选择一个，也可以使用自己的仓库。

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

## 第一部分：选择工作流程（15 分）

选一个你会**反复执行**的流程，其中至少包含一处真正需要判断的环节。仅仅运行一条固定命令，用 shell 别名就够了，还算不上 Skill。

如果没有想法，可以参考：

| Skill | 功能 |
| --- | --- |
| 从 issue 生成实现计划 | 阅读 issue，检查相关代码，指出含糊之处，给出分步计划 |
| PR 审查 | 按固定标准审查 diff，引用准确的文件和行号，区分阻碍合并的问题与一般建议 |
| 复现 bug | 构造最小复现，提出假设，运行针对性测试，之后才建议修复方案 |
| 编写测试 | 判断哪些行为需要测试，遵循仓库惯例，编写并运行测试 |
| 帮助熟悉仓库 | 围绕“添加身份认证”等任务，为新开发者介绍架构、命令和可能的修改位置 |
| 代码库迁移 | 为 Pydantic v1 升级 v2 等迁移，整理检测规则和安全的转换步骤 |
| 依赖升级 | 阅读变更日志，找出不兼容变更，完成修改并验证 |
| 从截图实现界面 | 使用仓库已有组件复现截图中的 UI，再进行视觉验证 |
| 性能调查 | 建立基线，修改前先做性能剖析，只做一次有针对性的修复，再比较结果 |
| 安全审查 | 针对一类明确的安全问题检查新代码，要求证据，而非猜测性警告 |

先手动完整执行一次。没有亲自做过，就无法把流程准确写进 Skill。

## 第二部分：编写 Skill（60 分）

按照 Agent Skills 标准编写 `SKILL.md`，放到你使用的 Agent 能发现 Skill 的目录。Claude Code、Cursor 等读取相同格式；具体查找哪个目录，请查所用 Agent 的文档。评分重点如下：

- **描述要能触发 Skill。** 既说明它做什么，也说明何时该用，措辞要接近用户实际会输入的话。许多 Skill 的失败，恰恰是根本没有被触发。
- **写判断方法，不只写步骤。** 情况含糊时如何决定，哪些事不该做，都要明确。复述工具文档并不能教会 Agent 新东西。
- **确定性的操作给出准确命令。**
- **正文保持精简。** 每次触发，正文都会进入上下文。较长参考材料应放到 Skill 目录的配套文件，再在正文引用，不要全部内联堆进去。

## 第三部分：测试（25 分）

- **触发测试**：准备三条应触发的提示，再准备一条接近适用范围、但不应触发的提示，如实记录结果。如果后一条也触发，说明描述范围过宽。
- **执行测试**：在所选仓库中端到端运行 Skill，展示结果。

## 提交内容

在 `week3/` 中提交 Skill 目录，包括 `SKILL.md` 和所有配套文件，以及完成的 `writeup.md`。

## 评分标准（共 100 分）

| 部分 | 分值 | 满分要求 |
| --- | --- | --- |
| 一、工作流程 | 15 | 确实值得重复使用，包含真正的判断环节，且事先手动执行过 |
| 二、Skill | 60 | 描述能触发；包含判断方法和准确命令；正文精简，必要时使用配套文件 |
| 三、测试 | 25 | 有触发测试，包括一条接近但不应触发的提示，并完成端到端运行 |

## 提交步骤

1. 将所有更改推送到远程仓库，以供评分。
2. **确认已将 `mihail911`、`isaackann` 和 `vdaita` 添加为作业仓库的协作者。**
3. 通过 Gradescope 提交。

---

**继续阅读**：[本周作答模板](writeup.md) · [课程目录](../index.md) · [下一周：让仓库适合 Agent 使用](../week4/assignment.md)
