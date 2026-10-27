---
title: 第 1 周 · 作答模板
description: CS146S 第 1 周作答模板中文译稿。捕获并分析真实 Claude Code 会话的系统提示词、工具接口、错误恢复与上下文管理。
tags:
- CS146S
- 软件开发
- 编程 Agent
layout: article
date: '2026-10-27'
---

<div class="reader-kicker">CS146S / 中文译稿 / WEEK 1 / 作答模板</div>

# 第 1 周作业报告

<div class="translation-meta" markdown>
**课程** [CS146S · The Modern Software Developer](https://themodernsoftware.dev)

**原文版本** [df18839](https://github.com/mihail911/modern-software-dev-assignments/tree/df18839a8174ca57edad89c03948bd295459c9c0)

**对照阅读** [英文作答模板](https://github.com/mihail911/modern-software-dev-assignments/blob/df18839a8174ca57edad89c03948bd295459c9c0/week1/writeup.md)

**中文整理** LoTusY · 个人译稿
</div>

[本周作业说明](assignment.md) · [课程说明](../overview.md) · [课程目录](../index.md)

> 这是供学习者填写的作答模板。`TODO` 为原有待填项；正式提交请按作业要求填写课程仓库中的 `writeup.md`。

## 第一部分：捕获会话

**环境配置**：信息应足以让读者复现捕获过程。

```text
claude --version：TODO
mitmproxy 版本：  TODO
代理启动命令：   TODO
设置文件：       TODO（路径及 env 配置块）
```

**会话情况**：执行什么任务、使用哪个仓库，共产生多少个 `POST /v1/messages` 请求？

> TODO

| 要求 | 证据 |
| --- | --- |
| 至少修改两个文件 | TODO |
| 至少失败一次 | TODO |
| 任务需要明确规划 | TODO |
| 使用自己的仓库 | TODO |

**脱敏情况**：下面引用的片段中，你隐去了什么信息，为什么？

> TODO

## 第二部分：系统提示词批注

**a. 结构。** 按原顺序列出主要部分，每项用一句话说明其作用，并解释为什么按这个顺序安排。

> TODO

**b. 语气与详略。** 引用控制回复方式的指令，再说明它们在防止什么失败。

```text
TODO
```

> TODO

**c. 何时不该行动。** 引用破坏性操作的执行条件、范围限制或拒绝条件，说明各自的作用。

```text
TODO
```

> TODO

**d. 环境上下文。** Agent 获知了哪些机器、仓库、会话信息？位于请求的 `system` 字段，还是某条 `role: "system"` 消息中？

> TODO

**e. `<system-reminder>`。** 它出现在哪里？引用实例，指出至少两种有证据支持的用途，并解释为何要在对话中途注入，而非只在开头说明一次。

```text
TODO
```

> TODO

## 第三部分：工具设计批注

**工具清单。** 不同请求之间，工具集合是否变化？如果变了，由什么触发？

| 内置工具 | MCP 工具 | 延迟加载工具 | **总数** | 会话中途是否变化？ |
| --- | --- | --- | --- | --- |
| TODO | TODO | TODO | **TODO** | TODO |

**分析两个工具。** 请选有明显差别的工具。

| 分析项 | 工具 1 | 工具 2 |
| --- | --- | --- |
| 名称 | TODO | TODO |
| Schema 的关键字段 | TODO | TODO |
| 必填、可选、不开放的参数或能力，以及原因 | TODO | TODO |
| 描述在防范什么：引用原句并指出错误行为 | TODO | TODO |
| 刻意不做什么，这又说明什么 | TODO | TODO |

为什么选择这两个工具？

> TODO

## 第四部分：行为分析

**每项回答必须标为 `[OBSERVED]` 或 `[INFERRED]`，并注明证据。未标注的回答不得分。**

**a. 错误恢复**：`TODO: label` · 证据：`TODO`

Agent 看到的内容，请逐字引用：

```text
TODO
```

随后尝试了什么，经过多少轮才恢复：

> TODO

**b. 规划**：`TODO: label` · 证据：`TODO`

> TODO

**c. 计划与任务状态**：`TODO: label` · 证据：`TODO`

计划如何创建和推进？模型每轮能看到哪些任务状态信息，它们放在请求的哪里？

> TODO

**d. 子 Agent**：`TODO: label` · 证据：`TODO`

何时委派任务？子 Agent 收到什么信息，又返回什么？

> TODO

**e. 上下文管理**：`TODO: label` · 证据：`TODO`

会话变长后，请求载荷有哪些变化？

> TODO

## 第五部分：反思

**两项你会借鉴的设计**，以及各自解决的问题：

1. TODO
2. TODO

**一项你会采用不同做法的设计**：同时分析原设计可能存在的理由。

> TODO

**这次调用轨迹分析带来的一条改变**：今后你会怎样调整对编程 Agent 的引导方式？

> TODO

## 提交检查

1. 用 `Command (⌘) + F` 搜索 `TODO`，确认待填内容都已完成。
2. 确认引用片段中没有混入凭据或含 `x-api-key` 的请求头。
3. 将全部更改推送到远程仓库，并通过 Gradescope 提交。
4. 别忘了从仓库的 `.claude/settings.json` 中移除 `ANTHROPIC_BASE_URL`！

---

**继续阅读**：[课程目录](../index.md) · [返回本周作业说明](assignment.md)
