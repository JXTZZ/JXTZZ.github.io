---
title: 第 2 周 · 构建 MCP 服务器
description: CS146S 第 2 周作业说明中文译稿。为第三方 API 构建 MCP 服务器，设计工具接口、OAuth 流程与协议层测试。
tags:
- CS146S
- 软件开发
- 编程 Agent
layout: article
date: '2026-10-27'
---

<div class="reader-kicker">CS146S / 中文译稿 / WEEK 2 / 作业说明</div>

# 第 2 周：为第三方 API 构建 MCP 服务器

<div class="translation-meta" markdown>
**课程** [CS146S · The Modern Software Developer](https://themodernsoftware.dev)

**原文版本** [对应版本](https://github.com/mihail911/modern-software-dev-assignments/tree/df18839a8174ca57edad89c03948bd295459c9c0)

**对照阅读** [英文作业说明](https://github.com/mihail911/modern-software-dev-assignments/blob/df18839a8174ca57edad89c03948bd295459c9c0/week2/assignment.md)

**中文整理** LoTusY · 个人译稿
</div>

[本周作答模板](writeup.md) · [课程说明](../overview.md) · [课程目录](../index.md)

## 作业概述

构建一个 MCP 服务器，让编程 Agent 能够调用第三方 API。接通 API 只是相对简单的部分，真正的评分重点是**工具是否便于 Agent 使用**：模型能否选对工具、正确调用，并在出错后恢复。此外，还要实现完整可用的 **OAuth** 流程，因为许多值得封装的 API，并不能仅靠一个静态密钥访问。

### 学习目标

- **设计**工具接口，服务于一个只能看到 Schema 和描述的调用者。
- **实现**长期运行服务器中的 OAuth token 获取、缓存和刷新。
- **观察**接口设计的不同选择，如何改变 Agent 的实际行为。

## 参考资料

- **[FastMCP](https://github.com/jlowin/fastmcp)**：本次开发所用的框架。
- **[MCP 规范](https://modelcontextprotocol.io/)**：了解工具注解、传输方式和错误语义。
- **[MCP 服务器参考实现](https://github.com/modelcontextprotocol/servers)**：动手前先阅读一个实现。

## 第一部分：构建服务器（25 分）

选择一个使用 **OAuth 2.0** 的 API。GitHub、Google（Gmail、Calendar、Drive）、Notion、Linear、Slack、Spotify 和 Strava 都可以。尽量选自己实际使用的服务，这样更容易判断工具应该提供什么功能。

要求如下：

- **至少三个工具，而且必须能串联使用**：一个工具的输出可以直接作为另一个的输入。例如，`search_x` 返回的 ID 能交给 `get_x(id)`。
- **至少一个写操作工具**，能够更改真实状态。
- 代码放在 `week2/` 中，通过 stdio 通信，并用一条有文档说明的命令启动。

## 第二部分：让工具便于 Agent 使用（30 分）

这一部分是作业的核心。以下所有设计，都要从这个事实出发：Agent 只能看到你的 Schema 和 docstring，除此之外一无所知。

- **把约束写进 Schema，而不只写在文字说明里。** `sort: Literal["stars", "forks"]` 比 `sort: str` 再配一句 docstring 解释更有效。如果 Schema 与 docstring 冲突，Agent 就会遇到自己无法解释的校验失败。
- **整理输出结构。** 不要原样转发 API 返回的 JSON，只保留 Agent 需要的字段。原始载荷会消耗大量 token，还会把你的接口约定与第三方 API 紧紧绑定。
- **把错误作为数据返回，而不是抛出堆栈追踪。** 错误要有结构、能指导下一步，并包含足够细节，让 Agent 区分“不应重试”，例如 ID 错误，与“可以重试”，例如限流、短暂网络故障。堆栈追踪会终止 Agent 当前轮次，而 `{"error": "rate limited, retry after 42s"}` 这样的响应不会。
- **docstring 就是接口约定。** 说清 ID 从哪里来，例如“`list_templates` 返回的数字 ID”，工具才能串联。跨工具的工作流程说明，则放在 `FastMCP(instructions=...)` 中。
- **写操作要有一道刹车。** 使用工具注解，例如为读取工具设置 `readOnlyHint`；对不可逆操作，提供 `dry_run` 参数，或拆成“先预览、再提交”两个阶段。

## 第三部分：OAuth（25 分）

实现真实的 OAuth 授权流程，不能直接粘贴一个现成的 bearer token。

- 完成授权码换取 token 的全过程，**缓存** token，并在过期时**静默刷新**。
- **权限范围最小化。** 只读就申请只读权限，并在报告中解释每项 scope 为什么必要。
- **不要把秘密信息放进仓库。** 从第一次提交起，就通过环境变量读取。提交 `.mcp.json.example`，将真实配置文件加入 `.gitignore`。
- **处理会话中途 token 失效。** 长期运行的服务器不应在工具调用期间尝试打开浏览器，而应返回能指导后续处理的错误。

## 第四部分：接入与演示（20 分）

- 将服务器注册到 Claude Code 或 Cursor，并提交配置示例 `.mcp.json.example`。
- **展示一次完整的端到端会话**：Agent 至少串联调用两个工具，回应一条真实提示。记录提示词、实际调用的工具与参数，以及最终结果。
- **展示一次失败**，以及你的错误设计如何处理它。必要时主动触发，例如传入错误 ID，或撤销 token。
- 至少编写一个**通过 MCP 协议**访问服务器的测试，不能只是直接调用 Python 函数。

## 提交内容

在 `week2/` 中提交服务器代码、`.mcp.json.example`、测试，以及完成的 `writeup.md`。

## 评分标准（共 100 分）

| 部分 | 分值 | 满分要求 |
| --- | --- | --- |
| 一、服务器 | 25 | 至少三个能串联的工具，包含一个写操作；能通过文档中的命令启动 |
| 二、Agent 使用体验 | 30 | Schema 层约束、精简输出、结构化错误、支持串联的 docstring，以及写操作保护机制 |
| 三、OAuth | 25 | 授权码交换、缓存、刷新均可用；权限最小且有理由；未提交秘密信息；合理处理会话中途过期 |
| 四、接入 | 20 | 提交配置；展示真实的串联调用和失败处理过程；包含一个协议层测试 |

提交秘密信息，或工具 docstring 与 Schema 相互矛盾，都会扣分。

## 提交步骤

1. 将所有更改推送到远程仓库，以供评分。
2. **确认已将 `mihail911`、`isaackann` 和 `vdaita` 添加为作业仓库的协作者。**
3. 通过 Gradescope 提交。
4. **可选清理**：完成后，从 Agent 配置中移除服务器，使用 `claude mcp remove <name>`，或删除本地 `.mcp.json` 的对应条目；删除缓存 token 文件；到服务提供方的账号设置中撤销该 OAuth 应用的访问权限。

---

**继续阅读**：[本周作答模板](writeup.md) · [课程目录](../index.md) · [下一周：编写一个 Skill](../week3/assignment.md)
