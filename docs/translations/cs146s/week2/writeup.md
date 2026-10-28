---
title: 第 2 周 · 作答模板
description: CS146S 第 2 周作答模板中文译稿。为第三方 API 构建 MCP 服务器，设计工具接口、OAuth 流程与协议层测试。
tags:
- CS146S
- 软件开发
- 编程 Agent
layout: article
date: '2026-10-27'
---

<div class="reader-kicker">CS146S / 中文译稿 / WEEK 2 / 作答模板</div>

# 第 2 周作业报告

<div class="translation-meta" markdown>
**课程** [CS146S · The Modern Software Developer](https://themodernsoftware.dev)

**原文版本** [对应版本](https://github.com/mihail911/modern-software-dev-assignments/tree/df18839a8174ca57edad89c03948bd295459c9c0)

**对照阅读** [英文作答模板](https://github.com/mihail911/modern-software-dev-assignments/blob/df18839a8174ca57edad89c03948bd295459c9c0/week2/writeup.md)

**中文整理** LoTusY · 个人译稿
</div>

[本周作业说明](assignment.md) · [课程说明](../overview.md) · [课程目录](../index.md)

> 这是供学习者填写的作答模板。`TODO` 为原有待填项；正式提交请按作业要求填写课程仓库中的 `writeup.md`。

## 第一部分：服务器

**选择的 API**，以及选择理由：

> TODO

**启动方式**：用一条命令启动。

```text
TODO
```

| 工具 | 功能 | 读／写 | 可与哪些工具串联 |
| --- | --- | --- | --- |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

## 第二部分：Agent 使用体验

每项都要指出代码位置，使用 `file:line` 格式，并说明设计带来的作用。

| 设计选择 | 代码位置 | 原因与作用 |
| --- | --- | --- |
| Schema 层面的约束 | TODO | TODO |
| 输出整理：保留和去掉的字段 | TODO | TODO |
| 结构化错误：区分应重试与不应重试 | TODO | TODO |
| 帮助串联工具的 docstring | TODO | TODO |
| 写操作的保护机制 | TODO | TODO |

**观察到 Agent 误用工具后，你做出的一项修改：**

> TODO

## 第三部分：OAuth

**授权流程**：如何获取、缓存和刷新 token？

> TODO

**申请的 scope**，以及每项权限为何必要：

> TODO

**秘密信息管理**：哪些放在环境变量中，哪些由 `.gitignore` 排除？

> TODO

**会话中途 token 失效**：Agent 会看到什么？

> TODO

## 第四部分：接入

**注册配置**：给出 `.mcp.json.example`，并说明使用的客户端。

> TODO

**端到端会话记录**：提示词、实际调用的工具与参数，以及结果。

```text
TODO
```

**一次得到处理的失败**：你触发了什么问题，Agent 看到了什么，接下来做了什么？

```text
TODO
```

**协议层测试**：覆盖什么，怎样运行？

> TODO

## 提交检查

1. 用 `Command (⌘) + F` 搜索 `TODO`，确认待填内容都已完成。
2. 确认没有提交 token、客户端密钥、缓存 token 文件或真实的 `.mcp.json`。
3. 将全部更改推送到远程仓库，并通过 Gradescope 提交。
4. 可选清理：从 Agent 配置中移除服务器，删除缓存 token，并撤销 OAuth 应用的访问权限。

---

**继续阅读**：[本周作业说明](assignment.md) · [课程目录](../index.md) · [下一周：编写一个 Skill](../week3/assignment.md)
