# LoTusY · 学习与研究笔记

具身智能、论文阅读、课程翻译与工程实践的个人博客。基于 [Material for MkDocs](https://github.com/squidfunk/mkdocs-material)，无前端打包工具。

[访问博客](https://jxtzz.github.io/) · [MIT《机器人操作》中文讲义](https://jxtzz.github.io/translations/robotic-manipulation/)

## 本地运行与检查

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m mkdocs serve
```

预览地址：`http://127.0.0.1:8000`。提交前执行：

```powershell
.\.venv\Scripts\python.exe -m mkdocs build --strict
.\.venv\Scripts\python.exe scripts/check_site.py .build/site
node scripts/check_ui.mjs
.\.venv\Scripts\python.exe scripts/audit_assets.py
```

UI 检查需要 Node 22+ 与 Chrome；可用 `CHROME_BIN` 指定可执行文件，或通过 `--cdp=http://127.0.0.1:9333` 连接隔离的测试浏览器。它自带本地服务器，覆盖 320/390/768/1440px、深浅色、目录、公式、表格、图片、搜索及静态回退。截图、报告与资源测量均写入 `.build/`。公式实测需要能访问 MathJax CDN，失败会报错，不跳过检查。

## 项目结构

```text
docs/                Markdown 文章、课程与原始附件（公开 URL 保持稳定）
  stylesheets/       extra：共享基础与列表；reader：阅读器；home-atlas：首页；course：课程
  javascripts/       按页面加载的阅读器、星图与公式初始化
overrides/main.html  小范围模板扩展，复用主题原生头部、搜索和配色控件
scripts/             元数据索引、构建钩子、资源测量与回归检查
mkdocs.yml           导航、精选学习顺序与构建配置
```

## 内容维护

新文章放入相应栏目，使用 YAML frontmatter：

```yaml
---
title: 页面标题
description: 一两句话说明这篇文章讨论的问题和内容。
tags: [PyTorch, 张量运算]
layout: article
---
```

`date` 和 `updated` 可选，格式为 `YYYY-MM-DD`：`date` 是可确认的内容记录/发布日期，`updated` 是站内内容修订日期。不知道就省略，界面显示“日期未注明”；不用复制时间、文件 mtime 或论文发表日期代替。初次整理保留了日志日期，仅从可确认的 Git 修改补录部分 `updated`，附 `updated_source: git:<提交>`；原著版本日期继续留在译文署名区。

更新正文时填写实际修订日期；纯目录移动、样式调整和元数据整理不需要给全部文章刷新日期。描述随内容修订，标签来自文章实际内容。论文可用 `pdf: assets/papers/文件名.pdf` 关联原文；资料库的未配笔记资源维护在 `papers/library.md` 的 `resources` 中，大小自动读取文件。

首页最近更新、论文/工程/日志列表及其导航由 `scripts/content.py` 生成，不用重复登记；占位符为 `<!-- article-list: recent|papers|engineering|journal -->`（选择一个名称）。最近更新先覆盖五个栏目，再补充有日期的记录；未注明日期不冒充最新发布。

理论学习路径、课程顺序和首页精选阅读路径保留人工编排，分别维护栏目首页、`mkdocs.yml` 与 `docs/index.md`。不要修改生成的 `content-index.json`。未发布文件同时配置 `.gitignore` 与 `exclude_docs`，不要忽略整个 `docs/`。

旧链接映射集中在 `scripts/site_hooks.py`，保留查询参数和章节锚点。移动公开文章前先新增重定向；文章正文、公式、参考文献锚点和附件链接须保留。

## 资源与发布

图片的原始像素保持不变；构建补充宽高、异步解码及懒加载，首屏关键图保持立即加载。MathJax 根据渲染标记只在公式页加载。论文图暂不统一转为有损 WebP/JPEG；`audit_assets.py` 评估最大的 PNG 是否有无损重压缩空间，实际压缩前应复查图中文字和线条。

资源总体积与页面下载量分别测量：源资源约 332.4 MiB，其中 34 个 PDF 约 296.5 MiB；这些 PDF 未嵌入首页，只在下载时请求。浏览器报告记录本地 HTTP、无压缩、关闭缓存时的首页传输量，不能等同于线上压缩后的速度。搜索索引也是页面加载的一部分。

PDF 暂不迁移或删除。若以后外置，可先将现有文件复制到 R2/S3 等存储并核对 SHA-256，再在支持 HTTP 301 的 CDN/域名层把原 `.pdf` 路径映射到新地址，验证下载、范围请求与缓存后切换。GitHub Pages 不提供这种路径级服务器重定向，不能用 HTML 页面替代 PDF。执行前需要存储账户、公开下载域名、CDN/域名控制权及明确上传授权；没有兼容映射时继续保留原文件。删除工作树文件也不会自动缩小 Git 历史，不做历史重写。

推送 `main` 后，Actions 严格构建、检查链接/译文/元数据并运行浏览器检查，全部通过后才发布；PR 只检查。译文与论文资料保留原作者署名和来源，版权归原作者及出版方。
