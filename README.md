# LoTusY · 学习与研究笔记

记录具身智能、论文阅读、课程翻译与工程实践的个人博客。

[访问博客](https://jxtzz.github.io/) · [MIT《机器人操作》中文讲义](https://jxtzz.github.io/translations/robotic-manipulation/)

## 本地预览

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m mkdocs serve
```

打开 `http://127.0.0.1:8000`。构建与检查：

```powershell
.\.venv\Scripts\python.exe -m mkdocs build --strict
.\.venv\Scripts\python.exe scripts/check_site.py .build/site
```

## 项目结构

```text
docs/         文章、插图与前端资源
overrides/    主题模板定制
scripts/      构建钩子与检查
mkdocs.yml    站点配置与导航
```

新增文章放入 `docs/` 并更新导航；未发布内容同时配置 `.gitignore` 与 `exclude_docs`，不忽略整个 `docs/`。构建产物统一放在 `.build/site/`。

推送 `main` 后，GitHub Actions 自动检查并发布到 GitHub Pages；Pull Request 只执行检查。

基于 [Material for MkDocs](https://github.com/squidfunk/mkdocs-material)。译文与论文资料保留原作者署名和来源，版权归原作者及出版方。
