# LoTusY · 研究与学习笔记

个人技术博客：具身智能论文阅读、MIT 课程翻译、深度学习基础、工程实践与学习日志。

在线地址：https://jxtzz.github.io/

## 课程翻译

[MIT《机器人操作》中文讲义](https://jxtzz.github.io/translations/robotic-manipulation/)包含全书目录与前言、12 章正文及 5 篇附录，共 18 篇译稿。译文与资源统一放在 `docs/translations/robotic-manipulation/`：

- `index.md`：专题首页与全书入口。
- `preface.md`：全书目录与前言，主要小节链接到中文锚点，缩进子条目保持普通文字。
- `introduction.md`、`robot.md` 等英文主题文件名：正文与附录译稿，完整路径见 `mkdocs.yml`。
- `assets/`：原文插图；`pdfs/`：译稿对应的英文原文 PDF。

修订译稿时，保留原作者署名、原文链接、版本日期和参考文献锚点；章节交叉引用使用站内 Markdown 路径。原始的嵌套 `docs/` 导入目录已合并，后续直接维护上述文件。

## 本地预览

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m mkdocs serve
```

在浏览器打开 `http://127.0.0.1:8000`。

## 构建与检查

```powershell
.\.venv\Scripts\python.exe -m mkdocs build --strict --site-dir .build/site
.\.venv\Scripts\python.exe scripts/check_site.py .build/site
```

`docs/` 是文章与附件源文件，`mkdocs.yml` 管理导航，`docs/stylesheets/extra.css` 管理外观。不要提交或编辑生成的 HTML；本地构建统一输出到被忽略的 `.build/site`。

推送 `main` 后，GitHub Actions 先构建并检查站内链接，再将通过检查的产物发布到 `gh-pages`。GitHub Pages 应使用 `gh-pages` 分支根目录。Pull Request 只执行检查。每次线上构建还会写入版本标记；浏览器若短暂拿到旧的 HTML，会自动重新请求同一部署版本，避免不同页面显示两套界面。
