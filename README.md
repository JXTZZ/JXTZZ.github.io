# LoTusY · 研究与学习笔记

个人技术博客：具身智能论文阅读、深度学习基础、工程实践与学习日志。

在线地址：https://jxtzz.github.io/

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
