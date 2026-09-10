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

`docs/` 是文章与附件源文件，`mkdocs.yml` 管理导航，`docs/stylesheets/extra.css` 管理外观。不要编辑生成的 HTML。仓库中旧的 `site/` 是历史构建产物；当前发布仅使用 `.build/site`。

推送 `main` 后，GitHub Actions 先构建并检查站内链接，再将通过检查的产物发布到 `gh-pages`。GitHub Pages 应使用 `gh-pages` 分支根目录。Pull Request 只执行检查。

## 本次资料迁移

迁移了 13 篇独立笔记、9 张图片、12 份论文 PDF；两个来源中相同的 5 篇论文笔记合并展示。未导入 `.obsidian` 应用配置；原始目录未作修改。

`migration-manifest.json` 记录来源相对路径、目标路径及文件哈希。`scripts/import_notes.py` 仅处理本次明确选择的资料集，会拒绝覆盖已编辑的文章。重复导入前请检查博客内的后续修改；新文章可直接写入 `docs/` 并添加导航。

```powershell
.\.venv\Scripts\python.exe scripts/import_notes.py --obsidian 'D:\AppData\Document\Obsidian' --papers 'D:\Project\paper'
```

导入时修正了图片链接、部分标题层级、列表格式及一处包含本机用户名的路径。论文内容保留原有笔记，不代表已经完成逐项学术核验。PDF 原文版权归作者与出版方所有。
