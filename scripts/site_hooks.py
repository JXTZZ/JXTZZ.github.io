"""Small build customizations: deployment version and legacy article URLs."""
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import urljoin
from mkdocs.utils import get_relative_url
from mkdocs.structure.toc import AnchorLink, TableOfContents

from content import CATALOG, collect, expand_lists
from page_assets import prepare_content

REDIRECTS = {
    'attention/python': 'engineering/tensor-inplace',
    'attention/Linear_Algebra': 'engineering/tensor-operations',
    'meeting/image-to-arkts': 'engineering/image-to-arkts',
}


def on_files(files, config, **kwargs):
    return collect(files, config)


def on_page_markdown(markdown, page, config, **kwargs):
    return expand_lists(markdown, page, config)


def on_page_content(html, page, config, **kwargs):
    # Older paper notes have several H1s. Material displays only the first H1's
    # children; gather every section under that root without editing the source
    # headings, their HTML level or the bookmarked IDs.
    items = list(page.toc)
    if page.file.src_uri.startswith('papers/') and page.meta.get('layout') == 'article' and len(items) > 1 and items[0].level == 1:
        root = AnchorLink(items[0].title, items[0].id, 1)
        root.children = list(items[0].children) + items[1:]
        page.toc = TableOfContents([root])
    return prepare_content(html, page, config)


def outline(items):
    rows = []
    for item in items:
        if re.sub('<[^>]+>', '', item.title).startswith('本章目录'):
            continue
        child = outline(item.children) if item.children else ''
        rows.append(f'<li><a href="{escape(item.url, quote=True)}">{item.title}</a>{child}</li>')
    return '<ul>' + ''.join(rows) + '</ul>' if rows else ''


def on_post_page(output, page, config, **kwargs):
    # The upstream header has no title sub-block. Replace just its first title span,
    # rather than copying the complete header and owning its search/palette controls.
    home = get_relative_url('', page.url)
    brand = escape(config['site_name'] + ' · 回到首页', quote=True)
    pattern = r'(<div class="md-header__topic">\s*)<span class="md-ellipsis">(.*?)</span>'
    output, count = re.subn(pattern, lambda m: m[1] + f'<a class="md-ellipsis header-home-link" href="{home}" aria-label="{brand}" title="{brand}">' + m[2] + '</a>', output, count=1, flags=re.S)
    if count != 1:
        raise ValueError('Material header title changed: review the small homepage-link customization')
    if page.meta.get('layout') != 'article':
        return output
    items = list(page.toc)
    if items and items[0].level == 1:
        items = items[0].children
    toc = outline(items)
    details = ('<details class="mobile-toc"><summary>本页目录<span aria-hidden="true">展开 / 收起</span></summary>'
               '<nav aria-label="本页目录">' + toc + '</nav></details>') if toc else ''
    stamp = page.meta.get('updated') or page.meta.get('date')
    time = (f'<time datetime="{stamp}">{"站内更新" if page.meta.get("updated") else "记录"} {stamp}</time>' if stamp else '<span>日期未注明</span>')
    tags = ''.join(f'<span class="article-tag">{escape(t)}</span>' for t in page.meta.get('tags', []))
    meta = f'<div class="article-meta">{time}{tags}</div>'
    # Only the article H1; no search result/header markup is moved.
    start = output.index('<article')
    end = output.index('</h1>', start) + len('</h1>')
    output = output[:end] + meta + details + output[end:]
    # Keep the source outline and its anchor, but collapse it on desktop and avoid
    # duplicating the mobile outline. The original section contents are untouched.
    chapter = r'(<h2\b[^>]*>本章目录.*?</h2>\s*)(<ul>.*?</ul>)(?=\s*(?:<hr|<p|<h2))'
    output = re.sub(chapter, r'<details class="chapter-contents"><summary>本章目录 · 章节概览</summary>\1\2</details>', output, count=1, flags=re.S)
    return output


def on_post_build(config, **kwargs):
    root = Path(config['site_dir'])
    version = str(config['extra'].get('site_version', 'local'))
    (root / 'site-version.json').write_text(json.dumps({'version': version}) + '\n', encoding='utf-8')
    (root / 'content-index.json').write_text(json.dumps(CATALOG, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for old, new in REDIRECTS.items():
        page = root / old / 'index.html'
        page.parent.mkdir(parents=True, exist_ok=True)
        target = '../../' + new + '/'
        canonical = urljoin(config['site_url'], new + '/')
        page.write_text(
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="site-version" content="{escape(version, quote=True)}">'
            '<meta name="robots" content="noindex">'
            f'<link rel="canonical" href="{escape(canonical, quote=True)}">'
            '<title>文章已迁移 · LoTusY</title>'
            f'<script>location.replace({json.dumps(target)}+location.search+location.hash)</script>'
            f'<noscript><meta http-equiv="refresh" content="0;url={target}"></noscript>'
            f'</head><body><p>文章已迁移，<a href="{target}">继续阅读</a>。</p></body></html>\n',
            encoding='utf-8',
        )
