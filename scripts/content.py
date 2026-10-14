"""Metadata-driven lists and navigation, using MkDocs' included YAML dependency."""
from datetime import date, datetime
from html import escape
from pathlib import Path
import re
from urllib.parse import quote

from mkdocs.utils import get_relative_url
import yaml

SECTIONS = {'papers': '论文', 'translations': '课程', 'deep-learning': '理论',
            'engineering': '工程', 'journal': '日志'}
AUTO_NAV = {'论文阅读': 'papers', '工程实践': 'engineering', '学习日志': 'journal'}
CATALOG = []


def metadata(text):
    text = text.lstrip('\ufeff')
    if text.startswith('---\n'):
        match = re.match(r'^---\n(.*?)\n---\n', text, re.S)
        if match:
            return yaml.safe_load(match[1]) or {}
    return {}


def iso_date(value, field, source):
    if value in (None, ''):
        return None
    if isinstance(value, datetime):
        value = value.date()
    try:
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(value)):
            raise ValueError('date format')
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as error:
        raise ValueError(f'{source}: {field} must be YYYY-MM-DD, or omitted') from error


def sort_key(record):
    return (record.get('updated') or record.get('date') or '', record['source'])


def collect(files, config):
    previous_sources = {r['source'] for r in CATALOG}
    CATALOG.clear()
    for file in files.documentation_pages():
        if not file.inclusion.is_included():
            continue
        source = file.src_uri
        section = source.split('/')[0]
        if section not in SECTIONS or source.endswith('/index.md') or source == 'papers/library.md':
            continue
        meta = metadata(file.content_string)
        title = meta.get('title') or re.search(r'^# (.+)$', file.content_string, re.M).group(1)
        description = meta.get('description', '')
        tags = meta.get('tags', [])
        if not description or not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ValueError(f'{source}: description and a list of tags are required')
        record = {'source': source, 'url': file.url, 'section': section, 'title': str(title),
                  'description': description, 'tags': tags,
                  'date': iso_date(meta.get('date'), 'date', source),
                  'updated': iso_date(meta.get('updated'), 'updated', source),
                  'updated_source': meta.get('updated_source'), 'pdf': meta.get('pdf')}
        if record['date'] and record['updated'] and record['updated'] < record['date']:
            raise ValueError(f'{source}: updated cannot precede date')
        CATALOG.append(record)
    CATALOG.sort(key=sort_key, reverse=True)
    for entry in config['nav']:
        for label, section in AUTO_NAV.items():
            if label not in entry:
                continue
            # Manual entries (e.g. the PDF library) remain; article links come from metadata.
            existing = entry[label]
            index = [item for item in existing if item == f'{section}/index.md']
            sources = {r['source'] for r in CATALOG if r['section'] == section} | previous_sources
            manual = [item for item in existing if item not in index and not (isinstance(item, dict) and any(v in sources for v in item.values()))]
            entry[label] = index + [{r['title']: r['source']} for r in CATALOG if r['section'] == section] + manual
    return files


def list_html(records, page, *, name='文章列表'):
    if not records:
        return '<p class="list-empty" role="status">这里还没有公开文章，整理完成后会出现在此处。</p>'
    rows = []
    for record in records:
        href = get_relative_url(record['url'], page.file.url)
        stamp = record['updated'] or record['date']
        prefix = '更新' if record['updated'] else '记录'
        time = (f'<time datetime="{stamp}">{prefix} {stamp}</time>' if stamp else '<span>日期未注明</span>')
        tags = ''.join(f'<span class="article-tag">{escape(tag)}</span>' for tag in record['tags'][:3])
        rows.append(
            f'<li><a class="article-row" data-section="{record["section"]}" href="{escape(href, quote=True)}">'
            f'<div class="article-row__meta"><span class="article-section">{SECTIONS[record["section"]]}</span>{time}</div>'
            f'<span class="article-row__title">{escape(record["title"])}</span>'
            f'<span class="article-row__description">{escape(record["description"])}</span>'
            f'<span class="article-row__tags">{tags}</span><span class="article-row__arrow" aria-hidden="true">↗</span></a></li>'
        )
    return f'<ul class="article-list" aria-label="{escape(name, quote=True)}">' + ''.join(rows) + '</ul>'


def resource_html(page, config):
    resources = [{'title': r['title'], 'pdf': r['pdf'], 'note': r['url']} for r in CATALOG if r.get('pdf')]
    resources += page.meta.get('resources', [])
    rows = []
    for record in resources:
        path = (Path(config['docs_dir']) / record['pdf']).resolve()
        if not path.is_file() or not path.is_relative_to(Path(config['docs_dir']).resolve()):
            raise ValueError('Missing PDF: ' + record['pdf'])
        pdf = quote(get_relative_url(record['pdf'], page.file.url), safe='/')
        note = ('<a href="' + get_relative_url(record['note'], page.file.url) + '">阅读笔记</a>') if record.get('note') else '<span>笔记待整理</span>'
        rows.append(f'<li class="resource-row"><div><strong>{escape(record["title"])}</strong><small>PDF · {path.stat().st_size / 1024**2:.1f} MB</small></div><div class="resource-actions">{note}<a href="{pdf}">下载原文 ↗</a></div></li>')
    return '<ul class="resource-list" aria-label="论文原文">' + ''.join(rows) + '</ul>'


def expand_lists(markdown, page, config):
    def replace(match):
        section = match[1]
        records = [r for r in CATALOG if r['section'] == section]
        return list_html(records, page)
    markdown = re.sub(r'<!-- article-list: (papers|engineering|journal) -->', replace, markdown)
    return markdown.replace('<!-- resource-list -->', resource_html(page, config)) if '<!-- resource-list -->' in markdown else markdown
