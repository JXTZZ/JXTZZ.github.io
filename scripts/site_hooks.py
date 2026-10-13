"""Small build customizations: deployment version and legacy article URLs."""
from html import escape
import json
from pathlib import Path
from urllib.parse import urljoin

REDIRECTS = {
    'attention/python': 'engineering/tensor-inplace',
    'attention/Linear_Algebra': 'engineering/tensor-operations',
    'meeting/image-to-arkts': 'engineering/image-to-arkts',
}


def on_post_build(config, **kwargs):
    root = Path(config['site_dir'])
    version = str(config['extra'].get('site_version', 'local'))
    (root / 'site-version.json').write_text(json.dumps({'version': version}) + '\n', encoding='utf-8')
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
