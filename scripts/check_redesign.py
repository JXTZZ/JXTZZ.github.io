"""Compare a redesign with a saved build: routes, IDs, links and article content."""
import json
import re
from pathlib import Path
from hashlib import sha256
from collections import Counter
import sys
from urllib.parse import urljoin
from check_reader import Document


def inventory(root):
    pages = {}
    for file in root.rglob('*.html'):
        if '.hallmark' in file.parts:
            continue
        route = file.relative_to(root).as_posix()
        # A newly added no-script navigation fallback is shell, not article prose.
        html = re.sub(r'<noscript>.*?</noscript>', '', file.read_text('utf-8'), flags=re.S)
        nodes = list(Document(html).root.elements())
        article = next((n for n in nodes if n.tag == 'article'), None)
        body = list(article.elements()) if article else []
        pages[route] = {
            'has_article': article is not None,
            'duplicate_ids': [key for key, count in Counter(n.attrs['id'] for n in body if n.attrs.get('id')).items() if count > 1],
            'ids': sorted({n.attrs['id'] for n in body if n.attrs.get('id')}),
            'links': sorted({urljoin('https://jxtzz.github.io/' + route, n.attrs['href'])
                             for n in body if n.tag == 'a' and n.attrs.get('href') and not n.has_class('headerlink')}),
            'text': ' '.join(article.text().split()) if article else '',
            'math': [n.text() for n in body if n.has_class('arithmatex')],
            'code': [n.text() for n in body if n.tag == 'code'],
            'images': [n.attrs.get('src') for n in body if n.tag == 'img'],
        }
    return pages


def check(before, after):
    old, new = inventory(before), inventory(after)
    assert old.keys() == new.keys(), f'Public routes changed: {old.keys() ^ new.keys()}'
    indexes = {'index.html', 'about/index.html', 'papers/index.html', 'papers/library/index.html',
               'deep-learning/index.html', 'engineering/index.html', 'journal/index.html',
               'translations/index.html', 'translations/robotic-manipulation/index.html',
               'translations/cs146s/index.html'}
    errors = []
    for route, page in old.items():
        current = new[route]
        if route in indexes:
            assert not current['duplicate_ids'], f'{route}: duplicate IDs: {current["duplicate_ids"]}'
        for key in ('ids', 'links'):
            missing = set(page[key]) - set(current[key])
            if missing:
                errors.append(f'{route}: lost {key}: {missing}')
        if route not in indexes:
            for key in ('text', 'math', 'code', 'images'):
                if page[key] != current[key]:
                    errors.append(f'{route}: article {key} changed')
    old_catalog = json.loads((before / 'content-index.json').read_text('utf-8'))
    new_catalog = json.loads((after / 'content-index.json').read_text('utf-8'))
    assert old_catalog == new_catalog, 'Publication metadata changed'
    protected = 0
    for file in before.rglob('*'):
        if file.is_file() and file.suffix.lower() in {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'} and '.hallmark' not in file.parts:
            target = after / file.relative_to(before)
            assert target.is_file() and sha256(file.read_bytes()).digest() == sha256(target.read_bytes()).digest(), f'Resource changed: {target}'
            protected += 1
    assert not errors, '\n'.join(errors)
    result = {'routes': len(old), 'articles_unchanged': sum(p['has_article'] for route, p in old.items() if route not in indexes and route != '404.html'),
              'non_index_pages_unchanged': len(old.keys() - indexes),
              'preserved_article_ids': sum(len(p['ids']) for p in old.values()),
              'preserved_link_targets': sum(len(p['links']) for p in old.values()),
              'catalog_unchanged': len(old_catalog), 'unchanged_binary_resources': protected}
    print('PASS: redesign preservation ' + json.dumps(result))
    return result


if __name__ == '__main__':
    check(Path(sys.argv[1] if len(sys.argv) > 1 else '.build/redesign-before'),
          Path(sys.argv[2] if len(sys.argv) > 2 else '.build/site'))
