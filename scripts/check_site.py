"""Check all generated HTML local links, images, scripts and stylesheets."""
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit
import json
from check_reader import check_reader


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        attribute = 'href' if tag in ('a', 'link') else 'src'
        if tag in ('a', 'link', 'img', 'script', 'source') and attrs.get(attribute):
            self.links.append(attrs[attribute])


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.build/site').resolve()
    pages = {}
    for page in root.rglob('*.html'):
        parser = Links()
        parser.feed(page.read_text(encoding='utf-8'))
        pages[page] = parser
    if not pages:
        raise SystemExit('No generated HTML found')
    errors, checked = [], 0
    for page, parser in pages.items():
        # MkDocs 404 is served at unknown nesting depths; its relative paths vary at runtime.
        if page.name == '404.html':
            continue
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (root / unquote(url.path).lstrip('/')) if url.path.startswith('/') else (page.parent / unquote(url.path))
            if not url.path:
                target = page
            if target.is_dir():
                target = target / 'index.html'
            target = target.resolve()
            checked += 1
            if not target.is_relative_to(root) or not target.is_file():
                errors.append(f'{page.relative_to(root)}: missing {link}')
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                # MathJax and Material may create their own dynamic anchors.
                if not url.fragment.startswith(('__', 'mjx')):
                    errors.append(f'{page.relative_to(root)}: missing anchor {link}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(pages)} HTML pages, {checked} local links and assets checked.')
    check_reader(root)
    from check_reader import Document
    records = json.loads((root / 'content-index.json').read_text(encoding='utf-8'))
    assert len({r['source'] for r in records}) == len(records), 'Duplicate articles in generated metadata'
    assert all(r['description'] and isinstance(r['tags'], list) for r in records), 'Incomplete metadata'
    assert not any('/idea/' in r['source'] or 'human-depth-ambiguity' in r['source'] for r in records), 'Unpublished metadata exposed'
    home = Document((root / 'index.html').read_text(encoding='utf-8')).root
    assert not any(n.has_class('home-recent') or n.has_class('recent-notes') for n in home.elements()), 'Homepage must not contain recent updates'
    assert not any(n.has_class('hero-thread') or n.has_class('reading-note') for n in home.elements()), 'Removed homepage reading modules returned'
    assert len([n for n in home.elements() if n.has_class('topic-card')]) == 3, 'Homepage collection cards missing'
    course = next(n for n in home.elements() if n.has_class('featured-course'))
    assert 'Robotic Manipulation' in next(n for n in course.elements() if n.tag == 'h2').text(), 'Homepage course title must be English'
    image_count = math_pages = 0
    for page in pages:
        if page.name == '404.html':
            continue
        nodes = list(Document(page.read_text(encoding='utf-8')).root.elements())
        articles = [n for n in nodes if n.tag == 'article']
        if not articles:
            continue  # Legacy URL redirects intentionally have no article.
        if page.relative_to(root).as_posix().removesuffix('index.html') in {r['url'] for r in records if r['section'] == 'papers'}:
            headings = [n for n in articles[0].elements() if n.tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6') and n.attrs.get('id')]
            toc = next(n for n in nodes if n.has_class('md-nav--secondary'))
            toc_links = [n for n in toc.elements() if n.tag == 'a']
            links = {n.attrs.get('href') for n in toc_links}
            assert headings[1:] and all('#' + n.attrs['id'] in links for n in headings[1:]), f'{page}: incomplete paper outline'
            record = next(r for r in records if r['url'] == page.relative_to(root).as_posix().removesuffix('index.html'))
            assert record['method'] and toc_links[0].text().strip() == record['method'], f'{page}: first outline entry must be the method name'
        descriptions = [n.attrs.get('content') for n in nodes if n.tag == 'meta' and n.attrs.get('name') == 'description']
        assert descriptions and descriptions[0], f'{page}: page description missing'
        marked = any(n.has_class('arithmatex') for n in nodes)
        loader = any(n.tag == 'script' and 'tex-mml-chtml' in n.attrs.get('src', '') for n in nodes)
        assert marked == loader, f'{page}: MathJax must load exactly on math pages'
        math_pages += loader
        for n in articles[0].elements():
            if n.tag == 'img':
                assert 'width' in n.attrs and 'height' in n.attrs and n.attrs.get('loading') in ('eager', 'lazy'), f'{page}: missing intrinsic image/loading metadata'
                image_count += 1
    print(f'PASS: {len(records)} metadata entries, all paper outlines, homepage collections, {math_pages} conditional math pages and {image_count} dimensioned images.')


if __name__ == '__main__':
    main()
