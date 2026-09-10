"""Check all generated HTML local links, images, scripts and stylesheets."""
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


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


if __name__ == '__main__':
    main()
