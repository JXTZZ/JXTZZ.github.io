"""Check the translated book's rendered Markdown structure after a MkDocs build."""
from html.parser import HTMLParser
from pathlib import Path
import re
import sys


BOOK = (
    'preface', 'introduction', 'robot', 'pick-and-place', 'pose-estimation',
    'bin-picking', 'motion-planning', 'mobile-manipulation', 'manipulator-control',
    'object-detection', 'deep-perception', 'reinforcement-learning', 'soft-robots',
    'spatial-algebra', 'drake', 'drakegym', 'manipulation-station', 'miscellaneous',
)
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta',
        'param', 'source', 'track', 'wbr'}


class Element:
    def __init__(self, tag, attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def has_class(self, name):
        return name in self.attrs.get('class', '').split()

    def elements(self):
        for child in self.children:
            if isinstance(child, Element):
                yield child
                yield from child.elements()

    def text(self):
        return ''.join(c.text() if isinstance(c, Element) else c for c in self.children)


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.root = Element('document')
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = Element(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def outside_math(node):
    if node.tag in {'pre', 'code', 'script'} or node.has_class('arithmatex'):
        return ''
    return ''.join(outside_math(c) if isinstance(c, Element) else c for c in node.children)


def chapter_name(source, index):
    title = re.search(r'^title: (.+)$', source, re.M).group(1)
    return title.split(' · ', 1)[1] if 0 < index <= 12 else title


def check_reader(root, source_root=None):
    source_root = source_root or Path(__file__).resolve().parents[1] / 'docs/translations/robotic-manipulation'
    sources = [(source_root / (slug + '.md')).read_text('utf-8') for slug in BOOK]
    displays = references = badges = 0
    errors = []
    for index, (slug, source) in enumerate(zip(BOOK, sources)):
        page = root / 'translations/robotic-manipulation' / slug / 'index.html'
        article = next(n for n in Document(page.read_text('utf-8')).root.elements() if n.tag == 'article')
        nodes = list(article.elements())

        def require(condition, message):
            if not condition:
                errors.append(f'{slug}: {message}')

        count = len(re.findall(r'^\s*\$\$\s*$', source, re.M)) // 2
        rendered = sum(n.tag == 'div' and n.has_class('arithmatex') for n in nodes)
        require(count == rendered, f'{count} display formulas in source, {rendered} recognized in HTML')
        require(not re.search(r'\$\$|\$[^\n$]+\$', outside_math(article)), 'unrecognized math delimiters in article')
        displays += rendered

        if '## 参考文献\n' in source:
            section = source.split('## 参考文献\n', 1)[1].split('\n---', 1)[0]
            ids = re.findall(r'\d+\. <a id="([^"]+)"></a>', section)
            wrappers = [n for n in nodes if n.has_class('references')]
            lists = [n for w in wrappers for n in w.elements() if n.tag == 'ol']
            require(len(wrappers) == len(lists) == 1, 'bibliography must be one continuous ordered list')
            if len(lists) == 1:
                items = [n for n in lists[0].children if isinstance(n, Element) and n.tag == 'li']
                actual_ids = [n.attrs['id'] for item in items for n in item.elements() if n.tag == 'a' and 'id' in n.attrs]
                require(len(items) == len(ids) and actual_ids == ids, 'bibliography entries or citation anchors changed')
                require(lists[0].attrs.get('start', '1') == '1' and all('value' not in n.attrs for n in items), 'bibliography numbering overridden')
                references += len(items)

        footers = [n for n in nodes if n.tag == 'p' and any(c.tag == 'strong' and c.text() == '继续阅读' for c in n.elements())]
        require(len(footers) == 1, 'missing or duplicate reading footer')
        if len(footers) == 1:
            footer = footers[0]
            children = [c for c in footer.children if isinstance(c, Element)]
            links = [(n.text(), n.attrs.get('href')) for n in children if n.tag == 'a']
            expected = []
            if index:
                expected.append((f'上一章：{chapter_name(sources[index - 1], index - 1)}（中文）', f'../{BOOK[index - 1]}/'))
            if index > 1:
                expected.append(('全书目录与前言（中文）', '../preface/'))
            if index + 1 < len(BOOK):
                expected.append((f'下一章：{chapter_name(sources[index + 1], index + 1)}（中文）', f'../{BOOK[index + 1]}/'))
            expected.append(('无障碍访问', 'https://accessibility.mit.edu/'))
            require(links == expected, 'reading links do not name the correct neighboring chapters')
            breaks = [i for i, n in enumerate(children) if n.tag == 'br']
            require(len(breaks) == 1 and breaks[0] + 1 < len(children) and children[breaks[0] + 1].tag == 'a' and children[breaks[0] + 1].text() == '无障碍访问', 'accessibility must begin the second footer line')
            require('© Russ Tedrake, 2026' in footer.text(), 'footer copyright missing')

        if slug == 'object-detection':
            badge_links = [n for n in nodes if n.tag == 'a' and any(c.tag == 'img' and c.attrs.get('alt') == 'Open in Colab' for c in n.elements())]
            expected_targets = [f'https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/book/segmentation/segmentation_{kind}.ipynb' for kind in ('train', 'inference')]
            require([n.attrs.get('href') for n in badge_links] == expected_targets, 'Colab badges must link to training and inference notebooks')
            badges += len(badge_links)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(BOOK)} readers, {displays} display formulas, {references} sequential references, {badges} Colab badges and named two-line footers.')


if __name__ == '__main__':
    check_reader(Path(sys.argv[1] if len(sys.argv) > 1 else '.build/site').resolve())
