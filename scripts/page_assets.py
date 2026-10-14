"""Intrinsic image dimensions and rendered-page helpers; no image resampling."""
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
import posixpath
import re
import struct
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


@lru_cache(maxsize=512)
def image_dimensions(path):
    path = Path(path)
    data = path.read_bytes()
    if data.startswith(b'\x89PNG'):
        return struct.unpack('>II', data[16:24])
    if data[:3] == b'GIF':
        return struct.unpack('<HH', data[6:10])
    if data[:2] == b'\xff\xd8':
        i = 2
        while i + 8 < len(data):
            if data[i] != 255:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0xD9) or marker == 255:
                i += 2
                continue
            length = int.from_bytes(data[i + 2:i + 4], 'big')
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack('>HH', data[i + 5:i + 9])
                return w, h
            i += 2 + length
    if path.suffix.lower() == '.svg':
        root = ET.fromstring(data)
        # Graphviz SVGs often use points. Preserve their CSS-pixel intrinsic size,
        # rather than treating viewBox units as pixels and shrinking the lettering.
        factors = {'': 1, 'px': 1, 'pt': 96 / 72, 'pc': 16, 'in': 96,
                   'cm': 96 / 2.54, 'mm': 96 / 25.4}
        def length(value):
            match = re.fullmatch(r'([\d.]+)(px|pt|pc|in|cm|mm)?', value)
            return round(float(match[1]) * factors[match[2] or '']) if match else None
        w, h = length(root.get('width', '')), length(root.get('height', ''))
        if w and h:
            return w, h
        box = root.get('viewBox', '').replace(',', ' ').split()
        if len(box) == 4:
            return round(float(box[2])), round(float(box[3]))
    return None


class ImageMarkup(HTMLParser):
    def __init__(self, page, config):
        super().__init__(convert_charrefs=False)
        self.page, self.config, self.parts = page, config, []
        self.text_length, self.images = 0, 0

    def handle_starttag(self, tag, attrs):
        raw = self.get_starttag_text()
        if tag == 'img':
            attrs = dict(attrs)
            url = urlsplit(attrs.get('src', ''))
            if not url.scheme and not url.netloc:
                # Markdown has already rewritten src relative to the output URL.
                relative = posixpath.normpath(posixpath.join(posixpath.dirname(self.page.url), unquote(url.path)))
                path = (Path(self.config['docs_dir']) / relative).resolve()
                if path.is_file() and path.is_relative_to(Path(self.config['docs_dir']).resolve()):
                    dims = image_dimensions(str(path))
                    additions = []
                    if dims and 'width' not in attrs and 'height' not in attrs:
                        additions += [f'width="{dims[0]}"', f'height="{dims[1]}"']
                    critical = 'atlas-fallback' in attrs.get('class', '') or (self.images == 0 and self.text_length < 700)
                    if 'loading' not in attrs:
                        additions.append('loading="eager"' if critical else 'loading="lazy"')
                    if 'decoding' not in attrs:
                        additions.append('decoding="async"')
                    if 'atlas-fallback' in attrs.get('class', ''):
                        additions.append('fetchpriority="high"')
                    raw = re.sub(r'\s*/?>$', ' ' + ' '.join(additions) + '>', raw)
                    self.images += 1
        self.parts.append(raw)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        self.parts.append(f'</{tag}>')

    def handle_data(self, text):
        self.text_length += len(text.strip())
        self.parts.append(text)

    def handle_entityref(self, name):
        self.parts.append(f'&{name};')

    def handle_charref(self, name):
        self.parts.append(f'&#{name};')

    def handle_comment(self, text):
        self.parts.append(f'<!--{text}-->')


def prepare_content(content, page, config):
    page.meta['has_math'] = bool(re.search(r'class="arithmatex"', content))
    parser = ImageMarkup(page, config)
    parser.feed(content)
    return ''.join(parser.parts)
