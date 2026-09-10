"""Import the reviewed source set, without modifying either source directory.

Usage: python scripts/import_notes.py --obsidian PATH --papers PATH
Existing destination files are only accepted when their content is identical.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obsidian', type=Path, required=True)
    parser.add_argument('--papers', type=Path, required=True)
    args = parser.parse_args()
    obs, paper = args.obsidian, args.papers
    records, planned = [], {}

    def stage(source, destination, data=None):
        raw = source.read_bytes()
        def clean(text):
            return '\n'.join(line.rstrip() for line in text.splitlines()).rstrip() + '\n'
        payload = raw if data is None else clean(data).encode('utf-8')
        target = DOCS / destination
        if target.exists() and target.read_bytes() != payload:
            if data is None or clean(target.read_text(encoding='utf-8')).encode('utf-8') != payload:
                raise SystemExit(f'Refusing to overwrite edited file: {target}')
        if target in planned and planned[target] != payload:
            raise SystemExit(f'Conflicting sources for {target}')
        planned[target] = payload
        base = obs if source.is_relative_to(obs) else paper
        records.append({'source': ('obsidian/' if base == obs else 'paper/') + source.relative_to(base).as_posix(),
                        'destination': destination, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                        'output_sha256': hashlib.sha256(payload).hexdigest()})

    common = {'ALOHA Unleashed.md': 'aloha', 'DP3_科研汇报深度阅读.md': 'dp3',
              'Octo_科研汇报深度阅读.md': 'octo', 'OpenVLA_科研汇报深度阅读.md': 'openvla',
              'PI0_科研汇报深度阅读.md': 'pi0'}
    for name, slug in common.items():
        first, second = obs / '01_Robitic' / name, paper / name
        if first.read_bytes() != second.read_bytes():
            raise SystemExit(f'Duplicate versions differ; review before import: {name}')
        content = first.read_text(encoding='utf-8')
        content = content.replace('![](pipeline.png)', '![ALOHA 策略流程](assets/aloha-pipeline.png)')
        content = content.replace('![](assignment.png)', '![ALOHA 任务示意](assets/aloha-assignment.png)')
        content += f'\n\n---\n\n[阅读论文 PDF](../assets/papers/{slug}.pdf) · [返回论文阅读](index.md)\n'
        stage(first, f'papers/{slug}.md', content)
        stage(second, f'papers/{slug}.md', content)
    for name in ('pipeline', 'assignment'):
        stage(obs / '01_Robitic' / f'{name}.png', f'papers/assets/aloha-{name}.png')
    source = obs / '01_Robitic' / 'SPOT.md'
    content = '# SPOT：概念与阅读随记\n\n' + source.read_text(encoding='utf-8')
    content = re.sub(r'^-(?=\S)', '- ', content, flags=re.M)
    content = content.replace('![](Pasted%20image%2020260907193739.png)', '![SPOT 方法示意](assets/spot-notes.png)')
    content += '\n\n[继续阅读 SPOT 深度笔记](spot.md)\n'
    stage(source, 'papers/spot-notes.md', content)
    stage(obs / '01_Robitic' / 'Pasted image 20260907193739.png', 'papers/assets/spot-notes.png')
    for folder, slug, title in [('2025 SPOT (ICRA)', 'spot', 'SPOT：从人类演示学习物体位姿轨迹'),
                                 ('2026 MEM (arXiv)', 'mem', 'MEM：面向长时程任务的机器人记忆')]:
        source = paper / folder / (folder + '.md')
        content = source.read_text(encoding='utf-8')
        content = content.split('\n', 1)[1]
        # Exported reports used several H1s; keep one page title and a nested outline.
        content = re.sub(r'^(#{1,5}) ', r'\1# ', content, flags=re.M)
        content = '# ' + title + '\n' + content
        for image in sorted((paper / folder / '图片和附件').glob('*.png')):
            dest = f'assets/{slug}-{image.stem.replace(" ", "-")}.png'
            content = content.replace('图片和附件/' + image.name.replace(' ', '%20'), dest)
            stage(image, 'papers/' + dest)
        stage(source, f'papers/{slug}.md', content)
    dates = [('03_weekly', '2026.9.4', '2026-09-04', None),
             ('04_daily', '2026.9.7', '2026-09-07', '论文速记与 WSL 网络配置'),
             ('04_daily', '2026.9.8', '2026-09-08', 'RoboSPA 与 Snipaste 自启排障'),
             ('04_daily', '2026.9.10', '2026-09-10', None)]
    for folder, filename, date, title in dates:
        source = obs / folder / (filename + '.md')
        content = source.read_text(encoding='utf-8')
        if title:
            if date == '2026-09-08':
                content = content.replace('### 有意思的文章', '## 有意思的文章').replace('#### RoboSPA', '### RoboSPA')
                content = content.replace('### 碰到的问题及解决方案\n', '')
                content = content.replace('# Snipaste 开机自启失效排查与修复记录', '## Snipaste 开机自启失效排查与修复记录')
                content = re.sub(r'^## (?!有意思|Snipaste)', '### ', content, flags=re.M)
                content = content.replace(r'C:\Users\28177\Documents\Codex\2026-09-08\wei\outputs', r'%USERPROFILE%\Documents\Snipaste-Fix\outputs（示例路径）')
            else:
                content = re.sub(r'^(#{3,4}) ', lambda m: m[1][1:] + ' ', content, flags=re.M)
            content = f'# {title}\n\n' + content
        stage(source, f'journal/{date}.md', f'---\ndate: {date}\n---\n\n' + content)
    source = obs / '02_CV' / 'CV发展.md'
    content = '# 计算机视觉学习提纲\n\n> 这是一份待展开的学习清单，目前保留关键词，后续逐项补充阅读与实现。\n\n'
    content += '\n'.join('- ' + line for line in source.read_text(encoding='utf-8').splitlines() if line.strip()) + '\n'
    stage(source, 'deep-learning/cv-roadmap.md', content)
    pdfs = [(next((obs / '01_Robitic').glob('*.pdf')), 'aloha', 'ALOHA Unleashed')]
    for prefix, slug, label in [('Black', 'pi0', 'π₀'), ('Kim', 'openvla', 'OpenVLA'), ('Team', 'octo', 'Octo'), ('Ze', 'dp3', '3D Diffusion Policy')]:
        pdfs.append((next(paper.glob(prefix + '*.pdf')), slug, label))
    advanced = [('1-', 'fish', 'Teach a Robot to FISH'), ('2-', 'pi05', 'π₀.₅'), ('3-', 'pi06', 'π₀.₆ Model Card'),
                ('4-', 'vla-rl', 'VLA-RL'), ('5-', 'action-to-action', 'Action-to-Action Flow Matching'),
                ('6-', 'rdt-1b', 'RDT-1B'), ('8-', 'pld', 'PLD')]
    for prefix, slug, label in advanced:
        pdfs.append((next((paper / 'advanced').glob(prefix + '*.pdf')), slug, label))
    library = '# 论文资料库\n\n整理阅读时使用的论文 PDF。原文版权归各作者及出版方所有；笔记是个人理解，引用时请以论文原文为准。\n\n'
    library += '| 论文 | 阅读笔记 | PDF |\n| --- | --- | --- |\n'
    for source, slug, label in pdfs:
        stage(source, f'assets/papers/{slug}.pdf')
        note = f'[阅读笔记]({slug}.md)' if slug in common.values() else '待整理'
        library += f'| {label} | {note} | [PDF · {source.stat().st_size / 1024**2:.1f} MB](../assets/papers/{slug}.pdf) |\n'
    planned[DOCS / 'papers/library.md'] = library.encode('utf-8')
    # Only write after every conflict check has passed.
    for target, payload in planned.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    report = {'notes': 13, 'duplicate_notes': 5, 'images': 9, 'pdfs': 12,
              'excluded': ['.obsidian application configuration'], 'files': records}
    (ROOT / 'migration-manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Imported 13 unique notes, 9 images, 12 PDFs; merged 5 identical notes. {len(planned)} destination files.')


if __name__ == '__main__':
    main()
