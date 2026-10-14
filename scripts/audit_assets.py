"""Measure source assets and lossless PNG potential without altering images."""
import json
from pathlib import Path
import struct
import sys
import zlib


def audit(root=Path('docs')):
    assets = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in
              {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'}]
    entries = [{'path': p.as_posix(), 'bytes': p.stat().st_size} for p in assets]
    entries.sort(key=lambda r: r['bytes'], reverse=True)
    png_samples = []
    for p in sorted((p for p in assets if p.suffix.lower() == '.png'), key=lambda p: p.stat().st_size, reverse=True)[:3]:
        raw, pos, chunks = p.read_bytes(), 8, []
        while pos + 12 <= len(raw):
            length = struct.unpack('>I', raw[pos:pos + 4])[0]
            if raw[pos + 4:pos + 8] == b'IDAT':
                chunks.append(raw[pos + 8:pos + 8 + length])
            pos += length + 12
        compressed = b''.join(chunks)
        recompressed = zlib.compress(zlib.decompress(compressed), level=9)
        png_samples.append({'path': p.as_posix(), 'idat_bytes': len(compressed),
                            'level9_idat_bytes': len(recompressed),
                            'potential_idat_saving_bytes': max(0, len(compressed) - len(recompressed))})
    pdfs = [r for r in entries if r['path'].lower().endswith('.pdf')]
    return {'source_resource_bytes': sum(r['bytes'] for r in entries), 'resource_files': len(entries),
            'pdf_bytes': sum(r['bytes'] for r in pdfs), 'pdf_files': len(pdfs),
            'largest_resources': entries[:10], 'lossless_png_samples': png_samples,
            'notes': ['Source bytes are not page transfer bytes.',
                      'PNG estimate covers IDAT only; original files are unchanged.',
                      'PDFs are downloads, not eagerly embedded page resources.']}


if __name__ == '__main__':
    output = Path(sys.argv[1] if len(sys.argv) > 1 else '.build/resource-audit.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{result["resource_files"]} assets: {result["source_resource_bytes"] / 1024**2:.1f} MiB; '
          f'{result["pdf_files"]} PDFs: {result["pdf_bytes"] / 1024**2:.1f} MiB. Report: {output}')
