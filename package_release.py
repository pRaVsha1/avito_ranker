"""Собрать переносимые данные и артефакты для GitHub Release.

Код и notebook хранятся в Git. Этот архив содержит только данные/артефакты,
без временных кэшей кандидатов. Части по 256 МиБ укладываются в лимит GitHub.
"""

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
VERSION = 'v1.0.0'
PART_BYTES = 256 * 1024 * 1024


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def include(path):
    if not path.is_file() or '.cache' in path.parts:
        return False
    if path.name.startswith('candidates_') or (
        path.name.startswith('advanced_') and path.suffix == '.pkl'
    ):
        return False
    if path.name in {'iteration2_timeline.json', 'dev_per_query_dense_meta.csv'}:
        return False
    return path.suffix in {
        '.json',
        '.csv',
        '.png',
        '.parquet',
        '.pkl',
        '.npy',
        '.cbm',
        '.safetensors',
        '.model',
        '.md',
        '.txt',
    }


def main():
    destination = ROOT / 'dist'
    destination.mkdir(exist_ok=True)
    paths = sorted(
        [p for p in (ROOT / 'artifacts').rglob('*') if include(p)]
        + list((ROOT / 'data').glob('*.parquet'))
    )
    files = {
        str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
        for p in paths
    }
    archive_path = destination / f'avito-artifacts-{VERSION}.zip'
    with zipfile.ZipFile(archive_path, 'w', allowZip64=True) as archive:
        for path in paths:
            # Большие уже сжатые таблицы, FP32 и веса не перепаковываем.
            method = (
                zipfile.ZIP_STORED
                if path.suffix in {'.parquet', '.npy', '.safetensors'}
                else zipfile.ZIP_DEFLATED
            )
            archive.write(
                path, str(path.relative_to(ROOT)), compress_type=method, compresslevel=3
            )
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    parts = []
    with archive_path.open('rb') as incoming:
        number = 1
        while True:
            first = incoming.read(8 * 1024 * 1024)
            if not first:
                break
            part = destination / f'{archive_path.name}.chunk{number:02d}'
            with part.open('wb') as outgoing:
                outgoing.write(first)
                remaining = PART_BYTES - len(first)
                while remaining and (
                    block := incoming.read(min(8 * 1024 * 1024, remaining))
                ):
                    outgoing.write(block)
                    remaining -= len(block)
            parts.append(
                {
                    'name': part.name,
                    'bytes': part.stat().st_size,
                    'sha256': sha256(part),
                    'url': f'https://github.com/pRaVsha1/avito_ranker/releases/download/{VERSION}/{part.name}',
                }
            )
            number += 1
    manifest = {
        'version': VERSION,
        'archive': {
            'name': archive_path.name,
            'bytes': archive_path.stat().st_size,
            'sha256': sha256(archive_path),
        },
        'parts': parts,
        'files': files,
    }
    (ROOT / 'release_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    )
    checksums = ''.join(f'{p["sha256"]}  {p["name"]}\n' for p in parts)
    checksums += f'{sha256(ROOT / "answer.csv")}  answer.csv\n'
    (destination / 'SHA256SUMS.txt').write_text(checksums)
    print(
        f'Архив: {archive_path.stat().st_size / 1024**3:.2f} ГиБ, {len(files)} файлов, {len(parts)} частей',
        flush=True,
    )


if __name__ == '__main__':
    main()
