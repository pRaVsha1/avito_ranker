"""Однократная загрузка проверенного релиза; предсказание не использует сеть.

Манифест хранится в Git рядом с кодом. Проверяем каждую часть, собранный ZIP
и каждый извлечённый файл, чтобы нельзя было незаметно смешать версии моделей.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import time
import urllib.request
import zipfile

import certifi

CHUNK = 8 * 1024 * 1024


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def matches(path, entry):
    return (
        path.is_file()
        and path.stat().st_size == entry['bytes']
        and sha256(path) == entry['sha256']
    )


def download(url, target, entry):
    """Сохраняем проверенный файл атомарно; сетевой обрыв допускает докачку."""
    if matches(target, entry):
        print('Уже проверено:', target.name, flush=True)
        return
    partial = target.with_name(target.name + '.partial')
    context = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == entry['bytes'] and matches(partial, entry):
            partial.replace(target)
            return
        if offset >= entry['bytes']:
            partial.unlink()
            offset = 0
        headers = {'User-Agent': 'avito-ranker-artifact-downloader'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(
                request, context=context, timeout=120
            ) as incoming:
                resumed = offset > 0 and incoming.status == 206
                if resumed and not incoming.headers.get('Content-Range', '').startswith(
                    f'bytes {offset}-'
                ):
                    raise ValueError('Сервер вернул неверный диапазон докачки.')
                size = offset if resumed else 0
                reported = size
                with partial.open('ab' if resumed else 'wb') as outgoing:
                    while block := incoming.read(CHUNK):
                        outgoing.write(block)
                        size += len(block)
                        if size - reported >= 256 * 1024 * 1024:
                            print(
                                f'{target.name}: {size / entry["bytes"]:.0%}',
                                flush=True,
                            )
                            reported = size
            if not matches(partial, entry):
                partial.unlink(missing_ok=True)
                raise ValueError(f'Неверная контрольная сумма: {target.name}')
            partial.replace(target)
            print('Загружено и проверено:', target.name, flush=True)
            return
        except (OSError, ValueError) as error:
            if attempt == 2:
                raise
            print(f'Повтор загрузки после ошибки: {error}', flush=True)
            time.sleep(2)


def extract_verified(archive_path, destination, files):
    """Извлекаем только перечисленные data/artifacts, проверяя пути и SHA-256."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(files):
            raise ValueError('Состав ZIP отличается от манифеста.')
        for name in names:
            relative = Path(name)
            target = (destination / relative).resolve()
            if (
                relative.is_absolute()
                or '..' in relative.parts
                or relative.parts[0] not in {'data', 'artifacts'}
                or not target.is_relative_to(destination)
            ):
                raise ValueError(f'Недопустимый путь в архиве: {name}')
            entry = files[name]
            if matches(target, entry):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.extracting')
            try:
                with archive.open(name) as incoming, temporary.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, CHUNK)
                if not matches(temporary, entry):
                    raise ValueError(f'Неверная контрольная сумма файла: {name}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path(__file__).with_name('release_manifest.json'),
    )
    parser.add_argument('--destination', type=Path, default=Path('.'))
    parser.add_argument('--archive-dir', type=Path, default=Path('.downloads'))
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Проверить распакованные файлы без сети',
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    missing = [
        name
        for name, entry in manifest['files'].items()
        if not matches(args.destination / name, entry)
    ]
    if not missing:
        print(f'Все {len(manifest["files"])} файлов уже на месте; SHA-256 совпадают.')
        return
    if args.verify_only:
        raise SystemExit('Отсутствуют или отличаются файлы: ' + ', '.join(missing))
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    for part in manifest['parts']:
        download(part['url'], args.archive_dir / part['name'], part)
    archive = args.archive_dir / manifest['archive']['name']
    if not matches(archive, manifest['archive']):
        temporary = archive.with_suffix('.assembling')
        with temporary.open('wb') as outgoing:
            for part in manifest['parts']:
                with (args.archive_dir / part['name']).open('rb') as incoming:
                    shutil.copyfileobj(incoming, outgoing, CHUNK)
        if not matches(temporary, manifest['archive']):
            raise ValueError('Неверная контрольная сумма собранного архива.')
        temporary.replace(archive)
    extract_verified(archive, args.destination, manifest['files'])
    print(
        f'Готово: {len(manifest["files"])} проверенных файлов. Дальнейший запуск работает офлайн.'
    )


if __name__ == '__main__':
    main()
