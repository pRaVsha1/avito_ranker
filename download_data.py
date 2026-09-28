"""Загрузка исходных данных с проверкой SHA-256; обучение не использует сеть."""

import argparse
import hashlib
import json
import ssl
import certifi
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

URL = "https://disk.yandex.ru/d/sNhfo0YOjGtufg"
SHA256 = "8dd3cba59201bae333c11db89c5111198fa10cc52a70c70bdc57bd6a248fb777"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    folder = Path(args.data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "dataset.zip"
    if not archive.exists():
        endpoint = "https://cloud-api.yandex.net/v1/disk/public/resources?"
        with urllib.request.urlopen(
            endpoint + urllib.parse.urlencode({"public_key": URL}),
            timeout=60,
            context=SSL_CONTEXT,
        ) as response:
            metadata = json.load(response)
        entry = next(
            x for x in metadata["_embedded"]["items"] if x["name"] == "dataset.zip"
        )
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(
            entry["file"], timeout=120, context=SSL_CONTEXT
        ) as response, temporary.open("wb") as output:
            total = 0
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                total += len(chunk)
                print(
                    f"download {total / 1e6:.1f}/{entry['size'] / 1e6:.1f} MB",
                    flush=True,
                )
        temporary.replace(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError(f"Archive checksum mismatch: {digest}")
    with zipfile.ZipFile(archive) as source:
        for name in [
            "train.parquet",
            "benchmark_queries.parquet",
            "benchmark_items.parquet",
        ]:
            target = folder / name
            if not target.exists():
                with source.open(name) as incoming, target.open("wb") as outgoing:
                    while chunk := incoming.read(8 * 1024 * 1024):
                        outgoing.write(chunk)
            print(name, target.stat().st_size, flush=True)
    print("SHA256 verified", digest, flush=True)


if __name__ == "__main__":
    main()
