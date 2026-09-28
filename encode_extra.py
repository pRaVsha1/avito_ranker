"""Дополнительные локальные эмбеддинги: E5 base и короткий текст объявления.

Загрузка весов отделена от вычислений. encode читает только локальные файлы;
предсказание использует сохранённые numpy-массивы и не требует PyTorch/GPU.
"""

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd

BASE_REVISION = 'd128750597153bb5987e10b1c3493a34e5a4502a'


def download(out):
    from huggingface_hub import snapshot_download

    snapshot_download(
        'intfloat/multilingual-e5-base',
        revision=BASE_REVISION,
        local_dir=out / 'e5-base',
        allow_patterns=[
            'README.md',
            'config.json',
            'model.safetensors',
            'tokenizer.json',
            'tokenizer_config.json',
            'special_tokens_map.json',
            'sentencepiece.bpe.model',
        ],
    )


def encode(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    out = Path(args.artifacts_dir)
    torch.set_num_threads(2)
    model_dir = out / f'e5-{args.model}'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            model_dir, local_files_only=True, attn_implementation='eager'
        )
        .eval()
        .to(f'cuda:{args.gpu}')
    )
    corpus = pd.read_parquet(out / 'corpus.parquet')
    queries = pd.read_parquet(out / 'queries_all.parquet')
    short = (
        'passage: '
        + corpus.item_title_raw.fillna('')
        + '. '
        + corpus.item_infm_params_text.fillna('').str.slice(0, 350)
    )
    text = {
        'items_title': short.tolist(),
        'items': (
            short + '. ' + corpus.item_description_raw.fillna('').str.slice(0, 2000)
        ).tolist(),
        'queries': (
            'query: '
            + queries.search_query.fillna('')
            + '. '
            + queries.search_infm_params_text.fillna('')
        ).tolist(),
        'queries_text': ('query: ' + queries.search_query.fillna('')).tolist(),
    }
    kinds = (
        ['items_title']
        if args.model == 'small'
        else (
            ['items', 'queries', 'queries_text']
            if args.model == 'domain'
            else ['items', 'items_title', 'queries', 'queries_text']
        )
    )
    for kind in kinds:
        path = out / f'emb_{args.model}_{kind}_{args.shard}.npz'
        if path.exists():
            print('Already encoded', path, flush=True)
            continue
        positions = np.arange(args.shard, len(text[kind]), args.shards)
        vectors = np.empty((len(positions), model.config.hidden_size), np.float32)
        started = time.monotonic()
        for offset in range(0, len(positions), args.batch_size):
            selected = positions[offset : offset + args.batch_size]
            batch = tokenizer(
                [text[kind][i] for i in selected],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256,
            ).to(f'cuda:{args.gpu}')
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1)
                vectors[offset : offset + len(selected)] = (
                    torch.nn.functional.normalize(
                        (hidden * mask).sum(1) / mask.sum(1), p=2, dim=1
                    )
                    .cpu()
                    .numpy()
                )
            if offset % (args.batch_size * 100) == 0:
                print(
                    time.strftime('%H:%M:%S'),
                    args.model,
                    kind,
                    offset,
                    len(positions),
                    'seconds',
                    round(time.monotonic() - started),
                    flush=True,
                )
        np.savez(path, positions=positions, embeddings=vectors)
        print('Saved', path, flush=True)


def merge(out, model, shards):
    kinds = (
        ['items_title']
        if model == 'small'
        else (
            ['items', 'queries', 'queries_text']
            if model == 'domain'
            else ['items', 'items_title', 'queries', 'queries_text']
        )
    )
    for kind in kinds:
        frame = 'corpus' if kind.startswith('items') else 'queries_all'
        n = len(
            pd.read_parquet(
                out / f'{frame}.parquet',
                columns=['item_id' if frame == 'corpus' else 'query_id'],
            )
        )
        vectors = None
        seen = np.zeros(n, bool)
        for shard in range(shards):
            saved = np.load(out / f'emb_{model}_{kind}_{shard}.npz')
            if vectors is None:
                vectors = np.empty((n, saved['embeddings'].shape[1]), np.float32)
            assert not seen[saved['positions']].any()
            seen[saved['positions']] = True
            vectors[saved['positions']] = saved['embeddings']
        assert seen.all()
        np.save(out / f'emb_{model}_{kind}.npy', vectors)
    (out / f'emb_{model}_READY').write_text('All merged arrays have been written.\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['download', 'encode', 'merge'])
    p.add_argument('--artifacts-dir', default='artifacts')
    p.add_argument('--model', choices=['small', 'base', 'domain'], default='base')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    if args.command == 'download':
        download(Path(args.artifacts_dir))
    elif args.command == 'merge':
        merge(Path(args.artifacts_dir), args.model, args.shards)
    else:
        encode(args)
