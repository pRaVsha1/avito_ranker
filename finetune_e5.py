"""Ограниченное дообучение E5-small на парах разрешённой обучающей истории.

Используется контрастивная функция с несколькими позитивами: известные
положительные объявления не становятся отрицательными примерами в батче.
Dev, test и тексты обучения ранжировщика полностью исключены из этих пар.
"""

import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
from solution import normalize, dump_json, SEED, log


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--artifacts-dir', default='artifacts')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--max-seconds', type=int, default=900)
    a = p.parse_args()
    out = Path(a.artifacts_dir)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    allowed = set(
        pd.read_parquet(
            out / 'history_train.parquet', columns=['search_query']
        ).search_query.map(normalize)
    )
    for split in ['train', 'dev', 'test']:
        excluded = set(
            pd.read_parquet(
                out / f'queries_{split}.parquet', columns=['search_query']
            ).search_query.map(normalize)
        )
        assert not allowed & excluded
    frame = pd.read_parquet(
        Path(a.data_dir) / 'train.parquet',
        columns=[
            'search_query',
            'search_infm_params_text',
            'item_id',
            'item_title_raw',
            'item_infm_params_text',
            'item_description_raw',
        ],
    )
    frame['normalized_query'] = frame.search_query.map(normalize)
    frame = (
        frame[frame.normalized_query.isin(allowed)]
        .drop_duplicates(['normalized_query', 'search_infm_params_text', 'item_id'])
        .reset_index(drop=True)
    )
    groups = list(frame.groupby('normalized_query', sort=True).indices.values())
    known = {q: set(ids) for q, ids in frame.groupby('normalized_query').item_id}
    queries = (
        'query: '
        + frame.search_query.fillna('')
        + '. '
        + frame.search_infm_params_text.fillna('')
    ).to_numpy()
    passages = (
        'passage: '
        + frame.item_title_raw.fillna('')
        + '. '
        + frame.item_infm_params_text.fillna('').str.slice(0, 350)
        + '. '
        + frame.item_description_raw.fillna('').str.slice(0, 2000)
    ).to_numpy()
    query_names = frame.normalized_query.to_numpy()
    item_ids = frame.item_id.to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(out / 'e5-small', local_files_only=True)
    model = AutoModel.from_pretrained(
        out / 'e5-small', local_files_only=True, attn_implementation='eager'
    ).to(f'cuda:{a.gpu}')
    # Словарь фиксируем: обучаются контекстные слои, что уменьшает память
    # оптимизатора и риск потери редких языковых представлений.
    for parameter in model.embeddings.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.01
    )
    rng = np.random.default_rng(SEED)
    started = time.monotonic()
    history = []

    def embed(texts, max_length):
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        ).to(f'cuda:{a.gpu}')
        hidden = model(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        return torch.nn.functional.normalize(
            (hidden * mask).sum(1) / mask.sum(1), p=2, dim=1
        )

    model.train()
    for step in range(a.steps):
        selected_groups = rng.choice(len(groups), size=a.batch_size, replace=False)
        selected = np.array([rng.choice(groups[i]) for i in selected_groups])
        q = embed(queries[selected].tolist(), 64)
        d = embed(passages[selected].tolist(), 192)
        logits = q @ d.T / 0.02
        positives = torch.tensor(
            [
                [item_ids[j] in known[query_names[i]] for j in selected]
                for i in selected
            ],
            device=logits.device,
        )
        assert positives.diagonal().all()
        positive_logits = logits.masked_fill(~positives, -torch.inf)
        loss = 0.5 * (
            (torch.logsumexp(logits, 1) - torch.logsumexp(positive_logits, 1)).mean()
            + (torch.logsumexp(logits, 0) - torch.logsumexp(positive_logits, 0)).mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Короткий прогрев, затем плавное уменьшение шага.
        scale = min((step + 1) / 80, 1.0) * max(1 - (step + 1) / a.steps, 0.05)
        optimizer.param_groups[0]['lr'] = 1e-5 * scale
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0:
            record = dict(
                step=step + 1,
                loss=float(loss.detach()),
                seconds=round(time.monotonic() - started, 1),
            )
            history.append(record)
            log('domain fine-tune', record)
        if time.monotonic() - started > a.max_seconds:
            break
    destination = out / 'e5-domain'
    model.eval().save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    dump_json(
        destination / 'training_config.json',
        dict(
            seed=SEED,
            steps_completed=step + 1,
            batch_size=a.batch_size,
            source_model='intfloat/multilingual-e5-small',
            source_revision='614241f622f53c4eeff9890bdc4f31cfecc418b3',
            learning_rate=1e-5,
            temperature=0.02,
            query_tokens=64,
            passage_tokens=192,
            embedding_layer_frozen=True,
            pairs=len(frame),
            query_groups=len(groups),
            history=history,
        ),
    )
    log('domain model saved', destination)


if __name__ == '__main__':
    main()
