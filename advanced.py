"""Вторая итерация поиска: несколько представлений и более сложные негативы.

Все новые варианты выбираются только по dev. Исходный Retriever и подтверждённый
CSV сохранены отдельно; никаких правил для конкретных query_id здесь нет.
"""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from solution import (
    Retriever,
    BM25,
    read_pickle,
    query_key,
    top_indices,
    baseline_score,
    blend_scores,
    recall,
    dump_json,
    normalize,
    tokenize,
    log,
    SEED,
)


class AdvancedRetriever(Retriever):
    def __init__(self, out, sources='title', full=False, benchmark_only=False):
        super().__init__(
            out, full=full, dense=True, metadata=True, benchmark_only=benchmark_only
        )
        self.sources = sources
        self.original_centers = set(self.history['centers'])
        self.title_emb = np.load(self.out / 'emb_small_items_title.npy', mmap_mode='r')
        if sources in ['base', 'domain']:
            from geo_prior import GeoPrior

            self.geo = GeoPrior(out, full=full)
            for loc, center in self.geo.data['centers'].items():
                self.history['centers'].setdefault(loc, center)
            self.radians_lat = np.radians(self.numeric['item_latitude'])
            self.radians_lon = np.radians(self.numeric['item_longitude'])
            self.base_items = np.load(self.out / 'emb_base_items.npy', mmap_mode='r')
            self.base_titles = np.load(
                self.out / 'emb_base_items_title.npy', mmap_mode='r'
            )
            self.base_queries = np.load(
                self.out / 'emb_base_queries.npy', mmap_mode='r'
            )
            self.base_queries_text = np.load(
                self.out / 'emb_base_queries_text.npy', mmap_mode='r'
            )
        if sources == 'domain':
            self.domain_items = np.load(
                self.out / 'emb_domain_items.npy', mmap_mode='r'
            )
            self.domain_queries = np.load(
                self.out / 'emb_domain_queries.npy', mmap_mode='r'
            )
            self.domain_queries_text = np.load(
                self.out / 'emb_domain_queries_text.npy', mmap_mode='r'
            )
        title_index = self.out / 'index_title.pkl'
        if not title_index.exists():
            with title_index.open('wb') as stream:
                pickle.dump(
                    BM25().fit(self.corpus.item_title_raw.fillna('')),
                    stream,
                    protocol=5,
                )
        self.title_bm25 = read_pickle(title_index)

    def retrieve(self, queries, training=False, negatives=250):
        rng = np.random.default_rng(SEED)
        records = []
        for start in range(0, len(queries), 16):
            batch = queries.iloc[start : start + 16]
            texts = batch.search_query.tolist()
            positions = [
                self.query_position[query_key(r)] for r in batch.itertuples(index=False)
            ]
            arrays = {
                'bm25': self.content['bm25'].score(texts),
                'char': (
                    self.content['char'].transform([normalize(t) for t in texts])
                    @ self.content['char_matrix'].T
                )
                .toarray()
                .astype(np.float32),
                'history': self.history['history'].score(texts),
                'dense': np.asarray(self.query_emb[positions] @ self.item_emb.T),
                'dense_text': np.asarray(
                    self.query_text_emb[positions] @ self.item_emb.T
                ),
                'title_bm25': self.title_bm25.score(texts),
                'dense_title': np.asarray(
                    self.query_text_emb[positions] @ self.title_emb.T
                ),
            }
            if self.sources in ['base', 'domain']:
                arrays.update(
                    base=np.asarray(self.base_queries[positions] @ self.base_items.T),
                    base_text=np.asarray(
                        self.base_queries_text[positions] @ self.base_items.T
                    ),
                    base_title=np.asarray(
                        self.base_queries_text[positions] @ self.base_titles.T
                    ),
                )
            if self.sources == 'domain':
                arrays.update(
                    domain=np.asarray(
                        self.domain_queries[positions] @ self.domain_items.T
                    ),
                    domain_text=np.asarray(
                        self.domain_queries_text[positions] @ self.domain_items.T
                    ),
                )
            for bi, row in enumerate(batch.itertuples(index=False)):
                scores = {k: v[bi] for k, v in arrays.items()}
                if self.sources in ['base', 'domain']:
                    center = self.history['centers'].get(row.search_location_id)
                    if center:
                        lat0, lon0 = np.radians(
                            [center['item_latitude'], center['item_longitude']]
                        )
                        h = (
                            np.sin((self.radians_lat - lat0) / 2) ** 2
                            + np.cos(self.radians_lat)
                            * np.cos(lat0)
                            * np.sin((self.radians_lon - lon0) / 2) ** 2
                        )
                        logdistance = np.log1p(
                            12742 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
                        )
                        # Неизвестные координаты не означают далёкую локацию.
                        # Нулевой штраф также не допускает NaN в оценке источника.
                        scores['geo_dense'] = scores[
                            'base_text'
                        ] - 0.025 * np.nan_to_num(logdistance, nan=0.0)
                    else:
                        scores['geo_dense'] = scores['base_text'].copy()
                ranks, found = {}, []
                local = self.allowed & (self.location == row.search_location_id)
                for name, values in scores.items():
                    for suffix, mask in [('global', self.allowed), ('local', local)]:
                        selected = top_indices(values, 200, mask & (values > 0))
                        rank = np.full(len(self.ids), 1e6, np.float32)
                        rank[selected] = np.arange(1, len(selected) + 1)
                        ranks[name + '_' + suffix] = rank
                        found.extend(selected)
                candidates = np.array(sorted(set(found)), np.int32)
                if len(candidates) < 50:
                    candidates = np.union1d(
                        candidates,
                        top_indices(self.history['popularity'], 50, self.allowed),
                    )
                relevant = set(getattr(row, 'relevant', []))
                if training:
                    # Позитивы никогда не берутся как негативы. На dev/test этот
                    # блок не исполняется: правильные ответы не подмешиваются.
                    positive = np.array(
                        [self.id_to_pos[i] for i in sorted(relevant)], np.int32
                    )
                    available = candidates[~np.isin(candidates, positive)]
                    fusion = sum(60 / (60 + r[available]) for r in ranks.values())
                    hard = available[
                        top_indices(fusion, min(int(negatives * 0.8), len(available)))
                    ]
                    rest = np.setdiff1d(available, hard)
                    random = rng.choice(
                        rest, size=min(negatives - len(hard), len(rest)), replace=False
                    )
                    candidates = np.unique(np.r_[positive, hard, random]).astype(
                        np.int32
                    )
                features = self.features(row, candidates, scores, ranks)
                if self.sources in ['base', 'domain']:
                    features = np.column_stack(
                        [
                            features,
                            self.geo.features(
                                row.search_location_id,
                                self.location[candidates],
                                self.numeric['item_microcat_id'][candidates],
                            ),
                        ]
                    )
                    self.feature_names += [
                        'geo_log_probability',
                        'geo_log_lift',
                        'service_locality_rate',
                    ]
                    features = np.column_stack(
                        [
                            features,
                            np.full(
                                len(candidates),
                                row.search_location_id not in self.original_centers,
                                np.float32,
                            ),
                        ]
                    )
                    self.feature_names += ['geo_center_inferred']
                records.append(
                    {
                        'query_id': row.query_id,
                        'indices': candidates,
                        'x': features,
                        'y': np.array(
                            [i in relevant for i in self.ids[candidates]], np.float32
                        ),
                        'n_relevant': len(relevant),
                    }
                )
            if start % 160 == 0:
                log(
                    'advanced',
                    self.sources,
                    start + len(batch),
                    len(queries),
                    'training',
                    training,
                )
        return records


def records(out, sources, split, training=False, negatives=250):
    cache = out / f'advanced_{sources}_{split}.pkl'
    if cache.exists():
        return read_pickle(cache)
    retriever = AdvancedRetriever(
        out, sources=sources, benchmark_only=split == 'benchmark'
    )
    frame = pd.read_parquet(out / f'queries_{split}.parquet')
    result = {
        'records': retriever.retrieve(frame, training=training, negatives=negatives),
        'feature_names': retriever.feature_names,
        'training_negatives': negatives,
    }
    with cache.open('wb') as stream:
        pickle.dump(result, stream, protocol=5)
    return result


def fusion_score(
    x, names, local_weight=1.0, extra_weight=1.0, base_weight=1.0, geo_weight=1.0
):
    score = np.zeros(len(x), np.float32)
    for i, name in enumerate(names):
        if not name.startswith('rrf_'):
            continue
        weight = local_weight if name.endswith('_local') else 1.0
        if name.startswith(('rrf_dense_title_', 'rrf_title_bm25_')):
            weight *= extra_weight
        if name.startswith(('rrf_base', 'rrf_domain')):
            weight *= base_weight
        if name.startswith('rrf_geo_dense_'):
            weight *= geo_weight
        score += weight * x[:, i]
    return score


def fit(out, sources, loss='pair', negatives=250, reuse=False, depth=7, tag=None):
    from catboost import CatBoostRanker, CatBoostClassifier, CatBoost, Pool

    dev = records(out, sources, 'dev')
    names = dev['feature_names']
    grid = []
    for local in [0.5, 1.0, 2.0]:
        for extra in [0.0, 0.5, 1.0]:
            for base in ([0.5, 1.0] if sources in ['base', 'domain'] else [0.0]):
                for geo in (
                    [0.0, 0.5, 1.0] if sources in ['base', 'domain'] else [0.0]
                ):
                    p = dict(
                        local_weight=local,
                        extra_weight=extra,
                        base_weight=base,
                        geo_weight=geo,
                    )
                    metric, ceiling = recall(
                        dev['records'], lambda x: fusion_score(x, names, **p)
                    )
                    grid.append(dict(**p, recall50=metric, candidate_recall=ceiling))
    best = max(grid, key=lambda r: r['recall50'])
    fusion = {
        k: best[k]
        for k in ['local_weight', 'extra_weight', 'base_weight', 'geo_weight']
    }
    log('best fusion', best)

    def pool(rows):
        return Pool(
            np.concatenate([r['x'] for r in rows]),
            label=np.concatenate([r['y'] for r in rows]),
            group_id=np.concatenate(
                [np.full(len(r['x']), i, np.int32) for i, r in enumerate(rows)]
            ),
            feature_names=names,
        )

    # Отдельное имя эксперимента сохраняет уже проверенные веса неизменными.
    model_name = loss + ('_' + tag if tag else '')
    model_path = out / f'advanced_{sources}_{model_name}.cbm'
    if reuse:
        model = CatBoost().load_model(str(model_path))
    else:
        training = records(out, sources, 'train', training=True, negatives=negatives)
        assert training['feature_names'] == names
        params = dict(
            iterations=700,
            depth=depth,
            learning_rate=0.05,
            random_seed=SEED,
            thread_count=12,
            verbose=50,
            allow_writing_files=False,
        )
        if loss == 'pair':
            model = CatBoostRanker(loss_function='PairLogit:max_pairs=1000', **params)
        else:
            model = CatBoostClassifier(
                loss_function='Logloss', class_weights=[1.0, 20.0], **params
            )
        model.fit(
            pool(training['records']),
            eval_set=pool(dev['records']),
            early_stopping_rounds=100,
        )
        model.save_model(str(model_path))
    # Общий класс обеспечивает одинаковый RawFormulaVal-интерфейс для
    # ранжировщика и классификатора (у CatBoostRanker другой predict API).
    model = CatBoost().load_model(str(model_path))
    flat = np.concatenate([r['x'] for r in dev['records']])
    boundaries = np.cumsum([0] + [len(r['x']) for r in dev['records']])
    fused = [fusion_score(r['x'], names, **fusion) for r in dev['records']]
    results = []
    for trees in sorted(
        set(
            min(n, model.tree_count_)
            for n in [25, 50, 75, 100, 150, 200, 300, 400, 500, 700, model.tree_count_]
        )
    ):
        raw = model.predict(flat, prediction_type='RawFormulaVal', ntree_end=trees)
        for weight in [0.25, 0.5, 0.65, 0.8, 1.0]:
            values = []
            for i, r in enumerate(dev['records']):
                chosen = top_indices(
                    blend_scores(
                        raw[boundaries[i] : boundaries[i + 1]], fused[i], weight
                    ),
                    50,
                )
                values.append(r['y'][chosen].sum() / max(r['n_relevant'], 1))
            results.append(
                dict(trees=trees, blend_weight=weight, recall50=float(np.mean(values)))
            )
    winner = max(results, key=lambda r: r['recall50'])
    config = dict(
        sources=sources,
        loss=loss,
        depth=depth,
        tag=tag,
        feature_names=names,
        fusion=fusion,
        grid=grid,
        ranker_results=results,
        candidate_recall=best['candidate_recall'],
        **winner,
    )
    dump_json(out / f'advanced_{sources}_{model_name}.json', config)
    log(
        'ADVANCED WINNER',
        sources,
        model_name,
        winner,
        'candidate recall',
        best['candidate_recall'],
    )


def predict(out, sources, loss, output, split='benchmark', full=False):
    from catboost import CatBoost

    if full and split != 'benchmark':
        raise ValueError('Полная история допустима только для benchmark.')
    config = json.loads((out / f'advanced_{sources}_{loss}.json').read_text())
    model = CatBoost().load_model(str(out / f'advanced_{sources}_{loss}.cbm'))
    if split == 'benchmark':
        retriever = AdvancedRetriever(
            out, sources=sources, full=full, benchmark_only=True
        )
        frame = pd.read_parquet(out / 'queries_benchmark.parquet')
        rows = retriever.retrieve(frame)
        names = retriever.feature_names
    else:
        saved = records(out, sources, split)
        rows, names = saved['records'], saved['feature_names']
    assert names == config['feature_names']

    def score(x):
        raw = model.predict(
            x, prediction_type='RawFormulaVal', ntree_end=config['trees']
        )
        return blend_scores(
            raw, fusion_score(x, names, **config['fusion']), config['blend_weight']
        )

    if split == 'benchmark':
        answers = [
            (
                r['query_id'],
                ' '.join(
                    sorted(retriever.ids[r['indices'][top_indices(score(r['x']), 50)]])
                ),
            )
            for r in rows
        ]
        pd.DataFrame(answers, columns=['query_id', 'answer']).to_csv(
            output, index=False, encoding='utf-8', lineterminator='\n'
        )
    else:
        metric, ceiling = recall(rows, score)
        dump_json(
            output,
            dict(
                split=split,
                recall50=metric,
                candidate_recall=ceiling,
                queries=len(rows),
            ),
        )
    log('saved', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['retrieve', 'train', 'predict'])
    p.add_argument('--data-dir', default='data')
    p.add_argument('--artifacts-dir', default='artifacts')
    p.add_argument('--output', default='answer_advanced.csv')
    p.add_argument('--sources', choices=['title', 'base', 'domain'], default='title')
    p.add_argument(
        '--split', choices=['train', 'dev', 'test', 'benchmark'], default='benchmark'
    )
    p.add_argument('--negatives', type=int, default=250)
    p.add_argument('--loss', choices=['pair', 'logloss'], default='pair')
    p.add_argument('--depth', type=int, default=7)
    p.add_argument(
        '--tag', help='Суффикс отдельного обучающего эксперимента, например d8.'
    )
    p.add_argument('--reuse', action='store_true')
    p.add_argument('--full', action='store_true')
    a = p.parse_args()
    out = Path(a.artifacts_dir)
    if a.command == 'retrieve':
        records(
            out, a.sources, a.split, training=a.split == 'train', negatives=a.negatives
        )
    elif a.command == 'train':
        fit(out, a.sources, a.loss, a.negatives, a.reuse, a.depth, a.tag)
    else:
        predict(out, a.sources, a.loss, a.output, a.split, a.full)
