"""Сравнение рангового смешивания и квот по dev без использования test.

Сравниваются только общие стратегии для всех запросов. Старый подтверждённый
ранжировщик служит дополнительной моделью, а не таблицей готовых ответов.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoost
from solution import (
    read_pickle,
    baseline_score,
    blend_scores,
    top_indices,
    dump_json,
    log,
)
from advanced import AdvancedRetriever, fusion_score, records


def rank_score(score):
    rank = np.empty(len(score), np.float32)
    rank[np.argsort(-score, kind='stable')] = np.arange(1, len(score) + 1)
    return 60 / (60 + rank)


class Components:
    def __init__(self, out, sources, names):
        self.out, self.sources, self.names = Path(out), sources, names
        self.legacy = json.loads((self.out / 'config_dense_meta.json').read_text())
        self.legacy_model = CatBoost().load_model(
            str(self.out / 'ranker_dense_meta.cbm')
        )
        self.legacy_indices = [names.index(n) for n in self.legacy['feature_names']]
        self.models = {}
        families = [(sources, '')] + (
            [('base', 'base_')] if sources == 'domain' else []
        )
        for family, prefix in families:
            for loss in ['pair', 'pair_d8', 'logloss', 'logloss_d8']:
                path = self.out / f'advanced_{family}_{loss}.json'
                if path.exists():
                    config = json.loads(path.read_text())
                    indices = [names.index(n) for n in config['feature_names']]
                    self.models[prefix + loss] = (
                        config,
                        CatBoost().load_model(
                            str(self.out / f'advanced_{family}_{loss}.cbm')
                        ),
                        indices,
                    )

    def score(self, x):
        old = x[:, self.legacy_indices].copy()
        if (
            'geo_center_inferred' in self.names
            and x[0, self.names.index('geo_center_inferred')] > 0
        ):
            old[:, self.legacy['feature_names'].index('log_distance')] = np.nan
        base = baseline_score(
            old, self.legacy['feature_names'], **self.legacy['baseline']
        )
        raw = self.legacy_model.predict(
            old, prediction_type='RawFormulaVal', ntree_end=self.legacy['trees']
        )
        result = {
            'legacy': blend_scores(raw, base, self.legacy['blend_weight']),
            'rrf': base,
        }
        for name, (config, model, indices) in self.models.items():
            view = x[:, indices]
            raw = model.predict(
                view, prediction_type='RawFormulaVal', ntree_end=config['trees']
            )
            fusion = fusion_score(view, config['feature_names'], **config['fusion'])
            result[name] = blend_scores(raw, fusion, config['blend_weight'])
            result[name + '_raw'] = raw
        return result


def select(scores, configuration):
    if configuration['method'] == 'rank':
        combined = sum(
            weight * rank_score(scores[name])
            for name, weight in configuration['weights'].items()
        )
        return top_indices(combined, 50)
    # Квота сохраняет сильные уникальные рекомендации первого источника,
    # затем второй источник заполняет оставшиеся места без повторов.
    first = top_indices(scores[configuration['first']], configuration['quota']).tolist()
    selected = set(first)
    for i in top_indices(scores[configuration['second']], 50):
        if i not in selected:
            first.append(int(i))
            selected.add(int(i))
        if len(first) == 50:
            break
    return np.asarray(first, np.int32)


def tune(out, sources, exclude=None):
    saved = records(out, sources, 'dev')
    components = Components(out, sources, saved['feature_names'])
    scores = [components.score(r['x']) for r in saved['records']]
    # Компонент можно исключить целиком по проверке воспроизводимости.
    # Это общая конфигурация для всех запросов, без правил по query_id.
    excluded = set(exclude or [])
    for values in scores:
        for name in excluded:
            values.pop(name, None)
            values.pop(name + '_raw', None)
    candidates = []
    available = list(scores[0])
    for name in available:
        candidates.append(dict(method='rank', weights={name: 1.0}))
    for new in [
        n for n in ['pair', 'pair_d8', 'logloss', 'logloss_d8'] if n in available
    ]:
        for old in ['legacy', 'rrf']:
            for weight in [0.25, 0.5, 0.75]:
                candidates.append(
                    dict(method='rank', weights={new: weight, old: 1 - weight})
                )
            for quota in [10, 20, 30, 40]:
                candidates.append(
                    dict(method='quota', first=new, second=old, quota=quota)
                )
                candidates.append(
                    dict(method='quota', first=old, second=new, quota=quota)
                )
    if 'pair' in available and 'logloss' in available:
        for weight in [0.25, 0.5, 0.75]:
            candidates.append(
                dict(method='rank', weights={'pair': weight, 'logloss': 1 - weight})
            )
        candidates.append(
            dict(
                method='rank', weights={'pair': 0.375, 'logloss': 0.375, 'legacy': 0.25}
            )
        )
    if 'base_logloss' in available and 'logloss' in available:
        for weight in [0.25, 0.5, 0.75]:
            candidates.append(
                dict(
                    method='rank',
                    weights={'logloss': weight, 'base_logloss': 1 - weight},
                )
            )
        candidates.append(
            dict(
                method='rank',
                weights={'logloss': 0.5, 'base_logloss': 0.25, 'legacy': 0.25},
            )
        )
    if 'logloss_d8' in available and 'logloss' in available:
        for weight in [0.25, 0.5, 0.75]:
            candidates.append(
                dict(
                    method='rank', weights={'logloss_d8': weight, 'logloss': 1 - weight}
                )
            )
    if 'pair_d8' in available and 'pair' in available:
        for weight in [0.25, 0.5, 0.75]:
            candidates.append(
                dict(method='rank', weights={'pair_d8': weight, 'pair': 1 - weight})
            )
    if 'base_pair_d8' in available and 'logloss' in available:
        for weight in [0.25, 0.5, 0.75]:
            candidates.append(
                dict(
                    method='rank',
                    weights={'base_pair_d8': weight, 'logloss': 1 - weight},
                )
            )
    if 'base_logloss_d8' in available and 'logloss' in available:
        for weight in [0.25, 0.5, 0.75]:
            candidates.append(
                dict(
                    method='rank',
                    weights={'base_logloss_d8': weight, 'logloss': 1 - weight},
                )
            )
    for candidate in candidates:
        values = [
            r['y'][select(s, candidate)].sum() / max(r['n_relevant'], 1)
            for r, s in zip(saved['records'], scores)
        ]
        candidate['recall50'] = float(np.mean(values))
    best = max(candidates, key=lambda p: p['recall50'])
    config = dict(
        sources=sources,
        feature_names=saved['feature_names'],
        selection=best,
        results=candidates,
        excluded_components=sorted(excluded),
        candidate_recall=float(
            np.mean([r['y'].sum() / max(r['n_relevant'], 1) for r in saved['records']])
        ),
    )
    dump_json(out / f'ensemble_{sources}.json', config)
    log('ENSEMBLE', sources, best)


def predict(out, sources, output, split, full):
    if full and split != 'benchmark':
        raise ValueError('Полная история не используется на отложенных запросах.')
    config = json.loads((out / f'ensemble_{sources}.json').read_text())
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
    components = Components(out, sources, names)
    chosen = [select(components.score(r['x']), config['selection']) for r in rows]
    if split == 'benchmark':
        answers = [
            (r['query_id'], ' '.join(sorted(retriever.ids[r['indices'][indices]])))
            for r, indices in zip(rows, chosen)
        ]
        pd.DataFrame(answers, columns=['query_id', 'answer']).to_csv(
            output, index=False, encoding='utf-8', lineterminator='\n'
        )
    else:
        metric = float(
            np.mean(
                [
                    r['y'][indices].sum() / max(r['n_relevant'], 1)
                    for r, indices in zip(rows, chosen)
                ]
            )
        )
        ceiling = float(np.mean([r['y'].sum() / max(r['n_relevant'], 1) for r in rows]))
        dump_json(
            output,
            dict(
                split=split,
                recall50=metric,
                candidate_recall=ceiling,
                queries=len(rows),
            ),
        )
    log('saved ensemble', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['tune', 'predict'])
    p.add_argument('--artifacts-dir', default='artifacts')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--sources', choices=['title', 'base', 'domain'], default='title')
    p.add_argument('--output', default='answer_ensemble.csv')
    p.add_argument('--split', choices=['benchmark', 'dev', 'test'], default='benchmark')
    p.add_argument('--full', action='store_true')
    p.add_argument('--exclude-component', action='append', default=[])
    a = p.parse_args()
    out = Path(a.artifacts_dir)
    if a.command == 'tune':
        tune(out, a.sources, a.exclude_component)
    else:
        predict(out, a.sources, a.output, a.split, a.full)
