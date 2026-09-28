"""Разбор только отложенных dev-ошибок; скрытая разметка не используется."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from solution import read_pickle, top_indices, baseline_score, normalize, blend_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifacts-dir', default='artifacts')
    p.add_argument('--dense', action='store_true')
    p.add_argument('--metadata', action='store_true')
    args = p.parse_args()
    out = Path(args.artifacts_dir)
    kind = ('dense' if args.dense else 'lexical') + ('_meta' if args.metadata else '')
    config = json.loads((out / f'config_{kind}.json').read_text())
    stored = read_pickle(out / f'candidates_dev_{kind}.pkl')
    corpus = pd.read_parquet(out / 'corpus.parquet')
    queries = pd.read_parquet(out / 'queries_dev.parquet').set_index('query_id')
    by_id = corpus.set_index('item_id')
    if config['use_ranker']:
        from catboost import CatBoostRanker

        model = CatBoostRanker().load_model(str(out / f'ranker_{kind}.cbm'))
        score = lambda x: blend_scores(
            model.predict(x, ntree_end=config['trees']),
            baseline_score(x, stored['feature_names'], **config['baseline']),
            config.get('blend_weight', 1.0),
        )
    else:
        score = lambda x: baseline_score(
            x, stored['feature_names'], **config['baseline']
        )
    errors = []
    per_query = []
    for r in stored['records']:
        chosen = top_indices(score(r['x']), 50)
        selected = corpus.iloc[r['indices'][chosen]]
        candidates = set(corpus.iloc[r['indices']].item_id)
        q = queries.loc[r['query_id']]
        relevant = set(q.relevant)
        picked = set(selected.item_id)
        per_query.append(
            {
                'query_id': r['query_id'],
                'recall50': len(picked & relevant) / len(relevant),
                'candidate_recall': len(candidates & relevant) / len(relevant),
                'query_words': len(normalize(q.search_query).split()),
                'has_filters': bool(q.search_infm_params_text),
            }
        )
        for item in sorted(relevant - picked):
            truth = by_id.loc[item]
            errors.append(
                {
                    'query': q.search_query,
                    'filters': q.search_infm_params_text,
                    'error_type': 'selection' if item in candidates else 'retrieval',
                    'relevant_title': truth.item_title_raw,
                    'same_location': truth.item_location_id == q.search_location_id,
                    'predicted_titles': ' | '.join(selected.item_title_raw.head(3)),
                }
            )
    frame = pd.DataFrame(errors)
    frame.to_csv(out / 'dev_errors.csv', index=False)
    pd.DataFrame(per_query).to_csv(out / f'dev_per_query_{kind}.csv', index=False)
    print(frame.groupby(['error_type', 'same_location']).size().to_string())
    print(frame.head(12).to_json(orient='records', force_ascii=False))


if __name__ == '__main__':
    main()
