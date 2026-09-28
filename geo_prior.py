"""Географические статистики только из разрешённой истории взаимодействий."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from solution import read_pickle, log


def build_geo(data_dir, artifacts_dir, full=False):
    out = Path(artifacts_dir)
    history = pd.read_parquet(
        out / ('history_full.parquet' if full else 'history_train.parquet'),
        columns=['search_location_id', 'item_id'],
    )
    items = pd.read_parquet(
        Path(data_dir) / 'train.parquet',
        columns=[
            'item_id',
            'item_location_id',
            'item_latitude',
            'item_longitude',
            'item_microcat_id',
        ],
    ).drop_duplicates('item_id')
    for c in ['item_latitude', 'item_longitude']:
        items[c] = pd.to_numeric(items[c], errors='coerce')
    joined = history.merge(items, on='item_id', how='left', validate='many_to_one')
    pair = joined.groupby(['search_location_id', 'item_location_id']).size().to_dict()
    totals = joined.search_location_id.value_counts().to_dict()
    global_prob = joined.item_location_id.value_counts(normalize=True).to_dict()
    joined['local_match'] = (
        joined.search_location_id == joined.item_location_id
    ).astype(float)
    service = joined.groupby('item_microcat_id').local_match.agg(['sum', 'count'])
    overall = float(joined.local_match.mean())
    # Онлайн-услуги и локальные работы имеют разную географию. Сглаживаем
    # долю совпадений по виду услуги, не вводя жёстких ручных правил.
    local_rate = ((service['sum'] + 50 * overall) / (service['count'] + 50)).to_dict()
    centers = (
        joined.groupby('search_location_id')[['item_latitude', 'item_longitude']]
        .median()
        .to_dict('index')
    )
    centers = {
        k: v
        for k, v in centers.items()
        if totals[k] >= 5 and np.isfinite(list(v.values())).all()
    }
    with (out / ('geo_full.pkl' if full else 'geo_train.pkl')).open('wb') as f:
        pickle.dump(
            dict(
                pair=pair,
                totals=totals,
                global_prob=global_prob,
                centers=centers,
                local_rate=local_rate,
                overall_local_rate=overall,
            ),
            f,
            protocol=5,
        )
    log('geo prior', full, len(pair), 'location pairs')


class GeoPrior:
    def __init__(self, out, full=False):
        self.data = read_pickle(
            Path(out) / ('geo_full.pkl' if full else 'geo_train.pkl')
        )

    def features(self, query_location, item_locations, microcategories):
        prior = np.array(
            [self.data['global_prob'].get(loc, 1e-7) for loc in item_locations],
            np.float32,
        )
        observed = np.array(
            [self.data['pair'].get((query_location, loc), 0) for loc in item_locations],
            np.float32,
        )
        # Сглаживание защищает редкие и прежде не встречавшиеся локации.
        probability = (observed + 20 * prior) / (
            self.data['totals'].get(query_location, 0) + 20
        )
        local_rate = np.array(
            [
                self.data['local_rate'].get(cat, self.data['overall_local_rate'])
                for cat in microcategories
            ],
            np.float32,
        )
        return np.column_stack(
            [
                np.log(np.maximum(probability, 1e-9)),
                np.log(np.maximum(probability / prior, 1e-9)),
                local_rate,
            ]
        ).astype(np.float32)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--artifacts-dir', default='artifacts')
    p.add_argument('--full', action='store_true')
    a = p.parse_args()
    build_geo(a.data_dir, a.artifacts_dir, a.full)
