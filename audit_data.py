"""Проверка схем, качества исходных данных и контрольных сумм без изменения данных."""

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from solution import normalize, SEARCH, dump_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--artifacts-dir', default='artifacts')
    args = p.parse_args()
    data = Path(args.data_dir)
    result = {}
    frames = {}
    for name in ['train', 'benchmark_queries', 'benchmark_items']:
        path = data / f'{name}.parquet'
        frame = pd.read_parquet(path)
        frames[name] = frame
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        result[name] = {
            'rows': len(frame),
            'columns': frame.columns.tolist(),
            'schema': str(pq.read_schema(path)),
            'sha256': checksum,
            'missing': frame.isna().sum().to_dict(),
            'duplicate_rows': int(frame.duplicated().sum()),
            'empty_texts': {
                c: int(frame[c].map(normalize).eq('').sum())
                for c in frame.columns
                if c.endswith(('_raw', '_text')) or c == 'search_query'
            },
        }
    train, queries, items = (
        frames[n] for n in ['train', 'benchmark_queries', 'benchmark_items']
    )
    assert all(c in train for c in SEARCH + items.columns.tolist())
    assert queries.query_id.is_unique and queries.query_id.str.len().eq(16).all()
    assert items.item_id.is_unique and items.item_id.str.fullmatch('[0-9a-f]{16}').all()
    assert train.item_id.str.fullmatch('[0-9a-f]{16}').all()
    result['overlap'] = {
        'unique_train_item_ids_in_benchmark': len(
            set(train.item_id) & set(items.item_id)
        ),
        'benchmark_normalized_texts_in_train': int(
            queries.search_query.map(normalize)
            .isin(train.search_query.map(normalize))
            .sum()
        ),
        'duplicate_positive_pairs': int(train.duplicated(SEARCH + ['item_id']).sum()),
    }
    output = Path(args.artifacts_dir) / 'data_audit.json'
    dump_json(output, result)
    print(
        json.dumps(
            {'output': str(output), 'overlap': result['overlap']}, ensure_ascii=False
        )
    )


if __name__ == '__main__':
    main()
