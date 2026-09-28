"""Диагностика численных различий предсказаний, без доступа к меткам benchmark.

Идентификатор используется только для сопоставления двух запусков. Повторяется
исходный блок из 16 запросов, чтобы сохранить форму матричных операций BLAS.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from advanced import AdvancedRetriever
from ensemble import Components, select


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', default='answer.csv')
    p.add_argument('--other', default='answer_reproduced.csv')
    p.add_argument('--artifacts-dir', default='artifacts')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.artifacts_dir)
    first = pd.read_csv(a.reference, dtype=str).set_index('query_id')
    second = pd.read_csv(a.other, dtype=str).set_index('query_id')
    different = [
        q
        for q in first.index
        if set(first.loc[q, 'answer'].split()) != set(second.loc[q, 'answer'].split())
    ]
    assert different, 'No differences'
    frame = pd.read_parquet(out / 'queries_benchmark.parquet')
    position = np.flatnonzero(frame.query_id.eq(different[0]).to_numpy())[0]
    start = position // 16 * 16
    retriever = AdvancedRetriever(out, sources='domain', full=True, benchmark_only=True)
    rows = retriever.retrieve(frame.iloc[start : start + 16])
    row = rows[position - start]
    config = json.loads((out / 'ensemble_domain.json').read_text())
    scores = Components(out, 'domain', retriever.feature_names).score(row['x'])
    selected = select(scores, config['selection'])
    np.savez(
        a.output,
        x=row['x'],
        indices=row['indices'],
        ids=retriever.ids[row['indices']].astype(str),
        feature_names=np.array(retriever.feature_names),
        selected=selected,
        **scores
    )
    print('Different queries:', len(different), 'diagnostic block start:', start)


if __name__ == '__main__':
    main()
