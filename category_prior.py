"""Мягкий прогноз микрокатегории по тексту запроса; без жёстких фильтров.

Для локального эксперимента используются только history_train: ни dev/test,
ни собственные запросы ранжировщика в эту обучающую статистику не попадают.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from solution import normalize, tokenize, query_key, read_pickle, log

FEATURE_NAMES = [
    'microcat_log_prob',
    'microcat_log_lift',
    'microcat_inverse_rank',
    'microcat_entropy',
]


def documents(frame):
    text = frame.search_query.fillna('')
    return (
        text + ' ' + text + ' ' + text + ' ' + frame.search_infm_params_text.fillna('')
    )


def build_prior(data_dir, artifacts_dir, full=False):
    out = Path(artifacts_dir)
    history = pd.read_parquet(
        out / ('history_full.parquet' if full else 'history_train.parquet')
    )
    labels = pd.read_parquet(
        Path(data_dir) / 'train.parquet', columns=['item_id', 'item_microcat_id']
    ).drop_duplicates('item_id')
    history = history.merge(labels, on='item_id', how='left', validate='many_to_one')
    assert history.item_microcat_id.notna().all()
    norms = history.search_query.map(normalize)
    # Каждый уникальный запрос имеет суммарный вес 1, независимо от популярности.
    weights = (1 / norms.map(norms.value_counts())).to_numpy()
    vectorizer = CountVectorizer(
        analyzer=tokenize, min_df=1, max_features=100000, dtype=np.float32
    )
    x = vectorizer.fit_transform(documents(history))
    model = MultinomialNB(alpha=0.5)
    model.fit(x, history.item_microcat_id.to_numpy(), sample_weight=weights)
    path = out / ('microcat_full.pkl' if full else 'microcat_train.pkl')
    with path.open('wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'model': model}, f, protocol=5)
    log(
        'microcategory prior',
        path,
        len(history),
        'rows',
        len(model.classes_),
        'classes',
        x.shape[1],
        'words',
    )


class CategoryPrior:
    def __init__(self, artifacts_dir, corpus, full=False):
        out = Path(artifacts_dir)
        saved = read_pickle(
            out / ('microcat_full.pkl' if full else 'microcat_train.pkl')
        )
        queries = pd.read_parquet(out / 'queries_all.parquet')
        self.lookup = {
            query_key(r): i for i, r in enumerate(queries.itertuples(index=False))
        }
        model = saved['model']
        x = saved['vectorizer'].transform(documents(queries))
        self.log_probs = model.predict_log_proba(x).astype(np.float32)
        self.log_prior = model.class_log_prior_.astype(np.float32)
        mapping = {v: i for i, v in enumerate(model.classes_)}
        self.item_class = np.array(
            [mapping.get(v, -1) for v in corpus.item_microcat_id], np.int32
        )
        order = np.argsort(-self.log_probs, axis=1, kind='stable')
        self.inverse_rank = np.empty_like(self.log_probs)
        np.put_along_axis(
            self.inverse_rank,
            order,
            np.broadcast_to(1 / (1 + np.arange(len(mapping))), order.shape),
            axis=1,
        )
        self.entropy = -(np.exp(self.log_probs) * self.log_probs).sum(1) / np.log(
            max(len(mapping), 2)
        )

    def features(self, row, indices):
        qi = self.lookup[query_key(row)]
        classes = self.item_class[indices]
        known = classes >= 0
        result = np.full((len(indices), 4), np.nan, np.float32)
        result[known, 0] = np.clip(self.log_probs[qi, classes[known]], -30, 0)
        result[known, 1] = np.clip(
            self.log_probs[qi, classes[known]] - self.log_prior[classes[known]], -30, 30
        )
        result[known, 2] = self.inverse_rank[qi, classes[known]]
        result[:, 3] = self.entropy[qi]
        return result
