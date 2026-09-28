"""Поиск объявлений Авито: подготовка, индексы, валидация и CSV.

Все статистики обучения отделены от отложенных запросов. Идентификаторы
служат только ключами; правил для отдельных запросов здесь нет.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import RussianStemmer

SEED = 42
SEARCH = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
STEMMER = RussianStemmer()


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def normalize(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(
        unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е").split()
    )


@lru_cache(maxsize=400000)
def stem(word):
    return STEMMER.stem(word) if re.search("[а-я]", word) else word


def tokenize(text):
    return [stem(w) for w in re.findall(r"[а-яa-z0-9]+", normalize(text)) if len(w) > 1]


def dump_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def query_key(row):
    values = [getattr(row, c) if hasattr(row, c) else row[c] for c in SEARCH]
    return json.dumps(
        [str(v) for v in values], ensure_ascii=False, separators=(",", ":")
    )


def aggregate_queries(rows):
    """Одна оценочная единица — запрос с конкретной локацией и фильтрами."""
    result = (
        rows.groupby(SEARCH, sort=True, dropna=False)["item_id"]
        .agg(lambda x: sorted(set(x)))
        .reset_index()
    )
    result = result.rename(columns={"item_id": "relevant"})
    result["query_id"] = [
        hashlib.sha256(query_key(r).encode()).hexdigest()[:16]
        for _, r in result.iterrows()
    ]
    return result


def prepare(data_dir, artifacts_dir, dev_size=1200, train_size=8000):
    data, out = Path(data_dir), Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(data / "train.parquet")
    items = pd.read_parquet(data / "benchmark_items.parquet")
    queries = pd.read_parquet(data / "benchmark_queries.parquet")
    assert tr.shape[0] == 497673 and len(items) == 189212 and len(queries) == 2452
    assert items.item_id.is_unique and queries.query_id.is_unique
    assert items.item_id.str.fullmatch("[0-9a-f]{16}").all()
    tr["normalized_query"] = tr.search_query.map(normalize)
    unique = np.array(sorted(tr.normalized_query.unique()), dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique)
    n = len(unique)
    split = {
        q: ("train" if i < int(0.8 * n) else "dev" if i < int(0.9 * n) else "test")
        for i, q in enumerate(unique)
    }
    tr["split"] = tr.normalized_query.map(split)
    pd.DataFrame(
        {"normalized_query": list(split), "split": list(split.values())}
    ).to_parquet(out / "split.parquet", index=False)
    samples = {}
    for name in ["train", "dev", "test"]:
        groups = aggregate_queries(tr[tr.split == name])
        # В бенчмарке почти каждый текст уникален. Сначала выбираем случайный
        # контекст каждого текста, затем равномерно выбираем сами тексты.
        # Иначе популярные запросы с сотнями локаций завышают их долю в метрике.
        groups["_normalized_query"] = groups.search_query.map(normalize)
        groups = (
            groups.sample(frac=1, random_state=SEED)
            .drop_duplicates("_normalized_query")
            .drop(columns="_normalized_query")
        )
        limit = train_size if name == "train" else dev_size
        groups = (
            groups.sample(n=min(limit, len(groups)), random_state=SEED)
            .sort_values("query_id")
            .reset_index(drop=True)
        )
        samples[name] = groups
        groups.to_parquet(out / f"queries_{name}.parquet", index=False)
    # Для обучения ранжировщика тексты этих запросов убраны из истории кликов.
    # Это предотвращает использование собственной метки как признака.
    ranker_norm = set(samples["train"].search_query.map(normalize))
    history = tr[(tr.split == "train") & ~tr.normalized_query.isin(ranker_norm)]
    history[SEARCH + ["item_id"]].to_parquet(out / "history_train.parquet", index=False)
    tr[SEARCH + ["item_id"]].to_parquet(out / "history_full.parquet", index=False)
    needed = set()
    for frame in samples.values():
        for ids in frame.relevant:
            needed.update(ids)
    extra = tr[tr.item_id.isin(needed)][items.columns].drop_duplicates("item_id")
    corpus = (
        pd.concat([items, extra])
        .drop_duplicates("item_id", keep="first")
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    corpus["is_benchmark"] = corpus.item_id.isin(items.item_id)
    corpus.to_parquet(out / "corpus.parquet", index=False)
    queries.to_parquet(out / "queries_benchmark.parquet", index=False)
    # Эмбеддинги не содержат статистик взаимодействия, поэтому общий корпус безопасен.
    all_q = pd.concat(
        [samples[n][SEARCH + ["query_id"]] for n in samples] + [queries],
        ignore_index=True,
    )
    assert all_q.query_id.is_unique
    all_q.to_parquet(out / "queries_all.parquet", index=False)
    profile = {
        "train_rows": len(tr),
        "benchmark_items": len(items),
        "benchmark_queries": len(queries),
        "unique_train_queries": int(tr.search_query.nunique()),
        "unique_train_items": int(tr.item_id.nunique()),
        "interaction_item_overlap": float(tr.item_id.isin(items.item_id).mean()),
        "benchmark_query_overlap": float(
            queries.search_query.map(normalize).isin(tr.normalized_query).mean()
        ),
        "train_nulls": tr.isna().sum().to_dict(),
        "corpus_rows": len(corpus),
        "sample_sizes": {k: len(v) for k, v in samples.items()},
        "split_query_counts": dict(Counter(split.values())),
        "history_rows": len(history),
        "location_match_train": float(
            (tr.search_location_id == tr.item_location_id).mean()
        ),
        "category_counts": items.item_category_id.value_counts().head(10).to_dict(),
        "microcategory_counts": items.item_microcat_id.value_counts()
        .head(10)
        .to_dict(),
        "evaluation_note": "One random search context per normalized query, then uniform query sampling. Observed positives; corpus is benchmark plus sampled train/dev/test positives. Query groups are disjoint.",
    }
    dump_json(out / "profile.json", profile)
    log("prepared", profile)


class BM25:
    """BM25 через CSR: k1=1.2, b=0.75, положительный Robertson IDF."""

    def fit(self, texts):
        self.vectorizer = CountVectorizer(
            analyzer=tokenize, min_df=1, max_features=600000, dtype=np.float32
        )
        counts = self.vectorizer.fit_transform(texts).tocsr()
        self.length = np.asarray(counts.sum(axis=1)).ravel()
        average = max(float(self.length.mean()), 1.0)
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((counts.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
        row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
        counts.data *= 2.2 / (
            counts.data + 1.2 * (0.25 + 0.75 * self.length[row_ids] / average)
        )
        counts.data *= idf[counts.indices]
        self.matrix = counts.tocsc()
        return self

    def score(self, texts):
        q = self.vectorizer.transform(texts)
        q.data[:] = 1
        return (q @ self.matrix.T).toarray().astype(np.float32)


def build_index(artifacts_dir, full=False):
    out = Path(artifacts_dir)
    corpus = pd.read_parquet(out / "corpus.parquet")
    target = out / ("index_full.pkl" if full else "index_train.pkl")
    common = out / "index_content.pkl"
    if common.exists():
        with common.open("rb") as f:
            content = pickle.load(f)
    else:
        log("BM25 content", len(corpus))
        title = corpus.item_title_raw.fillna("").map(normalize)
        params = (
            corpus.item_infm_params_text.fillna("").map(normalize).str.slice(0, 600)
        )
        description = (
            corpus.item_description_raw.fillna("").map(normalize).str.slice(0, 2200)
        )
        docs = title + " " + title + " " + title + " " + params + " " + description
        bm25 = BM25().fit(docs)
        log("char TF-IDF")
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=600000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        char_matrix = char.fit_transform(title).tocsr()
        content = {
            "bm25": bm25,
            "char": char,
            "char_matrix": char_matrix,
            "titles": [set(tokenize(t)) for t in title],
            "params": [set(tokenize(p)) for p in params],
            "title_raw": title.tolist(),
        }
        with common.open("wb") as f:
            pickle.dump(content, f, protocol=5)
        log("content index saved")
    history = pd.read_parquet(
        out / ("history_full.parquet" if full else "history_train.parquet")
    )
    # Дубликаты исторического текста не размножаются искусственно.
    histories = history.groupby("item_id").search_query.agg(
        lambda x: " ".join(sorted(set(x)))
    )
    history_docs = corpus.item_id.map(histories).fillna("")
    history_bm = BM25().fit(history_docs)
    popularity = (
        corpus.item_id.map(history.item_id.value_counts())
        .fillna(0)
        .to_numpy(np.float32)
    )
    # География запроса восстанавливается только из координат его location_id.
    coords = corpus[["item_location_id", "item_latitude", "item_longitude"]].copy()
    for c in ["item_latitude", "item_longitude"]:
        coords[c] = pd.to_numeric(coords[c], errors="coerce")
    centers = (
        coords.groupby("item_location_id")[["item_latitude", "item_longitude"]]
        .median()
        .to_dict("index")
    )
    with target.open("wb") as f:
        pickle.dump(
            {"history": history_bm, "popularity": popularity, "centers": centers},
            f,
            protocol=5,
        )
    log("history index saved", target)


def encode(artifacts_dir, gpu=0, shard=0, shards=1, batch_size=32):
    """Локальная модель, FP32 и обычный attention совместимы с GTX 1080 Ti."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    out = Path(artifacts_dir)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(out / "e5-small", local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            out / "e5-small", local_files_only=True, attn_implementation="eager"
        )
        .eval()
        .to(f"cuda:{gpu}")
    )
    corpus = pd.read_parquet(out / "corpus.parquet")
    queries = pd.read_parquet(out / "queries_all.parquet")
    item_texts = (
        "passage: "
        + corpus.item_title_raw.fillna("")
        + ". "
        + corpus.item_infm_params_text.fillna("").str.slice(0, 350)
        + ". "
        + corpus.item_description_raw.fillna("").str.slice(0, 2000)
    ).tolist()
    query_texts = (
        "query: "
        + queries.search_query
        + ". "
        + queries.search_infm_params_text.fillna("")
    ).tolist()
    bare_queries = ("query: " + queries.search_query).tolist()
    for kind, texts in [
        ("items", item_texts),
        ("queries", query_texts),
        ("queries_text", bare_queries),
    ]:
        positions = np.arange(shard, len(texts), shards)
        destination = out / f"emb_{kind}_{shard}.npz"
        if destination.exists():
            log("already encoded", destination)
            continue
        vectors = np.zeros((len(positions), 384), dtype=np.float32)
        start = time.monotonic()
        for offset in range(0, len(positions), batch_size):
            ids = positions[offset : offset + batch_size]
            inputs = tokenizer(
                [texts[i] for i in ids],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(f"cuda:{gpu}")
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                embedding = (hidden * mask).sum(1) / mask.sum(1)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            vectors[offset : offset + len(ids)] = embedding.cpu().numpy()
            if offset % (batch_size * 100) == 0:
                log(
                    kind,
                    shard,
                    offset,
                    len(positions),
                    "seconds",
                    round(time.monotonic() - start),
                )
        np.savez(destination, positions=positions, embeddings=vectors)
        log("saved", destination)


def merge_embeddings(artifacts_dir, shards=2):
    out = Path(artifacts_dir)
    for kind, table in [
        ("items", "corpus"),
        ("queries", "queries_all"),
        ("queries_text", "queries_all"),
    ]:
        frame = pd.read_parquet(out / f"{table}.parquet")
        all_embeddings = np.empty((len(frame), 384), np.float32)
        seen = np.zeros(len(frame), bool)
        for i in range(shards):
            part = np.load(out / f"emb_{kind}_{i}.npz")
            all_embeddings[part["positions"]] = part["embeddings"]
            seen[part["positions"]] = True
        assert seen.all()
        np.save(out / f"emb_{kind}.npy", all_embeddings)
    log("embeddings merged")


def top_indices(scores, k=200, mask=None):
    """Детерминированный top-k, включая равенства на границе отсечения."""
    valid = np.flatnonzero(mask) if mask is not None else np.arange(len(scores))
    if not len(valid):
        return np.empty(0, np.int32)
    values = scores[valid]
    k = min(k, len(valid))
    threshold = np.partition(values, len(values) - k)[len(values) - k]
    above = valid[values > threshold]
    tied = valid[values == threshold][: k - len(above)]
    selected = np.r_[above, tied]
    return selected[np.lexsort((selected, -scores[selected]))].astype(np.int32)


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Индекс доступен и из CLI, и при импорте модуля в Jupyter.
        if module == "__main__" and name in globals():
            return globals()[name]
        return super().find_class(module, name)


def read_pickle(path):
    with Path(path).open("rb") as f:
        return CompatibleUnpickler(f).load()


class Retriever:
    def __init__(
        self,
        artifacts_dir,
        full=False,
        dense=False,
        benchmark_only=False,
        metadata=False,
    ):
        self.out = Path(artifacts_dir)
        self.corpus = pd.read_parquet(self.out / "corpus.parquet")
        self.content = read_pickle(self.out / "index_content.pkl")
        self.history = read_pickle(
            self.out / ("index_full.pkl" if full else "index_train.pkl")
        )
        self.ids = self.corpus.item_id.to_numpy()
        self.id_to_pos = dict(zip(self.ids, range(len(self.ids))))
        self.location = self.corpus.item_location_id.to_numpy()
        self.category = self.corpus.item_category_id.to_numpy()
        self.allowed = (
            self.corpus.is_benchmark.to_numpy()
            if benchmark_only
            else np.ones(len(self.ids), bool)
        )
        self.dense = dense
        self.metadata = metadata
        if metadata:
            from category_prior import CategoryPrior

            self.prior = CategoryPrior(self.out, self.corpus, full=full)
        if dense:
            self.item_emb = np.load(self.out / "emb_items.npy", mmap_mode="r")
            self.query_emb = np.load(self.out / "emb_queries.npy", mmap_mode="r")
            self.query_text_emb = np.load(
                self.out / "emb_queries_text.npy", mmap_mode="r"
            )
            cached_queries = pd.read_parquet(self.out / "queries_all.parquet")
            # Кэш определяется содержимым запроса, а не его внешним query_id.
            self.query_position = {
                query_key(r): i
                for i, r in enumerate(cached_queries.itertuples(index=False))
            }
        self.numeric = {}
        for col in [
            "item_price",
            "item_rating",
            "item_rating_reviews_count",
            "item_latitude",
            "item_longitude",
            "item_is_phone_hidden",
            "item_is_message_forbidden",
            "item_microcat_id",
        ]:
            self.numeric[col] = pd.to_numeric(
                self.corpus[col], errors="coerce"
            ).to_numpy(dtype=np.float32)

    def features(self, row, candidates, scores, ranks):
        q_tokens = set(tokenize(row.search_query))
        f_tokens = set(tokenize(row.search_infm_params_text))
        n = len(candidates)
        columns, names = [], []

        def add(name, values):
            names.append(name)
            columns.append(np.broadcast_to(np.asarray(values, np.float32), (n,)))

        for name, values in scores.items():
            add(name, values[candidates])
            add(
                name + "_relative",
                values[candidates] / max(float(values[self.allowed].max()), 1e-6),
            )
        for name, positions in ranks.items():
            add("rrf_" + name, 60 / (60 + positions[candidates]))
        add("same_location", self.location[candidates] == row.search_location_id)
        add(
            "same_category",
            (self.category[candidates] == row.search_category)
            & (row.search_category != 0),
        )
        add("query_category_missing", row.search_category == 0)
        add("delivery", row.search_is_delivery_search)
        add("query_words", len(q_tokens))
        add("filter_words", len(f_tokens))
        add(
            "title_query_coverage",
            [
                len(q_tokens & self.content["titles"][i]) / max(len(q_tokens), 1)
                for i in candidates
            ],
        )
        add(
            "title_jaccard",
            [
                len(q_tokens & self.content["titles"][i])
                / max(len(q_tokens | self.content["titles"][i]), 1)
                for i in candidates
            ],
        )
        add(
            "title_phrase",
            [
                (
                    normalize(row.search_query) in self.content["title_raw"][i]
                    if normalize(row.search_query)
                    else False
                )
                for i in candidates
            ],
        )
        add(
            "filter_coverage",
            [
                len(f_tokens & self.content["params"][i]) / max(len(f_tokens), 1)
                for i in candidates
            ],
        )
        add(
            "query_params_coverage",
            [
                len(q_tokens & self.content["params"][i]) / max(len(q_tokens), 1)
                for i in candidates
            ],
        )
        add("log_popularity", np.log1p(self.history["popularity"][candidates]))
        for name in ["item_price", "item_rating_reviews_count"]:
            vals = self.numeric[name][candidates]
            add("log_" + name, np.log1p(np.maximum(vals, 0)))
        for name in [
            "item_rating",
            "item_is_phone_hidden",
            "item_is_message_forbidden",
            "item_microcat_id",
        ]:
            add(name, self.numeric[name][candidates])
        center = self.history["centers"].get(row.search_location_id)
        if center:
            lat, lon = np.radians(
                self.numeric["item_latitude"][candidates]
            ), np.radians(self.numeric["item_longitude"][candidates])
            lat0, lon0 = np.radians([center["item_latitude"], center["item_longitude"]])
            h = (
                np.sin((lat - lat0) / 2) ** 2
                + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
            )
            distance = 12742 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
            add("log_distance", np.log1p(distance))
        else:
            add("log_distance", np.nan)
        features = np.column_stack(columns).astype(np.float32)
        if getattr(self, 'metadata', False):
            from category_prior import FEATURE_NAMES

            features = np.column_stack([features, self.prior.features(row, candidates)])
            names = names + FEATURE_NAMES
        self.feature_names = names
        return features

    def retrieve(self, queries, training=False):
        records = []
        rng = np.random.default_rng(SEED)
        for start in range(0, len(queries), 16):
            batch = queries.iloc[start : start + 16]
            texts = batch.search_query.tolist()
            # Фильтры не должны подавлять короткий смысловой запрос.
            bm_scores = self.content["bm25"].score(texts)
            char_scores = (
                (
                    self.content["char"].transform([normalize(t) for t in texts])
                    @ self.content["char_matrix"].T
                )
                .toarray()
                .astype(np.float32)
            )
            history_scores = self.history["history"].score(texts)
            if self.dense:
                positions = [
                    self.query_position[query_key(r)]
                    for r in batch.itertuples(index=False)
                ]
                dense_scores = np.asarray(self.query_emb[positions] @ self.item_emb.T)
                dense_text_scores = np.asarray(
                    self.query_text_emb[positions] @ self.item_emb.T
                )
            for bi, row in enumerate(batch.itertuples(index=False)):
                scores = {
                    "bm25": bm_scores[bi],
                    "char": char_scores[bi],
                    "history": history_scores[bi],
                }
                if self.dense:
                    scores["dense"] = dense_scores[bi]
                    scores["dense_text"] = dense_text_scores[bi]
                ranks, found = {}, []
                local = self.allowed & (self.location == row.search_location_id)
                for name, values in scores.items():
                    for suffix, mask in [("global", self.allowed), ("local", local)]:
                        # Нулевой текстовый score не является свидетельством совпадения.
                        mask = mask & (values > 0)
                        selected = top_indices(values, 200, mask)
                        rank = np.full(len(self.ids), 1e6, np.float32)
                        rank[selected] = np.arange(1, len(selected) + 1)
                        ranks[name + "_" + suffix] = rank
                        found.extend(selected.tolist())
                candidates = np.array(sorted(set(found)), np.int32)
                if len(candidates) < 50:
                    fallback = top_indices(self.history["popularity"], 50, self.allowed)
                    candidates = np.array(
                        sorted(set(candidates) | set(fallback)), np.int32
                    )
                relevant = set(getattr(row, "relevant", []))
                if training:
                    # Только для обучения добавляем ненайденные позитивы; на dev/test — никогда.
                    positives = np.array(
                        [self.id_to_pos[i] for i in sorted(relevant)], np.int32
                    )
                    negatives = candidates[~np.isin(candidates, positives)]
                    fusion = sum(60 / (60 + r[negatives]) for r in ranks.values())
                    best = negatives[top_indices(fusion, min(25, len(negatives)))]
                    rest = np.setdiff1d(negatives, best)
                    random = rng.choice(rest, size=min(25, len(rest)), replace=False)
                    candidates = np.unique(np.r_[positives, best, random]).astype(
                        np.int32
                    )
                x = self.features(row, candidates, scores, ranks)
                labels = np.array(
                    [i in relevant for i in self.ids[candidates]], np.float32
                )
                records.append(
                    {
                        "query_id": row.query_id,
                        "indices": candidates,
                        "x": x,
                        "y": labels,
                        "n_relevant": len(relevant),
                    }
                )
            if start % 160 == 0:
                log(
                    "retrieved",
                    start + len(batch),
                    "/",
                    len(queries),
                    "training",
                    training,
                )
        return records


def baseline_score(x, names, local_weight=1.0, dense_weight=1.0, history_weight=0.5):
    result = np.zeros(len(x), np.float32)
    for j, name in enumerate(names):
        if name.startswith("rrf_"):
            weight = local_weight if name.endswith("local") else 1.0
            if "dense" in name:
                weight *= dense_weight
            if "history" in name:
                weight *= history_weight
            result += weight * x[:, j]
    # Совпадение локации — мягкий prior, без отсечения удалённых услуг.
    return result


def recall(records, score_function):
    values, ceilings = [], []
    for record in records:
        indices = top_indices(score_function(record["x"]), 50)
        denominator = max(record["n_relevant"], 1)
        values.append(float(record["y"][indices].sum() / denominator))
        ceilings.append(float(record["y"].sum() / denominator))
    return float(np.mean(values)), float(np.mean(ceilings))


def get_records(retriever, split, training=False, refresh=False):
    kind = "dense" if retriever.dense else "lexical"
    if retriever.metadata:
        from category_prior import FEATURE_NAMES

        base_path = retriever.out / f"candidates_{split}_{kind}.pkl"
        if refresh or not base_path.exists():
            # Сначала строим исходные кандидаты обычным поиском. Это важно для
            # независимого test, который до фиксации модели ещё не вычислялся.
            retriever.metadata = False
            try:
                get_records(retriever, split, training=training, refresh=refresh)
            finally:
                retriever.metadata = True
        base = read_pickle(base_path)
        queries = pd.read_parquet(retriever.out / f"queries_{split}.parquet").set_index(
            'query_id'
        )
        for r in base['records']:
            r['x'] = np.column_stack(
                [
                    r['x'],
                    retriever.prior.features(queries.loc[r['query_id']], r['indices']),
                ]
            )
        retriever.feature_names = base['feature_names'] + FEATURE_NAMES
        cache = retriever.out / f"candidates_{split}_{kind}_meta.pkl"
        with cache.open('wb') as f:
            pickle.dump(
                {'records': base['records'], 'feature_names': retriever.feature_names},
                f,
                protocol=5,
            )
        return base['records']
    cache = retriever.out / f"candidates_{split}_{kind}.pkl"
    if cache.exists() and not refresh:
        stored = read_pickle(cache)
        retriever.feature_names = stored["feature_names"]
        return stored["records"]
    queries = pd.read_parquet(retriever.out / f"queries_{split}.parquet")
    records = retriever.retrieve(queries, training=training)
    with cache.open("wb") as f:
        pickle.dump(
            {"records": records, "feature_names": retriever.feature_names},
            f,
            protocol=5,
        )
    return records


def blend_scores(model_score, base_score, weight):
    """Смешивание рангов уменьшает чувствительность к шкале PairLogit."""
    if weight == 1:
        return model_score
    if weight == 0:
        return base_score
    rank_model = np.empty(len(model_score), np.float32)
    rank_base = np.empty(len(base_score), np.float32)
    rank_model[np.argsort(-model_score, kind="stable")] = np.arange(
        1, len(model_score) + 1
    )
    rank_base[np.argsort(-base_score, kind="stable")] = np.arange(
        1, len(base_score) + 1
    )
    return weight * 60 / (60 + rank_model) + (1 - weight) * 60 / (60 + rank_base)


def train(artifacts_dir, dense=False, ranker=False, reuse_model=False, metadata=False):
    out = Path(artifacts_dir)
    retriever = Retriever(out, dense=dense, metadata=metadata)
    kind = ("dense" if dense else "lexical") + ("_meta" if metadata else "")
    dev = get_records(retriever, "dev")
    names = retriever.feature_names
    scores = []
    for lw in [0.5, 1.0, 2.0, 4.0]:
        for dw in ([0.5, 1.0, 2.0] if dense else [0.0]):
            for hw in [0.0, 0.5, 1.0]:
                parameters = {
                    "local_weight": lw,
                    "dense_weight": dw,
                    "history_weight": hw,
                }
                metric, ceiling = recall(
                    dev, lambda x: baseline_score(x, names, **parameters)
                )
                scores.append(
                    {**parameters, "recall50": metric, "candidate_recall": ceiling}
                )
    best = max(scores, key=lambda x: x["recall50"])
    config = {
        "dense": dense,
        "metadata": metadata,
        "feature_names": names,
        "baseline": {
            k: best[k] for k in ["local_weight", "dense_weight", "history_weight"]
        },
        "use_ranker": False,
        "dev_recall50": best["recall50"],
        "candidate_recall": best["candidate_recall"],
    }
    log("best baseline", best)
    if ranker:
        from catboost import CatBoostRanker, Pool

        def pool(records):
            return Pool(
                np.concatenate([r["x"] for r in records]),
                label=np.concatenate([r["y"] for r in records]),
                group_id=np.concatenate(
                    [np.full(len(r["x"]), i, np.int32) for i, r in enumerate(records)]
                ),
                feature_names=names,
            )

        model_path = out / f"ranker_{kind}.cbm"
        if reuse_model:
            model = CatBoostRanker().load_model(str(model_path))
        else:
            training = get_records(retriever, "train", training=True)
            model = CatBoostRanker(
                iterations=600,
                depth=6,
                learning_rate=0.07,
                loss_function="PairLogit:max_pairs=200",
                random_seed=SEED,
                thread_count=6,
                verbose=50,
                allow_writing_files=False,
            )
            model.fit(pool(training), eval_set=pool(dev), early_stopping_rounds=80)
        flat = np.concatenate([r["x"] for r in dev])
        lengths = np.cumsum([0] + [len(r["x"]) for r in dev])
        rank_results = []
        base_predictions = [
            baseline_score(r["x"], names, **config["baseline"]) for r in dev
        ]
        for trees in sorted(
            set(
                [
                    min(t, model.tree_count_)
                    for t in [
                        25,
                        50,
                        75,
                        100,
                        150,
                        200,
                        300,
                        400,
                        500,
                        600,
                        model.tree_count_,
                    ]
                ]
            )
        ):
            predictions = model.predict(flat, ntree_end=trees)
            for weight in [0.25, 0.5, 0.75, 1.0]:
                values = []
                for i, r in enumerate(dev):
                    mixed = blend_scores(
                        predictions[lengths[i] : lengths[i + 1]],
                        base_predictions[i],
                        weight,
                    )
                    chosen = top_indices(mixed, 50)
                    values.append(float(r["y"][chosen].sum() / max(r["n_relevant"], 1)))
                rank_results.append(
                    {
                        "trees": trees,
                        "blend_weight": weight,
                        "recall50": float(np.mean(values)),
                    }
                )
        winner = max(rank_results, key=lambda x: x["recall50"])
        model.save_model(str(model_path))
        config["ranker_results"] = rank_results
        if winner["recall50"] > config["dev_recall50"]:
            config.update(
                use_ranker=True,
                trees=winner["trees"],
                blend_weight=winner["blend_weight"],
                dev_recall50=winner["recall50"],
            )
        log("ranker best", winner)
    dump_json(out / f"config_{kind}.json", config)
    dump_json(out / f"validation_{kind}.json", {"grid": scores, "selected": config})
    log(
        "saved configuration",
        {
            k: v
            for k, v in config.items()
            if k not in ['ranker_results', 'feature_names']
        },
    )


def predict(
    artifacts_dir, output, dense=False, full=False, split="benchmark", metadata=False
):
    if full and split != "benchmark":
        raise ValueError(
            "Полная история содержит отложенные метки; --full разрешён только для benchmark."
        )
    out = Path(artifacts_dir)
    kind = ("dense" if dense else "lexical") + ("_meta" if metadata else "")
    config = json.loads((out / f"config_{kind}.json").read_text())
    retriever = Retriever(
        out,
        full=full,
        dense=dense,
        benchmark_only=(split == "benchmark"),
        metadata=metadata,
    )
    if split in ["dev", "test"] and not full:
        records = get_records(retriever, split)
    else:
        queries = pd.read_parquet(out / f"queries_{split}.parquet")
        records = retriever.retrieve(queries)
    names = retriever.feature_names
    assert names == config["feature_names"]
    if config["use_ranker"]:
        from catboost import CatBoostRanker

        model = CatBoostRanker().load_model(str(out / f"ranker_{kind}.cbm"))
        score = lambda x: blend_scores(
            model.predict(x, ntree_end=config["trees"]),
            baseline_score(x, names, **config["baseline"]),
            config.get("blend_weight", 1.0),
        )
    else:
        score = lambda x: baseline_score(x, names, **config["baseline"])
    if split != "benchmark":
        metric, ceiling = recall(records, score)
        report = {
            "split": split,
            "recall50": metric,
            "candidate_recall": ceiling,
            "queries": len(records),
        }
        dump_json(output, report)
        log("EVALUATION", report)
    else:
        result = []
        for r in records:
            chosen = top_indices(score(r["x"]), 50)
            ids = retriever.ids[r["indices"][chosen]]
            # Recall@50 зависит только от множества. Канонический порядок
            # устраняет несущественную перестановку внутри top-50 между BLAS.
            result.append((r["query_id"], " ".join(sorted(ids))))
        pd.DataFrame(result, columns=["query_id", "answer"]).to_csv(
            output, index=False, encoding="utf-8", lineterminator="\n"
        )
        log("saved", output, len(result))


def validate(data_dir, output):
    data = Path(data_dir)
    q = pd.read_parquet(data / "benchmark_queries.parquet", columns=["query_id"])
    it = pd.read_parquet(data / "benchmark_items.parquet", columns=["item_id"])
    answer = pd.read_csv(output, dtype=str, keep_default_na=False, encoding="utf-8")
    assert answer.columns.tolist() == ["query_id", "answer"], "Wrong columns"
    assert len(answer) == len(q) and answer.query_id.is_unique
    assert set(answer.query_id) == set(q.query_id)
    assert answer.query_id.str.len().eq(16).all()
    known = set(it.item_id)
    for row in answer.itertuples(index=False):
        ids = row.answer.split(" ")
        assert len(ids) == 50 and len(set(ids)) == 50, row.query_id
        assert all(
            re.fullmatch("[0-9a-f]{16}", i) and i in known for i in ids
        ), row.query_id
    digest = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    log("VALID", len(answer), "queries, 50 unique existing items each; sha256", digest)
    return {"rows": len(answer), "items_per_query": 50, "sha256": digest}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=[
            "prepare",
            "index",
            "encode",
            "merge",
            "validate",
            "train",
            "predict",
            "prior",
        ],
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--output", default="answer.csv")
    p.add_argument("--full", action="store_true")
    p.add_argument("--dense", action="store_true")
    p.add_argument("--metadata", action="store_true")
    p.add_argument("--ranker", action="store_true")
    p.add_argument("--reuse-model", action="store_true")
    p.add_argument("--split", default="benchmark", choices=["benchmark", "dev", "test"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.data_dir, a.artifacts_dir)
    elif a.command == "index":
        build_index(a.artifacts_dir, a.full)
    elif a.command == "encode":
        encode(a.artifacts_dir, a.gpu, a.shard, a.shards, a.batch_size)
    elif a.command == "merge":
        merge_embeddings(a.artifacts_dir, a.shards)
    elif a.command == "validate":
        validate(a.data_dir, a.output)
    elif a.command == "train":
        train(a.artifacts_dir, a.dense, a.ranker, a.reuse_model, a.metadata)
    elif a.command == "predict":
        predict(a.artifacts_dir, a.output, a.dense, a.full, a.split, a.metadata)
    elif a.command == "prior":
        from category_prior import build_prior

        build_prior(a.data_dir, a.artifacts_dir, a.full)


if __name__ == "__main__":
    main()
