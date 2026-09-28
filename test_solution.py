"""Проверки границ формата, детерминизма top-k и определения Recall@50."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from solution import normalize, top_indices, recall, validate, Retriever
from types import SimpleNamespace


class SolutionTests(unittest.TestCase):
    def test_normalization_and_empty(self):
        self.assertEqual(normalize("  ЁЛКА\n РЕМОНТ  "), "елка ремонт")
        self.assertEqual(normalize(None), "")
        self.assertEqual(normalize(float("nan")), "")

    def test_stable_ties_and_mask(self):
        values = np.array([1.0, 3.0, 3.0, 3.0, 0.0])
        np.testing.assert_array_equal(top_indices(values, 2), [1, 2])
        np.testing.assert_array_equal(
            top_indices(values, 5, np.array([0, 0, 1, 0, 0], bool)), [2]
        )
        self.assertEqual(len(top_indices(values, 50, np.zeros(5, bool))), 0)

    def test_recall_denominator_includes_missed_candidates(self):
        # Ненайденное релевантное объявление остаётся в знаменателе.
        records = [
            {"x": np.array([[3.0], [2.0]]), "y": np.array([1.0, 0.0]), "n_relevant": 2},
            {"x": np.array([[1.0]]), "y": np.array([1.0]), "n_relevant": 1},
        ]
        score, ceiling = recall(records, lambda x: x[:, 0])
        self.assertEqual(score, 0.75)
        self.assertEqual(ceiling, 0.75)

    def test_empty_query_unknown_location_missing_numbers(self):
        retriever = object.__new__(Retriever)
        retriever.location = np.array([1, 2])
        retriever.category = np.array([114, 114])
        retriever.allowed = np.ones(2, bool)
        retriever.content = {
            "titles": [set(), {"ремонт"}],
            "params": [set(), set()],
            "title_raw": ["", "ремонт"],
        }
        retriever.history = {"popularity": np.zeros(2), "centers": {}}
        retriever.numeric = {
            name: np.array([np.nan, 0.0], np.float32)
            for name in [
                "item_price",
                "item_rating",
                "item_rating_reviews_count",
                "item_latitude",
                "item_longitude",
                "item_is_phone_hidden",
                "item_is_message_forbidden",
                "item_microcat_id",
            ]
        }
        row = SimpleNamespace(
            search_query="",
            search_infm_params_text="",
            search_location_id=-1,
            search_category=0,
            search_is_delivery_search=0,
        )
        features = retriever.features(
            row,
            np.array([0, 1]),
            {"bm25": np.zeros(2)},
            {"bm25_global": np.full(2, 1e6)},
        )
        self.assertFalse(np.isinf(features).any())
        self.assertTrue(
            np.isnan(features[:, retriever.feature_names.index("log_distance")]).all()
        )
        self.assertTrue(
            (
                features[:, retriever.feature_names.index("title_query_coverage")] == 0
            ).all()
        )

    def test_csv_ids_and_rejections(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            query_id = "00WuFMaXSFZBxSzT"
            ids = [f"{i:016x}" for i in range(50)]
            pd.DataFrame({"query_id": [query_id]}).to_parquet(
                folder / "benchmark_queries.parquet"
            )
            pd.DataFrame({"item_id": ids}).to_parquet(
                folder / "benchmark_items.parquet"
            )
            answer = pd.DataFrame({"query_id": [query_id], "answer": [" ".join(ids)]})
            answer.to_csv(folder / "answer.csv", index=False)
            self.assertEqual(validate(folder, folder / "answer.csv")["rows"], 1)
            for broken in [
                " ".join(ids[:-1] + [ids[0]]),
                " ".join(ids[:-1]),
                " ".join(ids).upper(),
                " ".join(ids + ["ffffffffffffffff"]),
            ]:
                answer["answer"] = broken
                answer.to_csv(folder / "answer.csv", index=False)
                with self.assertRaises(AssertionError):
                    validate(folder, folder / "answer.csv")

    def test_retrieval_without_lexical_matches_has_fifty_candidates(self):
        # Если все текстовые источники пусты, остаётся стабильный резерв по
        # популярности. Он должен соблюдать маску корпуса и размер ответа.
        count = 65
        retriever = object.__new__(Retriever)
        retriever.ids = np.array([f'{i:016x}' for i in range(count)])
        retriever.allowed = np.r_[np.ones(60, bool), np.zeros(5, bool)]
        retriever.location = np.zeros(count)
        retriever.dense = False
        zeros = SimpleNamespace(
            score=lambda texts: np.zeros((len(texts), count), np.float32)
        )
        chars = SimpleNamespace(
            transform=lambda texts: sparse.csr_matrix((len(texts), 1), dtype=np.float32)
        )
        retriever.content = {
            'bm25': zeros,
            'char': chars,
            'char_matrix': sparse.csr_matrix((count, 1)),
        }
        retriever.history = {'history': zeros, 'popularity': np.zeros(count)}
        retriever.features = lambda row, candidates, scores, ranks: np.zeros(
            (len(candidates), 1)
        )
        queries = pd.DataFrame(
            {'query_id': ['A' * 16], 'search_query': [''], 'search_location_id': [-1]}
        )
        records = retriever.retrieve(queries)
        np.testing.assert_array_equal(records[0]['indices'], np.arange(50))


if __name__ == "__main__":
    unittest.main()
