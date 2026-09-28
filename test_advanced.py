"""Проверки новых общих правил отбора и географического сглаживания."""

import unittest
import numpy as np
from ensemble import select
from geo_prior import GeoPrior


class AdvancedTests(unittest.TestCase):
    def test_quota_fills_fifty_without_duplicates(self):
        scores = {'a': np.arange(100, dtype=float), 'b': np.arange(100, dtype=float)}
        result = select(
            scores, {'method': 'quota', 'first': 'a', 'second': 'b', 'quota': 20}
        )
        self.assertEqual(len(result), 50)
        self.assertEqual(len(set(result)), 50)
        np.testing.assert_array_equal(result, np.arange(99, 49, -1))

    def test_rank_mixture_ties_are_stable(self):
        scores = {'a': np.zeros(100), 'b': np.zeros(100)}
        result = select(scores, {'method': 'rank', 'weights': {'a': 0.5, 'b': 0.5}})
        np.testing.assert_array_equal(result, np.arange(50))

    def test_unknown_geography_is_smoothed(self):
        prior = object.__new__(GeoPrior)
        prior.data = {
            'global_prob': {1: 0.7, 2: 0.3},
            'pair': {(10, 1): 10},
            'totals': {10: 10},
            'local_rate': {100: 0.9},
            'overall_local_rate': 0.8,
        }
        values = prior.features(999, np.array([1, 2, 3]), np.array([100, 100, 999]))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values[:, 1], 0, atol=1e-6)
        np.testing.assert_allclose(values[:, 2], [0.9, 0.9, 0.8])


if __name__ == '__main__':
    unittest.main()
