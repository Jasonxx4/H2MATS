import unittest

import numpy as np

from hnsw_large_scale import _evaluate, vector_recall_metrics


class HNSWLargeScaleHelpersTest(unittest.TestCase):
    def test_vector_recall_metrics_compare_with_exact_embedding_neighbors(self):
        approximate = [np.asarray([0, 2, 3]), np.asarray([4, 5, 7])]
        exact = [np.asarray([0, 1, 2]), np.asarray([4, 5, 6])]
        metrics = vector_recall_metrics(approximate, exact, ks=(1, 3))

        self.assertEqual(metrics["queries"], 2)
        self.assertEqual(metrics["vector_recall_at_1"], 1.0)
        self.assertAlmostEqual(metrics["vector_recall_at_3"], 2.0 / 3.0)
        self.assertEqual(metrics["exact_top1_recall_at_3"], 1.0)

    def test_exact_source_metrics_use_initial_positive_mask(self):
        gallery = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [2.0, 0.0]], dtype=np.float32
        )
        queries = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        result = _evaluate(
            "exact",
            queries,
            np.asarray([10, 20]),
            gallery,
            np.asarray([10, 10, 20]),
            np.asarray([True, False, True]),
            top_k=1,
            exact_max_queries=0,
            hnsw_config={
                "metric": "l2",
                "m": 16,
                "ef_construction": 20,
                "ef_search": 20,
            },
        )
        self.assertEqual(result["queries"], 2)
        self.assertEqual(result["metrics"]["id_recall_at_1"], 1.0)

    def test_exact_max_queries_limits_only_exact_baseline(self):
        gallery = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32
        )
        queries = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32
        )
        result = _evaluate(
            "exact",
            queries,
            np.asarray([1, 2, 3]),
            gallery,
            np.asarray([1, 2, 3]),
            np.ones(3, dtype=bool),
            top_k=1,
            exact_max_queries=2,
            hnsw_config={
                "metric": "l2",
                "m": 16,
                "ef_construction": 20,
                "ef_search": 20,
            },
        )
        self.assertEqual(result["queries"], 2)


if __name__ == "__main__":
    unittest.main()
