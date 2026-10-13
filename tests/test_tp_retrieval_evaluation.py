import tempfile
import unittest
from pathlib import Path

import numpy as np

from tp_retrieval_evaluation import (
    close_mmap_array,
    compute_metric_distance_table,
    compute_metric_ground_truth,
    compute_tp_distance_table,
    compute_tp_ground_truth,
    tp_ground_truth_from_distance_table,
    tp_retrieval_metrics,
)


class TPRetrievalEvaluationTest(unittest.TestCase):
    def test_tp_truth_and_metrics_are_deterministic_and_cached(self):
        point_distances = np.asarray(
            [[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float32
        )
        queries = np.empty(2, dtype=object)
        queries[:] = [np.asarray([0, 1]), np.asarray([2])]
        gallery = np.empty(3, dtype=object)
        gallery[:] = [
            np.asarray([0, 1]),
            np.asarray([1, 2]),
            np.asarray([2]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            point_file = str(Path(directory) / "points.npy")
            cache_file = str(Path(directory) / "tp.npz")
            np.save(point_file, point_distances)
            truth, distances, report = compute_tp_ground_truth(
                queries,
                gallery,
                point_file,
                top_k=3,
                max_queries=0,
                cache_file=cache_file,
            )
            np.testing.assert_array_equal(truth[0], [0, 1, 2])
            np.testing.assert_array_equal(truth[1], [2, 1, 0])
            self.assertFalse(report["cache_reused"])
            self.assertEqual(float(distances[0, 0]), 0.0)
            cached_truth, _, cached_report = compute_tp_ground_truth(
                queries,
                gallery,
                point_file,
                top_k=3,
                max_queries=0,
                cache_file=cache_file,
            )
            np.testing.assert_array_equal(cached_truth, truth)
            self.assertTrue(cached_report["cache_reused"])

            metrics = tp_retrieval_metrics(
                [np.asarray([0, 2, 1]), np.asarray([1, 2, 0])], truth
            )
            self.assertAlmostEqual(metrics["r1_at_1"], 0.5)
            self.assertAlmostEqual(metrics["r1_at_10"], 1.0)
            self.assertEqual(metrics["queries"], 2)

    def test_distance_table_can_derive_subset_truth(self):
        point_distances = np.asarray(
            [[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float32
        )
        queries = np.empty(2, dtype=object)
        queries[:] = [np.asarray([0, 1]), np.asarray([2])]
        gallery = np.empty(3, dtype=object)
        gallery[:] = [
            np.asarray([0, 1]),
            np.asarray([1, 2]),
            np.asarray([2]),
        ]
        subset = np.asarray([0, 2], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            point_file = str(Path(directory) / "points.npy")
            table_file = str(Path(directory) / "tp_table.npy")
            np.save(point_file, point_distances)
            table, report = compute_tp_distance_table(
                queries,
                gallery,
                point_file,
                max_queries=0,
                cache_file=table_file,
            )
            self.assertFalse(report["cache_reused"])
            derived_truth, derived_distances, derived_report = (
                tp_ground_truth_from_distance_table(table, subset, top_k=2)
            )
            direct_truth, direct_distances, _ = compute_tp_ground_truth(
                queries,
                gallery[subset],
                point_file,
                top_k=2,
                max_queries=0,
            )
            np.testing.assert_array_equal(derived_truth, direct_truth)
            np.testing.assert_allclose(derived_distances, direct_distances)
            self.assertEqual(derived_report["queries_without_valid_neighbor"], 0)
            close_mmap_array(table)

    def test_non_tp_metrics_streaming_and_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point_file = str(root / "points.npy")
            np.save(
                point_file,
                np.asarray(
                    [[0, 1, 4], [1, 0, 2], [4, 2, 0]], dtype=np.float32
                ),
            )
            edge_file = root / "edge_weight.csv"
            edge_file.write_text(
                "s_node,e_node\n0,1\n1,0\n1,2\n2,1\n", encoding="utf-8"
            )
            queries = np.empty(1, dtype=object)
            queries[0] = np.asarray([0, 1])
            gallery = np.empty(3, dtype=object)
            gallery[:] = [np.asarray([0]), np.asarray([2]), np.asarray([1, 2])]

            for metric in ("DITA", "discret_frechet", "LCRS", "NetERP"):
                ids, distances, report = compute_metric_ground_truth(
                    queries,
                    gallery,
                    metric,
                    point_distance_file=point_file,
                    edge_file=str(edge_file),
                    node_count=3,
                    max_trajectory_length=2,
                    neterp_anchor_node=0,
                    top_k=2,
                    max_queries=1,
                    gallery_batch_size=2,
                )
                self.assertEqual(ids.shape, (1, 2))
                self.assertEqual(distances.shape, (1, 2))
                self.assertEqual(report["metric"], metric)

            table_file = str(root / "dita_table.npy")
            table, report = compute_metric_distance_table(
                queries,
                gallery,
                "DITA",
                point_distance_file=point_file,
                max_queries=1,
                gallery_batch_size=2,
                cache_file=table_file,
            )
            self.assertEqual(table.shape, (1, 3))
            self.assertEqual(report["metric"], "DITA")
            self.assertTrue(np.isfinite(table[:]).all())
            close_mmap_array(table)


if __name__ == "__main__":
    unittest.main()
