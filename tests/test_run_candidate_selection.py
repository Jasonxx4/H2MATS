import tempfile
import unittest

import numpy as np

from run_candidate_selection import _write_low_confidence_comparison
from candidate_selection import (
    AdaptiveCandidateSelectionIndex,
    CandidateSelectionConfig,
)
from trajectory_metric_backend import TrajectoryMetricBackend


class _CodeConfig:
    normalize = False
    metric = "l2"
    hamming_candidates = 50


class _FakeCode:
    config = _CodeConfig()
    embeddings = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [3.0, 0.0]],
        dtype=np.float32,
    )

    def __init__(self):
        self.hamming_calls = 0

    def hamming_search(self, *args, **kwargs):
        self.hamming_calls += 1
        return (
            np.asarray([2, 3], dtype=np.int64),
            np.asarray([1, 2], dtype=np.int32),
            {"used_full_scan": False},
        )

    def rerank_continuous(self, query_embedding, candidate_ids, top_k):
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        difference = self.embeddings[candidate_ids] - query_embedding
        distances = np.einsum("ij,ij->i", difference, difference)
        order = np.argsort(distances, kind="stable")[:top_k]
        return candidate_ids[order], distances[order]


class _MetricCodeConfig:
    bits = 4
    normalize = False
    metric = "l2"
    hamming_candidates = 50


class _FakeMetricCode:
    config = _MetricCodeConfig()
    codes = np.asarray(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]],
        dtype=np.uint8,
    )

    def __init__(self):
        self.hamming_calls = 0

    def hamming_search(self, query_code, candidate_limit=None, **kwargs):
        self.hamming_calls += 1
        ids = np.asarray([0, 1], dtype=np.int64)
        limit = len(ids) if candidate_limit is None else min(len(ids), int(candidate_limit))
        return ids[:limit], np.arange(limit, dtype=np.int32), {
            "used_full_scan": False
        }


def _index():
    trajectories = np.empty(4, dtype=object)
    trajectories[:] = [
        np.asarray([0, 1, 2]),
        np.asarray([0, 1, 3]),
        np.asarray([4, 5, 6]),
        np.asarray([4, 5, 7]),
    ]
    hierarchy = {
        "node_to_l1": np.arange(8, dtype=np.int64),
        "node_to_l2": np.asarray([0, 0, 1, 1, 2, 2, 3, 3]),
        "node_to_l3": np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        "l3_edges": np.asarray([[0, 1]], dtype=np.int64),
    }
    config = CandidateSelectionConfig(
        minhash_size=4,
        postings_per_level=2,
        min_route_candidates=1,
        max_route_candidates=4,
        structural_candidates=2,
        global_candidates=2,
        max_rerank_candidates=4,
        adaptive_initial_global=1,
        adaptive_initial_structural=1,
        adaptive_max_global=2,
        adaptive_max_structural=2,
        adaptive_max_rerank=4,
        adaptive_overlap_threshold=1.0,
        adaptive_min_route_recall=1,
    )
    return AdaptiveCandidateSelectionIndex(_FakeCode(), hierarchy, config).build(trajectories)


class AdaptiveCandidateSelectionTest(unittest.TestCase):
    def test_low_confidence_comparison_persists_tracks_and_subset_metrics(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = type("Args", (), {"top_k": 2, "output_dir": output_dir})()
            trajectories = np.empty(3, dtype=object)
            trajectories[:] = [
                np.asarray([1, 2, 3]),
                np.asarray([4, 5]),
                np.asarray([6, 7, 8, 9]),
            ]
            labels = np.asarray([[0, 1], [1, 0], [2, 3]], dtype=np.int64)
            predictions = {
                "code-scan": [
                    np.asarray([0, 1]),
                    np.asarray([1, 0]),
                    np.asarray([0, 1]),
                ],
                "adaptive-dual-metric": [
                    np.asarray([0, 1]),
                    np.asarray([1, 0]),
                    np.asarray([2, 3]),
                ],
                "adaptive-dual-metric-no-rerank": [
                    np.asarray([1, 0]),
                    np.asarray([1, 0]),
                    np.asarray([2, 3]),
                ],
            }
            results = [
                {
                    "method": "code-scan",
                    "latency_ms": {"mean": 1.0},
                },
                {
                    "method": "adaptive-dual-metric",
                    "latency_ms": {"mean": 3.0},
                    "adaptive_tp_budget": {
                        "low_confidence_query_ids": [0, 2],
                        "selected_query_ids": [2],
                    },
                },
                {
                    "method": "adaptive-dual-metric-no-rerank",
                    "latency_ms": {"mean": 1.5},
                },
            ]
            report = _write_low_confidence_comparison(
                args,
                results,
                predictions,
                trajectories,
                np.asarray([10, 11, 12]),
                np.asarray([0, 1, 2]),
                labels,
            )
            self.assertEqual(report["eligible_queries"], 2)
            self.assertEqual(report["reranked_queries"], 1)
            self.assertGreater(
                report["subset_report"]["reranked"]["methods"]
                ["delta_rerank_minus_global"]["HR@50"],
                0.0,
            )
            self.assertIn(
                "dual-code-no-rerank",
                report["subset_report"]["eligible"]["methods"],
            )
            self.assertIn(
                "dual_code_no_rerank_predictions",
                np.load(report["artifact"], allow_pickle=False).files,
            )
            with np.load(report["artifact"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(
                    saved["eligible_query_indices"], np.asarray([0, 2])
                )
                np.testing.assert_array_equal(
                    saved["eligible_trajectory_offsets"], np.asarray([0, 3, 7])
                )
                np.testing.assert_array_equal(
                    saved["reranked_positions_in_eligible"], np.asarray([1])
                )

    def _dual_index(self):
        index = _index()
        index.metric_code_index = _FakeMetricCode()
        index.config.dual_metric_search_candidates = 2
        index.config.dual_metric_candidates = 2
        index.config.dual_metric_high_quality_tp_candidates = 2
        index.config.dual_metric_tp_candidates = 3
        index.config.dual_metric_hard_tp_candidates = 4
        index.config.dual_metric_head_protect = 2
        index.config.dual_metric_structural_protect = 0
        return index

    def test_dual_metric_high_confidence_skips_metric_head_and_tp(self):
        index = self._dual_index()
        index.config.adaptive_gate = "global"

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                raise AssertionError("high-confidence dual path must skip TP")

        ids, _, details = index.query_adaptive_dual_metric(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(4, dtype=np.uint8),
            np.zeros(4, dtype=np.uint8),
            top_k=1,
            metric_backend=Metric(),
            force_global_fast_path=True,
        )
        self.assertEqual(len(ids), 1)
        self.assertTrue(details["adaptive_global_fast_path"])
        self.assertTrue(details["adaptive_dual_metric"])
        self.assertTrue(details["metric_rerank_skipped"])
        self.assertEqual(index.metric_code_index.hamming_calls, 0)

    def test_dual_metric_low_confidence_merges_metric_only_candidate(self):
        index = self._dual_index()
        index.config.adaptive_gate = "hybrid"
        index.config.adaptive_global_margin = 2.0
        metric_calls = []

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                metric_calls.append(len(trajectories))
                return np.arange(len(trajectories), dtype=np.float32)

        _, _, details = index.query_adaptive_dual_metric(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(4, dtype=np.uint8),
            np.zeros(4, dtype=np.uint8),
            top_k=1,
            metric_backend=Metric(),
            force_metric_rerank=True,
        )
        self.assertEqual(index.metric_code_index.hamming_calls, 1)
        self.assertTrue(metric_calls)
        self.assertTrue(details["tp_rerank_applied"])
        self.assertTrue(details["adaptive_dual_metric"])
        self.assertTrue(
            set((0, 1)).intersection(set(details["metric_candidate_ids"]))
        )
        self.assertTrue(
            set(details["metric_candidate_ids"]).intersection(
                set(details["global_candidate_ids"])
            )
            != set(details["metric_candidate_ids"])
        )

    def test_dual_metric_no_rerank_skips_tp_backend(self):
        index = self._dual_index()
        index.config.adaptive_gate = "hybrid"

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                raise AssertionError("no-rerank path must skip TP")

        ids, _, details = index.query_adaptive_dual_metric(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(4, dtype=np.uint8),
            np.zeros(4, dtype=np.uint8),
            top_k=1,
            metric_backend=Metric(),
            dual_metric_no_rerank=True,
        )
        self.assertEqual(len(ids), 1)
        self.assertTrue(details["adaptive_dual_metric_no_rerank"])
        self.assertTrue(details["metric_rerank_skipped"])
        self.assertFalse(details["tp_rerank_applied"])
        self.assertEqual(index.metric_code_index.hamming_calls, 1)

    def test_dual_metric_protects_global_top50(self):
        index = self._dual_index()
        index.config.adaptive_gate = "hybrid"

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                return np.arange(len(trajectories), dtype=np.float32)

        _, _, details = index.query_adaptive_dual_metric(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(4, dtype=np.uint8),
            np.zeros(4, dtype=np.uint8),
            top_k=1,
            metric_backend=Metric(),
            force_metric_rerank=True,
        )
        self.assertIsNotNone(details["global_top50_ids"])
        self.assertTrue(
            set(details["global_top50_ids"]).issubset(
                set(details["metric_candidate_ids"])
            )
        )
        self.assertGreaterEqual(details["dual_metric_protected_global"], 1)

    def test_dual_metric_hard_query_uses_hard_tp_budget(self):
        index = self._dual_index()
        index.config.adaptive_gate = "hybrid"
        index.config.adaptive_global_margin = 2.0

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                return np.arange(len(trajectories), dtype=np.float32)

        _, _, details = index.query_adaptive_dual_metric(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(4, dtype=np.uint8),
            np.zeros(4, dtype=np.uint8),
            top_k=1,
            metric_backend=Metric(),
            global_cache_override={
                "global_ids": np.asarray([2, 3], dtype=np.int64),
                "global_distances": np.asarray([0, 0], dtype=np.int32),
                "global_details": {"used_full_scan": False},
            },
            force_metric_rerank=True,
        )
        self.assertEqual(details["dual_metric_tp_budget"], 4)
        self.assertEqual(details["metric_prefilter_candidate_count"], 4)

    def test_adaptive_upgrade_and_global_core_protection(self):
        index = _index()
        ids, _, details = index.query_adaptive(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(16, dtype=np.uint8),
            top_k=1,
        )
        self.assertEqual(details["adaptive_stage"], 1)
        self.assertGreaterEqual(details["global_core_count"], 1)
        self.assertLessEqual(details["union_candidate_count"], 4)
        self.assertEqual(len(ids), 1)
        self.assertEqual(index.code_index.hamming_calls, 1)

    def test_global_gate_skips_topology_feature_construction(self):
        index = _index()
        index.config.adaptive_gate = "global"

        def fail_if_called(_trajectory):
            raise AssertionError("global gate should not build topology features")

        index._features = fail_if_called
        ids, _, details = index.query_adaptive(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(16, dtype=np.uint8),
            top_k=1,
        )
        self.assertEqual(len(ids), 1)
        self.assertTrue(details["adaptive_global_fast_path"])
        self.assertEqual(details["structural_candidate_count"], 0)
        self.assertEqual(index.code_index.hamming_calls, 1)

    def test_global_gate_skips_expensive_metric_by_default(self):
        index = _index()
        index.config.adaptive_gate = "global"

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                raise AssertionError("high-confidence path must skip TP")

        ids, _, details = index.query_adaptive(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(16, dtype=np.uint8),
            top_k=1,
            metric_backend=Metric(),
        )
        self.assertEqual(len(ids), 1)
        self.assertTrue(details["metric_rerank_skipped"])
        self.assertEqual(details["adaptive_metric_fast_path"], "skip")

    def test_metric_prefilter_keeps_global_and_structural_protection(self):
        index = _index()
        index.config.metric_prefilter_candidates = 2
        index.config.metric_prefilter_global_protect = 1
        index.config.metric_prefilter_structural_protect = 1
        selected, details = index._metric_prefilter_candidates(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2, 3], dtype=np.int64),
            global_ids=np.asarray([2, 3], dtype=np.int64),
            structural_ids=np.asarray([0, 1], dtype=np.int64),
        )
        self.assertEqual(len(selected), 2)
        self.assertIn(2, selected)
        self.assertIn(0, selected)
        self.assertTrue(details["metric_prefilter_applied"])
        self.assertEqual(details["metric_prefilter_candidate_count"], 2)

    def test_metric_rerank_uses_prefiltered_pool(self):
        index = _index()
        index.config.metric_prefilter_candidates = 2
        index.config.metric_prefilter_global_protect = 1
        index.config.metric_prefilter_structural_protect = 1

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                self.count = len(trajectories)
                return np.arange(len(trajectories), dtype=np.float32)

        metric = Metric()
        ids, _, details = index.query_adaptive(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(16, dtype=np.uint8),
            top_k=1,
            metric_backend=metric,
        )
        self.assertEqual(len(ids), 1)
        self.assertEqual(metric.count, 2)
        self.assertEqual(details["metric_prefilter_candidate_count"], 2)
        self.assertEqual(len(details["metric_candidate_ids"]), 2)

    def test_optional_global_top50_protection_reaches_metric_pool(self):
        index = _index()
        index.config.adaptive_gate = "global"
        index.config.adaptive_metric_fast_path = "metric"
        index.config.adaptive_preserve_global_top50 = True
        index.config.metric_prefilter_candidates = 50
        index.config.metric_prefilter_high_quality_candidates = 50

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                return np.arange(len(trajectories), dtype=np.float32)

        _, _, details = index.query_adaptive(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(16, dtype=np.uint8),
            top_k=1,
            metric_backend=Metric(),
        )
        self.assertIsNotNone(details["global_top50_ids"])
        self.assertTrue(
            set(details["global_top50_ids"]).issubset(
                set(details["metric_candidate_ids"])
            )
        )

    def test_global_top50_preserve_skip_keeps_confident_query_off_tp(self):
        index = _index()
        index.config.adaptive_gate = "global"
        index.config.adaptive_metric_fast_path = "skip"
        index.config.adaptive_preserve_global_top50 = True

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                raise AssertionError("confident query should skip TP")

        _, _, details = index.query_adaptive(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(16, dtype=np.uint8),
            top_k=1,
            metric_backend=Metric(),
        )
        self.assertTrue(details["adaptive_global_fast_path"])
        self.assertTrue(details["metric_rerank_skipped"])
        self.assertFalse(details["tp_rerank_applied"])

    def test_global_top50_preserve_skip_runs_tp_only_for_difficult_query(self):
        index = _index()
        index.config.adaptive_gate = "hybrid"
        index.config.adaptive_metric_fast_path = "skip"
        index.config.adaptive_preserve_global_top50 = True
        index.config.adaptive_min_route_recall = 999
        index.config.adaptive_global_margin = 2.0

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                self.count = len(trajectories)
                return np.arange(len(trajectories), dtype=np.float32)

        metric = Metric()
        _, _, details = index.query_adaptive(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(16, dtype=np.uint8),
            top_k=1,
            metric_backend=metric,
        )
        self.assertTrue(details["adaptive_difficult"])
        self.assertTrue(details["tp_rerank_applied"])
        self.assertGreater(metric.count, 0)
        self.assertTrue(
            set(details["global_top50_ids"]).issubset(
                set(details["metric_candidate_ids"])
            )
        )

    def test_global_top50_is_embedding_ranked_from_baseline_hamming_pool(self):
        class CodeConfig:
            normalize = False
            metric = "l2"
            hamming_candidates = 60

        class CodeIndex:
            config = CodeConfig()
            embeddings = np.arange(70, dtype=np.float32).reshape(-1, 1)

            def hamming_search(self, *args, candidate_limit=None, **kwargs):
                ids = np.arange(69, -1, -1, dtype=np.int64)
                ids = ids[: min(int(candidate_limit), len(ids))]
                return ids, np.arange(len(ids), dtype=np.int32), {
                    "used_full_scan": False
                }

            def rerank_continuous(self, query_embedding, candidate_ids, top_k):
                candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
                distances = np.square(self.embeddings[candidate_ids, 0])
                order = np.argsort(distances, kind="stable")[:top_k]
                return candidate_ids[order], distances[order]

        config = CandidateSelectionConfig(
            adaptive_gate="global",
            adaptive_metric_fast_path="metric",
            adaptive_preserve_global_top50=True,
            metric_prefilter_candidates=64,
            metric_prefilter_high_quality_candidates=64,
            metric_prefilter_global_protect=32,
            metric_prefilter_structural_protect=16,
        )
        index = AdaptiveCandidateSelectionIndex(CodeIndex(), {}, config)
        index.trajectories = np.asarray(
            [np.asarray([row], dtype=np.int64) for row in range(70)],
            dtype=object,
        )

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                return np.arange(len(trajectories), dtype=np.float32)

        _, _, details = index.query_adaptive(
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            np.zeros(16, dtype=np.uint8),
            top_k=50,
            metric_backend=Metric(),
        )
        np.testing.assert_array_equal(details["global_top50_ids"], np.arange(10, 60))
        self.assertTrue(
            set(details["global_top50_ids"]).issubset(
                set(details["metric_candidate_ids"])
            )
        )

    def test_tp_backend_matches_symmetric_point_set_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            point_file = f"{directory}/points.npy"
            matrix = np.asarray(
                [[0, 2, 5], [2, 0, 3], [5, 3, 0]], dtype=np.float32
            )
            np.save(point_file, matrix)
            backend = TrajectoryMetricBackend("TP", point_distance_file=point_file)
            values = backend.distances(
                np.asarray([0, 1]), np.asarray([np.asarray([1, 2])], dtype=object)
            )
            self.assertEqual(float(values[0]), 3.0)
            del backend

    def test_lcrs_backend_counts_road_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            edge_file = f"{directory}/edge.csv"
            with open(edge_file, "w", encoding="utf-8") as stream:
                stream.write("s_node,e_node\n0,1\n1,2\n")
            backend = TrajectoryMetricBackend(
                "LCRS",
                edge_file=edge_file,
                node_count=3,
                max_trajectory_length=3,
            )
            values = backend.distances(
                np.asarray([0, 1, 2]),
                np.asarray([np.asarray([0, 1, 2])], dtype=object),
            )
            self.assertEqual(float(values[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
