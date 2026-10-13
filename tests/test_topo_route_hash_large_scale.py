import importlib.util
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from topo_route_hash_large_scale import (
    CompactPostingIndex,
    TopoRouteHashConfig,
    TopoRouteHashIndex,
    AdaptiveTopoHashIndex,
    _compressed_level_path,
    _candidate_label_stats,
    _candidate_retention_stats,
    _build_metric_backend,
    _build_index,
    _method_comparison,
    _evaluate_method,
    _global_top50_diagnostics,
    _query_global_hash_tp,
    _minhash_signature,
    _select_low_confidence_queries,
    _path_tokens,
    parse_args,
    _token_hash,
)
from tp_retrieval_evaluation import tp_retrieval_metrics


class TopoRouteHashHelpersTest(unittest.TestCase):
    def test_large_scale_cli_accepts_metric_head_and_learned_weights(self):
        args = parse_args([
            "--gallery-file", "gallery.npz",
            "--query-file", "query.npz",
            "--hash-model-file", "candidate.pt",
            "--metric-hash-model-file", "metric.pt",
            "--dual-metric-weight-file", "weights.json",
            "--hierarchy-dir", "hierarchy",
            "--output-dir", "output",
            "--methods", "adaptive-dual-metric,global-hash",
            "--dual-metric-embedding-weight", "0.96",
            "--dual-metric-hash-weight", "0.02",
            "--dual-metric-topology-weight", "0.02",
        ])
        self.assertEqual(args.metric_hash_model_file, "metric.pt")
        self.assertEqual(args.dual_metric_weight_file, "weights.json")
        self.assertAlmostEqual(args.dual_metric_embedding_weight, 0.96)

    def test_global_hash_rerank_mode_is_explicit_and_preserves_default(self):
        base = [
            "--gallery-file", "gallery.npz",
            "--query-file", "query.npz",
            "--hash-model-file", "candidate.pt",
            "--hierarchy-dir", "hierarchy",
            "--output-dir", "output",
            "--methods", "global-hash",
            "--hash-backend", "flat",
        ]
        self.assertEqual(parse_args(base).global_hash_rerank_mode, "full")
        args = parse_args(base + ["--global-hash-rerank-mode", "hash-only"])
        self.assertEqual(args.global_hash_rerank_mode, "hash-only")
        tp_args = parse_args(base + [
            "--hamming-candidates", "64",
            "--global-hash-rerank-mode", "tp",
            "--rerank-metric", "TP",
        ])
        self.assertEqual(tp_args.hamming_candidates, 64)
        self.assertEqual(tp_args.global_hash_rerank_mode, "tp")
        self.assertEqual(tp_args.rerank_metric, "TP")

    def test_global_hash_tp_reranks_all_queries_and_matches_r1_plateau(self):
        class FakeHashIndex:
            def __init__(self):
                self.config = SimpleNamespace(hamming_candidates=64)
                self.calls = []
                # Reverse Hamming order to check the independent TP tie break.
                self.candidate_ids = np.arange(63, -1, -1, dtype=np.int64)

            def hamming_search(
                self, query_code, candidate_limit, minimum_candidates, exclude_ids
            ):
                self.calls.append((candidate_limit, minimum_candidates, exclude_ids))
                return (
                    self.candidate_ids.copy(),
                    np.arange(64, dtype=np.int32),
                    {
                        "candidate_ids": self.candidate_ids.copy(),
                        "used_full_scan": True,
                    },
                )

        class FakeTPBackend:
            name = "TP"

            def __init__(self):
                self.calls = []

            def distances(self, query_trajectory, gallery_trajectories):
                query_id = int(query_trajectory[0])
                candidate_ids = np.asarray(
                    [int(trajectory[0]) for trajectory in gallery_trajectories],
                    dtype=np.int64,
                )
                self.calls.append((query_id, len(candidate_ids)))
                if query_id == 100:
                    distances = np.ones(len(candidate_ids), dtype=np.float32)
                    distances[candidate_ids == 10] = 0.0
                    distances[candidate_ids == 20] = 0.0
                    distances[candidate_ids == 62] = np.inf
                    return distances
                return np.zeros(len(candidate_ids), dtype=np.float32)

        gallery_trajectories = np.empty(70, dtype=object)
        for row_id in range(len(gallery_trajectories)):
            gallery_trajectories[row_id] = np.asarray([row_id], dtype=np.int64)
        query_trajectories = np.empty(2, dtype=object)
        query_trajectories[0] = np.asarray([100], dtype=np.int64)
        query_trajectories[1] = np.asarray([101], dtype=np.int64)
        source_ids = np.arange(70, dtype=np.int64)
        source_ids[10] = 1000
        source_ids[69] = 1001
        hash_index = FakeHashIndex()
        backend = FakeTPBackend()
        index = SimpleNamespace(
            hash_index=hash_index,
            trajectories=gallery_trajectories,
            exclude_self=False,
        )

        result, predictions = _evaluate_method(
            "global-hash",
            index,
            np.zeros((2, 1), dtype=np.float32),
            query_trajectories,
            np.asarray([1000, 1001], dtype=np.int64),
            None,
            source_ids,
            np.ones(70, dtype=bool),
            query_codes=np.zeros((2, 1), dtype=np.uint8),
            top_k=50,
            metric_backend=backend,
            global_hash_rerank_mode="tp",
        )

        self.assertEqual(hash_index.calls, [(64, 50, None), (64, 50, None)])
        self.assertEqual(backend.calls, [(100, 64), (101, 64)])
        self.assertEqual(predictions[0][0], 10)  # TP tie breaks by Gallery row ID.
        self.assertNotIn(62, predictions[0])  # Non-finite TP distances are dropped.
        self.assertEqual(len(predictions[0]), 50)
        self.assertEqual(len(predictions[1]), 50)
        self.assertEqual(result["tp_rerank"]["queries"], 2)
        self.assertEqual(result["tp_rerank"]["fraction"], 1.0)
        self.assertFalse(result["tp_rerank"]["gated"])
        self.assertEqual(result["tp_rerank"]["candidate_count_min"], 64)
        self.assertEqual(result["tp_rerank"]["candidate_count_max"], 64)
        self.assertEqual(result["tp_rerank"]["invalid_candidate_count"], 1)
        self.assertEqual(
            result["tp_rerank"]["distance_source"], "online-trajectories"
        )

        tp_metrics = tp_retrieval_metrics(
            predictions,
            [np.asarray([10], dtype=np.int64), np.asarray([69], dtype=np.int64)],
        )
        self.assertEqual(tp_metrics["r1_at_1"], 0.5)
        self.assertEqual(tp_metrics["r1_at_10"], 0.5)
        self.assertEqual(tp_metrics["r1_at_50"], 0.5)

    def test_global_hash_tp_calls_real_trajectory_distance_backend(self):
        from hash_global import HashGlobalConfig

        class FakeHashIndex:
            config = HashGlobalConfig(
                bits=16, backend="flat", hamming_candidates=64
            )

            def hamming_search(
                self, query_code, candidate_limit, minimum_candidates, exclude_ids
            ):
                candidate_ids = np.arange(69, 5, -1, dtype=np.int64)
                return candidate_ids, np.zeros(64, dtype=np.int32), {
                    "candidate_ids": candidate_ids,
                    "used_full_scan": True,
                }

        with tempfile.TemporaryDirectory() as directory:
            point_distance_file = os.path.join(directory, "point_distances.npy")
            node_ids = np.arange(70, dtype=np.float32)
            np.save(point_distance_file, np.abs(node_ids[:, None] - node_ids[None, :]))
            backend = _build_metric_backend(SimpleNamespace(
                rerank_metric="TP",
                tp_point_distance_file=point_distance_file,
                tp_num_threads=1,
            ))
            try:
                trajectories = np.empty(70, dtype=object)
                for row_id in range(70):
                    trajectories[row_id] = np.asarray([row_id], dtype=np.int64)
                index = SimpleNamespace(
                    hash_index=FakeHashIndex(), trajectories=trajectories
                )
                ids, distances, details = _query_global_hash_tp(
                    index,
                    np.asarray([10], dtype=np.int64),
                    np.asarray([0, 0], dtype=np.uint8),
                    50,
                    backend,
                )
            finally:
                backend.close()

        self.assertEqual(ids[0], 10)
        self.assertEqual(len(ids), 50)
        self.assertTrue(np.all(np.diff(distances) >= 0.0))
        self.assertEqual(details["rerank_candidate_count"], 64)

    def test_global_hash_hash_only_mode_is_forwarded_to_index_query(self):
        class FakeHashIndex:
            def __init__(self):
                self.rerank_mode = None

            def query(self, query_embedding, **kwargs):
                self.rerank_mode = kwargs["rerank_mode"]
                return (
                    np.asarray([0], dtype=np.int64),
                    np.asarray([0], dtype=np.float32),
                    {
                        "candidate_ids": np.asarray([0], dtype=np.int64),
                        "used_full_scan": True,
                    },
                )

        hash_index = FakeHashIndex()
        index = SimpleNamespace(hash_index=hash_index, exclude_self=False)
        query_trajectories = np.empty(1, dtype=object)
        query_trajectories[0] = np.asarray([1, 2], dtype=np.int64)
        _evaluate_method(
            "global-hash",
            index,
            np.asarray([[0.0]], dtype=np.float32),
            query_trajectories,
            np.asarray([7], dtype=np.int64),
            None,
            np.asarray([7], dtype=np.int64),
            np.asarray([True]),
            query_codes=np.asarray([[0]], dtype=np.uint8),
            top_k=1,
            global_hash_rerank_mode="hash-only",
        )
        self.assertEqual(hash_index.rerank_mode, "hash-only")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch required")
    def test_global_hash_only_build_skips_hierarchy_and_topology_postings(self):
        from hash_global import BinaryHashHead, HashHeadConfig

        with tempfile.TemporaryDirectory() as directory:
            args = parse_args([
                "--gallery-file", "gallery.npz",
                "--query-file", "query.npz",
                "--hash-model-file", "candidate.pt",
                "--hierarchy-dir", f"{directory}/missing-hierarchy",
                "--output-dir", f"{directory}/output",
                "--methods", "global-hash",
                "--hash-backend", "flat",
                "--hash-device", "cpu",
                "--hamming-candidates", "5",
            ])
            trajectories = np.empty(2, dtype=object)
            trajectories[0] = np.asarray([0, 1], dtype=np.int64)
            trajectories[1] = np.asarray([1, 0], dtype=np.int64)
            gallery = {
                "embeddings": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                "trajectories": trajectories,
            }
            head = BinaryHashHead(HashHeadConfig(input_dim=2, bits=16, hidden_dim=8))
            scale_dir = f"{directory}/output/gallery_2"
            with patch("hash_global.load_hash_head", return_value=(head, {})):
                index, report = _build_index(
                    args, gallery, np.asarray([0, 1], dtype=np.int64), scale_dir
                )
            self.assertIsNone(index.metric_hash_index)
            self.assertEqual(report["topo_seconds"], 0.0)
            self.assertEqual(report["topo_disk_mb"], 0.0)
            self.assertFalse(os.path.exists(f"{scale_dir}/topo_index"))

    def test_select_low_confidence_queries_uses_fraction_cap(self):
        low, selected = _select_low_confidence_queries(
            np.asarray([0.01, 0.20, 0.03, 0.02, 0.40]),
            threshold=0.10,
            max_fraction=0.40,
        )
        np.testing.assert_array_equal(low, [True, False, True, True, False])
        np.testing.assert_array_equal(selected, [True, False, False, True, False])

    def test_config_defaults_tp_fraction_cap_to_fifteen_percent(self):
        self.assertEqual(TopoRouteHashConfig().adaptive_max_tp_fraction, 0.15)
        with self.assertRaises(ValueError):
            TopoRouteHashConfig(adaptive_max_tp_fraction=1.01).validate()

    def test_global_top50_diagnostics_separate_pool_and_final_retention(self):
        diagnostics = _global_top50_diagnostics(
            {
                "global_top50": [np.asarray([1, 2, 3])],
                "metric_pool": [np.asarray([1, 2, 3, 4])],
                "final": [np.asarray([1, 4])],
            },
            np.asarray([[1, 3, 4, -1]], dtype=np.int64),
        )
        self.assertEqual(diagnostics["global_top50_pool_retention"], 1.0)
        self.assertAlmostEqual(
            diagnostics["global_top50_final_retention"], 1.0 / 3.0
        )
        self.assertEqual(diagnostics["tp_truth50_pool_recall"], 1.0)

    def test_global_top50_diagnostics_reports_tp_fraction(self):
        diagnostics = _global_top50_diagnostics(
            {
                "queries": 2,
                "global_top50": [None, np.asarray([1, 2])],
                "metric_pool": [None, np.asarray([1, 2, 3])],
                "final": [np.asarray([0]), np.asarray([1, 3])],
                "tp_rerank_applied": [False, True],
            },
            np.asarray(
                [np.asarray([-1]), np.asarray([1, 3, 4])], dtype=object
            ),
        )
        self.assertEqual(diagnostics["queries"], 2)
        self.assertEqual(diagnostics["tp_rerank_queries"], 1)
        self.assertEqual(diagnostics["tp_rerank_fraction"], 0.5)
        self.assertEqual(diagnostics["global_top50_pool_retention"], 1.0)

    def test_method_comparison_accepts_adaptive_topology_results(self):
        comparison = _method_comparison(
            {
                "adaptive-topo-hash": [np.asarray([0], dtype=np.int64)],
                "global-hash": [np.asarray([1], dtype=np.int64)],
            },
            np.asarray([10], dtype=np.int64),
            np.asarray([10, 20], dtype=np.int64),
            np.ones(2, dtype=bool),
            top_k=1,
        )
        self.assertEqual(comparison["struct_rescue_at_1"], 1.0)
        self.assertEqual(comparison["global_only_hit_at_1"], 0.0)

    def test_large_scale_metric_backend_is_optional_and_supports_online_tp(self):
        self.assertIsNone(
            _build_metric_backend(SimpleNamespace(rerank_metric="embedding"))
        )
        with tempfile.TemporaryDirectory() as directory:
            point_file = f"{directory}/points.npy"
            np.save(point_file, np.eye(2, dtype=np.float32))
            backend = _build_metric_backend(
                SimpleNamespace(
                    rerank_metric="TP",
                    tp_point_distance_file=point_file,
                    tp_num_threads=1,
                )
            )
            self.assertEqual(backend.name, "TP")
            self.assertEqual(backend.tp_num_threads, 1)
            backend.close()

    def test_candidate_label_stats_measure_metric_truth_coverage(self):
        stats = _candidate_label_stats(
            [np.asarray([3, 5, 8]), np.asarray([2, 7])],
            np.asarray([[5, 6, 7], [1, 2, 3]], dtype=np.int64),
        )
        self.assertEqual(stats["truth_top1_hit_rate"], 0.5)
        self.assertAlmostEqual(stats["truth_top10_coverage"], 2.0 / 6.0)
        self.assertAlmostEqual(stats["truth_top50_coverage"], 2.0 / 6.0)

    def test_candidate_label_stats_align_sparse_metric_queries(self):
        labels = np.asarray(
            [
                [10, 11, 12],
                [20, 21, 22],
                [30, 31, 32],
                [40, 41, 42],
            ],
            dtype=np.int64,
        )
        stats = _candidate_label_stats(
            [np.asarray([21, 99]), np.asarray([40, 42])],
            labels,
            query_ids=np.asarray([1, 3]),
        )
        self.assertEqual(stats["truth_top1_hit_rate"], 0.5)
        self.assertAlmostEqual(stats["truth_top10_coverage"], 0.5)
        self.assertAlmostEqual(stats["truth_top50_coverage"], 0.5)

    def test_candidate_label_stats_reject_misaligned_query_ids(self):
        with self.assertRaises(ValueError):
            _candidate_label_stats(
                [np.asarray([1]), np.asarray([2])],
                np.asarray([[1], [2]], dtype=np.int64),
                query_ids=np.asarray([1]),
            )

    def test_candidate_retention_uses_the_same_sparse_query_rows(self):
        labels = np.asarray(
            [[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=np.int64
        )
        stats = _candidate_retention_stats(
            [np.asarray([20, 21, 22]), np.asarray([30, 31])],
            [np.asarray([20, 22]), np.asarray([31])],
            labels,
            np.asarray([1, 2]),
        )
        self.assertEqual(stats["truth_top1_retention"], 0.5)
        self.assertAlmostEqual(stats["truth_top10_retention"], 3.0 / 5.0)
        self.assertAlmostEqual(stats["truth_top50_retention"], 3.0 / 5.0)

    def test_hash_aware_metric_prefilter_uses_query_code(self):
        class HashConfig:
            normalize = False
            metric = "l2"

        class FakeHashIndex:
            config = HashConfig()
            embeddings = np.asarray(
                [[0.0], [0.1], [0.2], [10.0]], dtype=np.float32
            )
            codes = np.asarray(
                [[0], [255], [255], [0]], dtype=np.uint8
            )

        trajectories = np.empty(4, dtype=object)
        trajectories[:] = [np.asarray([0, 1])] * 4
        hierarchy = {
            "node_to_l1": np.arange(2, dtype=np.int64),
            "node_to_l2": np.zeros(2, dtype=np.int64),
            "node_to_l3": np.zeros(2, dtype=np.int64),
            "l3_edges": np.empty((0, 2), dtype=np.int64),
        }
        config = TopoRouteHashConfig(
            metric_prefilter_candidates=2,
            metric_prefilter_high_quality_candidates=2,
            metric_prefilter_global_protect=0,
            metric_prefilter_structural_protect=0,
            metric_prefilter_embedding_weight=0.0,
            metric_prefilter_hash_weight=1.0,
        )
        index = AdaptiveTopoHashIndex(FakeHashIndex(), hierarchy, config).build(
            trajectories
        )
        # The Global candidate code space ranks ID 3 first, whereas the
        # TP-distilled metric code space ranks IDs 1 and 2 first.
        index.metric_prefilter_codes = np.asarray(
            [[0], [0], [0], [255]], dtype=np.uint8
        )
        selected, details = index._metric_prefilter_candidates(
            np.asarray([0.0], dtype=np.float32),
            np.asarray([1, 2, 3], dtype=np.int64),
            budget_override=2,
            query_code=np.asarray([0], dtype=np.uint8),
        )
        np.testing.assert_array_equal(selected, [1, 2])
        self.assertEqual(details["metric_prefilter_candidate_count"], 2)

    def test_soft_hash_metric_prefilter_uses_continuous_metric_codes(self):
        class HashConfig:
            normalize = False
            metric = "l2"

        class FakeHashIndex:
            config = HashConfig()
            embeddings = np.asarray(
                [[0.0], [0.1], [0.2], [0.3]], dtype=np.float32
            )
            codes = np.zeros((4, 1), dtype=np.uint8)

        trajectories = np.empty(4, dtype=object)
        trajectories[:] = [np.asarray([0, 1])] * 4
        hierarchy = {
            "node_to_l1": np.arange(2, dtype=np.int64),
            "node_to_l2": np.zeros(2, dtype=np.int64),
            "node_to_l3": np.zeros(2, dtype=np.int64),
            "l3_edges": np.empty((0, 2), dtype=np.int64),
        }
        config = TopoRouteHashConfig(
            metric_prefilter_candidates=2,
            metric_prefilter_high_quality_candidates=2,
            metric_prefilter_global_protect=0,
            metric_prefilter_structural_protect=0,
            metric_prefilter_embedding_weight=0.0,
            metric_prefilter_hash_weight=1.0,
            metric_prefilter_hash_score="soft",
        )
        index = AdaptiveTopoHashIndex(FakeHashIndex(), hierarchy, config).build(
            trajectories
        )
        index.metric_prefilter_soft_codes = np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [0.9, 0.9], [-1.0, -1.0]],
            dtype=np.float16,
        )
        selected, _ = index._metric_prefilter_candidates(
            np.asarray([0.0], dtype=np.float32),
            np.asarray([1, 2, 3], dtype=np.int64),
            budget_override=2,
            query_code=np.asarray([1.0, 1.0], dtype=np.float16),
        )
        np.testing.assert_array_equal(selected, [1, 2])

    def test_level_paths_are_run_length_compressed_into_qgrams(self):
        assignment = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        path = _compressed_level_path(np.asarray([0, 1, 2, 3, 4]), assignment)
        np.testing.assert_array_equal(path, [0, 1, 2])
        self.assertEqual(_path_tokens(path, 2), [(0, 1), (1, 2)])

    def test_minhash_is_deterministic_and_changes_with_path_tokens(self):
        seeds = np.asarray([11, 13, 17, 19], dtype=np.uint64)
        left_keys = [_token_hash(3, (0, 1)), _token_hash(3, (1, 2))]
        right_keys = [_token_hash(3, (0, 1)), _token_hash(3, (1, 3))]
        first = _minhash_signature(left_keys, seeds)
        second = _minhash_signature(left_keys, seeds)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, _minhash_signature(right_keys, seeds)))

    def test_compact_postings_lookup_and_cap_candidates(self):
        postings = CompactPostingIndex.build(
            [np.asarray([1, 2]), np.asarray([1]), np.asarray([1, 3])],
            row_count=3,
            max_keys_per_row=2,
        )
        np.testing.assert_array_equal(postings.lookup(1), [0, 1, 2])
        self.assertEqual(len(postings.lookup_many([1, 2, 3], candidate_cap=2)), 2)

    def test_topology_route_can_rescue_a_positive_missing_from_global_hash(self):
        class HashConfig:
            normalize = False
            metric = "l2"

        class FakeHashIndex:
            config = HashConfig()
            embeddings = np.asarray(
                [[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [3.0, 0.0]],
                dtype=np.float32,
            )

            def hamming_search(self, *args, **kwargs):
                return (
                    np.asarray([2, 3]),
                    np.asarray([1, 2]),
                    {"used_full_scan": False},
                )

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
        config = TopoRouteHashConfig(
            minhash_size=4,
            postings_per_level=2,
            min_route_candidates=1,
            max_route_candidates=4,
            structural_candidates=2,
            global_candidates=2,
            max_rerank_candidates=4,
            embedding_weight=0.8,
        )
        index = TopoRouteHashIndex(FakeHashIndex(), hierarchy, config).build(
            trajectories
        )
        ids, _, details = index.query(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(16, dtype=np.uint8),
            top_k=1,
        )

        self.assertEqual(ids[0], 0)
        self.assertIn(0, details["structural_candidate_ids"])
        self.assertNotIn(0, details["global_candidate_ids"])
        self.assertTrue(details["top1_from_struct_only"])

    def test_global_only_mode_reranks_global_candidates_without_route_postings(self):
        class HashConfig:
            normalize = False
            metric = "l2"

        class FakeHashIndex:
            config = HashConfig()
            embeddings = np.asarray(
                [[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [3.0, 0.0]],
                dtype=np.float32,
            )

            def hamming_search(self, *args, **kwargs):
                return (
                    np.asarray([1, 2]),
                    np.asarray([1, 2]),
                    {"used_full_scan": False},
                )

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
        config = TopoRouteHashConfig(
            candidate_source="global-only",
            minhash_size=4,
            postings_per_level=2,
            min_route_candidates=1,
            max_route_candidates=4,
            structural_candidates=2,
            global_candidates=2,
            max_rerank_candidates=4,
            embedding_weight=0.8,
        )
        index = TopoRouteHashIndex(FakeHashIndex(), hierarchy, config).build(
            trajectories
        )
        ids, _, details = index.query(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(16, dtype=np.uint8),
            top_k=1,
        )

        np.testing.assert_array_equal(ids, [1])
        self.assertEqual(index.postings, {})
        self.assertEqual(details["route_after_cap"], 0)
        self.assertEqual(details["structural_candidate_count"], 0)
        self.assertEqual(details["global_candidate_count"], 2)
        self.assertEqual(details["union_candidate_count"], 2)
        self.assertEqual(details["final_struct_only"], 0)

    def test_grouped_metrics_follow_exact_query_limit_and_metadata(self):
        class HashConfig:
            normalize = False
            metric = "l2"

        class FakeHashIndex:
            config = HashConfig()

        class FakeIndex:
            hash_index = FakeHashIndex()

        query_embeddings = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [9.0, 9.0]], dtype=np.float32
        )
        gallery_embeddings = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], dtype=np.float32
        )
        trajectories = np.empty(3, dtype=object)
        trajectories[:] = [
            np.asarray([0, 1]),
            np.asarray([1, 2]),
            np.asarray([2, 3]),
        ]
        result, predictions = _evaluate_method(
            "exact",
            FakeIndex(),
            query_embeddings,
            trajectories,
            np.asarray([10, 20, 30]),
            gallery_embeddings,
            np.asarray([10, 20, 99]),
            np.ones(3, dtype=bool),
            np.zeros((3, 16), dtype=np.uint8),
            top_k=1,
            exact_max_queries=2,
            query_metadata={
                "perturbation_types": np.asarray(
                    ["neighbor_replace", "neighbor_replace", "neighbor_replace"]
                ),
                "severities": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                "effective_severities": np.asarray(
                    [0.125, 0.25, 0.375], dtype=np.float32
                ),
                "actual_replacement_counts": np.asarray([1, 2, 3]),
                "path_valid_ratios": np.asarray([1.0, 0.75, 0.5]),
            },
        )

        self.assertEqual(result["queries"], 2)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual([group["queries"] for group in result["groups"]], [1, 1])
        self.assertEqual(
            [group["metrics"]["id_recall_at_1"] for group in result["groups"]],
            [1.0, 1.0],
        )
        self.assertAlmostEqual(
            result["groups"][0]["effective_severities_mean"], 0.125
        )
        self.assertEqual(
            result["groups"][1]["actual_replacement_counts_mean"], 2.0
        )
        self.assertEqual(result["groups"][1]["path_valid_ratios_mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
