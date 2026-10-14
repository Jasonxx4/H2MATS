import unittest

import numpy as np

from hierarchical_code_large_scale import (
    _candidate_overlap_summary,
    _evaluate_method,
    gallery_composition,
    parse_gallery_sizes,
    remap_gallery_source_ids,
    select_gallery_rows,
    source_retrieval_metrics,
)


class HierarchicalCodeLargeScaleHelpersTest(unittest.TestCase):
    def test_candidate_overlap_summary_reports_exclusive_and_union_counts(self):
        summary = _candidate_overlap_summary(
            [np.asarray([1, 2, 2, 3]), np.asarray([5])],
            [np.asarray([2, 3, 4]), np.asarray([6, 7])],
        )

        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["struct_only"]["mean"], 1.0)
        self.assertEqual(summary["global_only"]["mean"], 1.5)
        self.assertEqual(summary["overlap"]["mean"], 1.0)
        self.assertEqual(summary["union_unique"]["mean"], 3.5)
        self.assertAlmostEqual(summary["overlap_rate"]["mean"], 0.25)

    def test_source_balanced_scales_are_nested_and_cover_queries(self):
        gallery_source_ids = np.asarray([0, 0, 1, 1, 2, 2, 90, 91, 92, 93])
        query_source_ids = np.asarray([0, 1, 2])

        small = select_gallery_rows(
            gallery_source_ids, query_source_ids, gallery_size=5, seed=17
        )
        large = select_gallery_rows(
            gallery_source_ids, query_source_ids, gallery_size=8, seed=17
        )

        self.assertTrue(set(small).issubset(set(large)))
        self.assertEqual(set(gallery_source_ids[small]) & {0, 1, 2}, {0, 1, 2})
        self.assertEqual(len(np.unique(small)), len(small))

    def test_source_balanced_rejects_too_small_gallery(self):
        with self.assertRaisesRegex(ValueError, "cannot preserve one positive"):
            select_gallery_rows(
                np.asarray([0, 0, 1, 1, 2, 2]),
                np.asarray([0, 1, 2]),
                gallery_size=2,
            )

    def test_prefix_selection_is_exact_prefix(self):
        rows = select_gallery_rows(
            np.asarray([5, 6, 7, 8]),
            np.asarray([5]),
            gallery_size=3,
            mode="prefix",
        )
        np.testing.assert_array_equal(rows, np.asarray([0, 1, 2]))

    def test_fixed_positive_scales_are_nested_and_exclude_extra_positives(self):
        gallery_source_ids = np.asarray(
            [0, 0, 0, 1, 1, 1, 90, 90, 91, 92, 93, 94]
        )
        query_source_ids = np.asarray([0, 1])

        small = select_gallery_rows(
            gallery_source_ids,
            query_source_ids,
            gallery_size=5,
            seed=17,
            mode="fixed-positive",
            positives_per_query=1,
        )
        large = select_gallery_rows(
            gallery_source_ids,
            query_source_ids,
            gallery_size=8,
            seed=17,
            mode="fixed-positive",
            positives_per_query=1,
        )

        self.assertTrue(set(small).issubset(set(large)))
        self.assertEqual(np.count_nonzero(gallery_source_ids[large] == 0), 1)
        self.assertEqual(np.count_nonzero(gallery_source_ids[large] == 1), 1)
        self.assertTrue(np.all(np.isin(gallery_source_ids[large][2:], [90, 91, 92, 93, 94])))

        composition = gallery_composition(
            gallery_source_ids[large], query_source_ids, positives_per_query=1
        )
        self.assertEqual(composition["positive_rows"], 2)
        self.assertEqual(composition["distractor_rows"], 6)
        self.assertEqual(composition["same_source_extra_rows"], 0)
        self.assertEqual(composition["sources_with_unexpected_positive_count"], 0)
        self.assertEqual(composition["query_positive_coverage"], 1.0)

    def test_composition_detects_uneven_positive_allocation(self):
        composition = gallery_composition(
            np.asarray([0, 0, 0, 1, 90]),
            np.asarray([0, 1]),
            positives_per_query=2,
        )

        self.assertEqual(composition["same_source_extra_rows"], 1)
        self.assertEqual(composition["sources_with_unexpected_positive_count"], 2)

    def test_fixed_positive_supports_multiple_positives(self):
        gallery_source_ids = np.asarray([0, 0, 0, 1, 1, 1, 90, 91])
        rows = select_gallery_rows(
            gallery_source_ids,
            np.asarray([0, 1]),
            gallery_size=6,
            seed=3,
            mode="fixed-positive",
            positives_per_query=2,
        )

        self.assertEqual(np.count_nonzero(gallery_source_ids[rows] == 0), 2)
        self.assertEqual(np.count_nonzero(gallery_source_ids[rows] == 1), 2)
        self.assertEqual(np.count_nonzero(gallery_source_ids[rows] >= 90), 2)

    def test_fixed_positive_rejects_insufficient_distractors(self):
        with self.assertRaisesRegex(ValueError, "independent distractor"):
            select_gallery_rows(
                np.asarray([0, 0, 1, 1]),
                np.asarray([0, 1]),
                gallery_size=3,
                mode="fixed-positive",
            )

    def test_fixed_positive_requires_every_query_source(self):
        with self.assertRaisesRegex(ValueError, "every Query source"):
            select_gallery_rows(
                np.asarray([0, 0, 90, 91]),
                np.asarray([0, 1]),
                gallery_size=3,
                mode="fixed-positive",
            )

    def test_initial_only_selection_keeps_one_clean_positive_per_source(self):
        gallery_source_ids = np.asarray([0, 0, 1, 1, 90])
        is_initial = np.asarray([True, False, True, False, True])
        rows = select_gallery_rows(
            gallery_source_ids,
            np.asarray([0, 1]),
            gallery_size=4,
            seed=11,
            mode="fixed-positive",
            positives_per_query=1,
            positive_mask=is_initial,
        )

        composition = gallery_composition(
            gallery_source_ids[rows],
            np.asarray([0, 1]),
            positives_per_query=1,
            positive_mask=is_initial[rows],
        )
        self.assertEqual(composition["positive_rows"], 2)
        self.assertEqual(composition["distractor_rows"], 2)
        self.assertEqual(composition["same_source_extra_rows"], 0)
        self.assertEqual(composition["sources_with_unexpected_positive_count"], 0)

    def test_multi_positive_source_metrics(self):
        gallery_source_ids = np.asarray([10, 20, 10, 30])
        query_source_ids = np.asarray([10, 20])
        predictions = [np.asarray([3, 0, 2]), np.asarray([0, 1, 3])]

        metrics = source_retrieval_metrics(
            predictions,
            query_source_ids,
            gallery_source_ids,
            ks=(1, 2, 3),
        )

        self.assertEqual(metrics["id_recall_at_1"], 0.0)
        self.assertEqual(metrics["id_recall_at_2"], 1.0)
        self.assertEqual(metrics["id_recall_at_3"], 1.0)
        self.assertEqual(metrics["mrr_at_3"], 0.5)
        self.assertEqual(metrics["positive_coverage_at_3"], 1.0)

    def test_initial_only_metrics_reject_same_source_perturbation(self):
        gallery_source_ids = np.asarray([10, 10, 20])
        is_initial = np.asarray([False, True, True])
        predictions = [np.asarray([0, 1]), np.asarray([2, 0])]

        metrics = source_retrieval_metrics(
            predictions,
            np.asarray([10, 20]),
            gallery_source_ids,
            ks=(1, 2),
            gallery_positive_mask=is_initial,
        )

        self.assertEqual(metrics["id_recall_at_1"], 0.5)
        self.assertEqual(metrics["id_recall_at_2"], 1.0)
        self.assertEqual(metrics["positive_coverage_at_2"], 1.0)

    def test_gallery_size_parser_supports_sweeps(self):
        self.assertEqual(
            parse_gallery_sizes(0, "200,50,100,100", available=500),
            [50, 100, 200],
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            parse_gallery_sizes(600, None, available=500)

    def test_exact_baseline_scans_float32_gallery(self):
        class Config:
            normalize = False
            metric = "l2"

        class CodeIndex:
            config = Config()

        class Index:
            code_index = CodeIndex()

        query_embeddings = np.asarray([[0.0, 0.0]], dtype=np.float32)
        query_trajectories = np.empty(1, dtype=object)
        query_trajectories[0] = np.asarray([0, 1], dtype=np.int64)
        gallery_embeddings = np.asarray(
            [[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32
        )
        result = _evaluate_method(
            "exact",
            Index(),
            query_embeddings,
            query_trajectories,
            np.asarray([7]),
            gallery_embeddings,
            np.asarray([8, 7, 9]),
            query_codes=np.zeros((1, 2), dtype=np.uint8),
            code_encoding_ms_per_query=0.0,
            top_k=2,
            exact_max_queries=0,
        )

        self.assertEqual(result["metrics"]["id_recall_at_1"], 1.0)
        self.assertEqual(result["metrics"]["mrr_at_2"], 1.0)

    def test_hier_overlap_uses_structural_candidates_before_novelty_filter(self):
        class Config:
            normalize = False
            metric = "l2"

        class CodeIndex:
            config = Config()

        class Index:
            code_index = CodeIndex()

            def query(self, *args, **kwargs):
                return np.asarray([0]), np.asarray([0.0]), {
                    "candidate_ids": np.asarray([0, 1, 2, 3]),
                    "union_before_cap_candidate_ids": np.asarray([0, 2, 1, 3]),
                    "struct_candidate_ids": np.asarray([0, 1]),
                    "struct_unique_candidate_ids": np.asarray([1, 3]),
                    "global_candidate_ids": np.asarray([0, 2]),
                    "route_pool_count": 4,
                    "route_pool_before_cap": 4,
                    "struct_candidate_count": 2,
                    "struct_unique_candidate_count": 2,
                    "global_candidate_count": 2,
                    "union_count_before_cap": 4,
                    "rerank_candidate_count": 4,
                    "pattern_count": 1,
                    "l2_count": 1,
                    "l3_count": 1,
                    "max_l3_hop": 0,
                    "bucket_count": 1,
                    "global_used_full_scan": False,
                }

        query_trajectories = np.empty(1, dtype=object)
        query_trajectories[0] = np.asarray([0, 1], dtype=np.int64)
        result = _evaluate_method(
            "hier-code",
            Index(),
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            query_trajectories,
            np.asarray([10]),
            gallery_embeddings=None,
            gallery_source_ids=np.asarray([10, 20, 30, 40]),
            query_codes=np.zeros((1, 2), dtype=np.uint8),
            code_encoding_ms_per_query=0.0,
            top_k=1,
            exact_max_queries=0,
        )

        overlap = result["candidate_overlap"]
        self.assertEqual(overlap["struct_only"]["mean"], 1.0)
        self.assertEqual(overlap["global_only"]["mean"], 1.0)
        self.assertEqual(overlap["overlap"]["mean"], 1.0)
        self.assertEqual(overlap["union_unique"]["mean"], 3.0)

    def test_legacy_gallery_ids_can_be_mapped_through_query_ids(self):
        gallery = {"source_ids": np.asarray([0, 1, 0], dtype=np.int64)}
        queries = {"source_ids": np.asarray([101, 205], dtype=np.int64)}
        remapped = remap_gallery_source_ids(gallery, queries, mode="query-row")
        np.testing.assert_array_equal(remapped["source_ids"], [101, 205, 101])


if __name__ == "__main__":
    unittest.main()
