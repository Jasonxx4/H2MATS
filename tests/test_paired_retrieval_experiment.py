import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np

from ann_index import ANNConfig, HierarchicalTrajectoryANN
from ann_pipeline import perturb_eval_command
from paired_retrieval_experiment import (
    evaluate_paired_queries,
    generate_perturbed_queries,
    paired_identification_metrics,
    remap_perturbed_source_ids,
)


class PerturbationGenerationTest(unittest.TestCase):
    def setUp(self):
        self.trajectories = np.asarray(
            [np.arange(start, start + length) for start, length in ((0, 12), (100, 15), (200, 18))],
            dtype=object,
        )
        self.coordinates = np.asarray(
            [np.column_stack((trajectory, trajectory + 0.5)) for trajectory in self.trajectories],
            dtype=object,
        )

    def test_queries_remain_aligned_and_reproducible(self):
        kwargs = dict(
            source_count=2,
            perturbations=("drop", "crop", "local_gap"),
            severities=(0.1, 0.3),
            seed=17,
            min_length=4,
        )
        first = generate_perturbed_queries(self.trajectories, self.coordinates, **kwargs)
        second = generate_perturbed_queries(self.trajectories, self.coordinates, **kwargs)

        self.assertEqual(len(first["trajectories"]), 12)
        np.testing.assert_array_equal(first["source_ids"], second["source_ids"])
        for idx, (trajectory, coordinate, source_id) in enumerate(
            zip(first["trajectories"], first["coordinates"], first["source_ids"])
        ):
            np.testing.assert_array_equal(trajectory, second["trajectories"][idx])
            np.testing.assert_allclose(coordinate[:, 0], trajectory)
            self.assertGreaterEqual(len(trajectory), 4)
            self.assertLess(len(trajectory), len(self.trajectories[source_id]))

    def test_sources_are_limited_to_gallery_prefix(self):
        payload = generate_perturbed_queries(
            self.trajectories,
            self.coordinates,
            source_count=0,
            perturbations=("drop",),
            severities=(0.2,),
            gallery_size=2,
        )
        self.assertEqual(set(payload["source_ids"]), {0, 1})

    def test_multiple_variants_are_generated_for_every_setting(self):
        payload = generate_perturbed_queries(
            self.trajectories,
            self.coordinates,
            source_count=2,
            perturbations=("drop", "crop", "local_gap"),
            severities=(0.1, 0.2, 0.3),
            variants_per_setting=4,
            seed=19,
        )

        self.assertEqual(len(payload["trajectories"]), 2 * 3 * 3 * 4)
        self.assertEqual(int(payload["variants_per_setting"]), 4)
        for source_id in np.unique(payload["source_ids"]):
            for perturbation in ("drop", "crop", "local_gap"):
                for severity in (0.1, 0.2, 0.3):
                    mask = (
                        (payload["source_ids"] == source_id)
                        & (payload["perturbation_types"] == perturbation)
                        & np.isclose(payload["severities"], severity)
                    )
                    np.testing.assert_array_equal(
                        payload["variant_ids"][mask], np.arange(4)
                    )

    def test_variants_per_setting_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "variants_per_setting"):
            generate_perturbed_queries(
                self.trajectories,
                self.coordinates,
                variants_per_setting=0,
            )

    def test_neighbor_replacement_updates_nodes_coordinates_and_metadata(self):
        trajectory = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
        trajectories = np.empty(1, dtype=object)
        trajectories[0] = trajectory
        coordinates = np.empty(1, dtype=object)
        coordinates[0] = np.column_stack((trajectory, trajectory + 0.5))
        adjacency = {
            0: np.asarray([1, 5]),
            1: np.asarray([0, 2, 5]),
            2: np.asarray([1, 3, 5]),
            3: np.asarray([2, 4, 5]),
            4: np.asarray([3, 5]),
            5: np.asarray([0, 1, 2, 3, 4]),
        }
        node_coordinates = {
            node: np.asarray([node * 10.0, node * 10.0 + 1.0], dtype=np.float32)
            for node in range(6)
        }
        payload = generate_perturbed_queries(
            trajectories,
            coordinates,
            source_count=1,
            perturbations=("neighbor_replace",),
            severities=(0.4,),
            min_length=2,
            seed=7,
            neighbor_adjacency=adjacency,
            node_coordinates=node_coordinates,
            neighbor_connectivity="strict",
        )

        self.assertEqual(payload["requested_replacement_counts"][0], 2)
        self.assertEqual(payload["actual_replacement_counts"][0], 2)
        self.assertEqual(payload["path_valid_ratios"][0], 1.0)
        self.assertAlmostEqual(payload["effective_severities"][0], 0.4)
        positions = payload["replaced_positions"][0]
        self.assertTrue(np.all(payload["trajectories"][0][positions] == 5))
        for position in positions:
            np.testing.assert_allclose(
                payload["coordinates"][0][position], node_coordinates[5]
            )

    def test_equal_length_queries_stay_one_dimensional_object_arrays(self):
        trajectories = [np.arange(10), np.arange(10, 20)]
        coordinates = [np.column_stack((item, item)) for item in trajectories]
        payload = generate_perturbed_queries(
            trajectories,
            coordinates,
            source_count=2,
            perturbations=("drop",),
            severities=(0.2,),
        )
        self.assertEqual(payload["trajectories"].shape, (2,))
        self.assertEqual(payload["coordinates"].shape, (2,))

    def test_source_ids_can_be_remapped_to_external_gallery_ids(self):
        payload = generate_perturbed_queries(
            self.trajectories,
            self.coordinates,
            source_count=2,
            perturbations=("drop",),
            severities=(0.2,),
            seed=3,
        )
        remapped = remap_perturbed_source_ids(
            payload, np.asarray([1001, 1007, 1013], dtype=np.int64)
        )
        external_ids = np.asarray([1001, 1007, 1013], dtype=np.int64)
        np.testing.assert_array_equal(
            remapped["source_ids"], external_ids[remapped["local_source_ids"]]
        )
        np.testing.assert_array_equal(
            remapped["local_source_ids"], payload["source_ids"]
        )


class PairedMetricTest(unittest.TestCase):
    def test_identification_metrics_use_known_source_rank(self):
        result = paired_identification_metrics(
            predictions=[np.array([3, 1]), np.array([2, 4]), np.array([9, 8])],
            source_ids=np.array([1, 2, 7]),
            ks=(1, 2),
        )
        self.assertAlmostEqual(result["id_recall_at_1"], 1.0 / 3.0)
        self.assertAlmostEqual(result["id_recall_at_2"], 2.0 / 3.0)
        self.assertAlmostEqual(result["mrr_at_2"], 0.5)
        self.assertEqual(result["misses_at_max_k"], 1)

    def test_paired_evaluation_does_not_exclude_source_id(self):
        embeddings = np.asarray([[0.0], [1.0]], dtype=np.float32)
        trajectories = np.asarray([np.array([0, 1]), np.array([2, 3])], dtype=object)

        def query_function(embedding, trajectory, top_k):
            return np.asarray([int(embedding[0])], dtype=np.int64)

        result = evaluate_paired_queries(
            query_function=query_function,
            embeddings=embeddings,
            trajectories=trajectories,
            source_ids=np.asarray([0, 1]),
            perturbation_types=np.asarray(["drop", "crop"]),
            severities=np.asarray([0.1, 0.2]),
            top_k=1,
        )
        self.assertEqual(result["id_recall_at_1"], 1.0)
        self.assertEqual(result["mrr_at_1"], 1.0)
        self.assertEqual(len(result["groups"]), 2)


class PairedPipelineIntegrationTest(unittest.TestCase):
    def test_saved_index_evaluates_all_three_methods(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            hierarchy_dir = os.path.join(temp_dir, "hierarchy")
            index_dir = os.path.join(temp_dir, "index")
            os.makedirs(hierarchy_dir)
            for name, values in (
                ("node_to_l1.npy", [0, 0, 1, 1, 2, 2]),
                ("node_to_l2.npy", [0, 0, 1, 1, 2, 2]),
                ("node_to_l3.npy", [0, 0, 1, 1, 2, 2]),
            ):
                np.save(os.path.join(hierarchy_dir, name), np.asarray(values, dtype=np.int64))

            trajectories = np.empty(3, dtype=object)
            trajectories[:] = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
            embeddings = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
            ann = HierarchicalTrajectoryANN(
                ANNConfig(backend="exact", global_fallback=3, exact_bucket_threshold=10)
            )
            ann.build(embeddings, trajectories, hierarchy_dir)
            ann.save(index_dir)

            query_file = os.path.join(temp_dir, "queries.npz")
            result_file = os.path.join(temp_dir, "results.json")
            query_trajectories = np.empty(2, dtype=object)
            query_trajectories[:] = [np.array([0]), np.array([2])]
            np.savez_compressed(
                query_file,
                embeddings=np.asarray([[0.1, 0.0], [9.9, 0.0]], dtype=np.float32),
                trajectories=query_trajectories,
                source_ids=np.asarray([0, 1], dtype=np.int64),
                perturbation_types=np.asarray(["drop", "drop"]),
                severities=np.asarray([0.1, 0.1], dtype=np.float32),
                gallery_size=np.asarray(3, dtype=np.int64),
            )
            args = argparse.Namespace(
                query_file=query_file,
                index_dir=index_dir,
                methods="hierarchical,global,exact",
                top_k=2,
                max_queries=0,
                exact_max_queries=0,
                result_file=result_file,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                perturb_eval_command(args)

            with open(result_file, "r", encoding="utf-8") as f:
                results = json.load(f)["results"]
            self.assertEqual([item["method"] for item in results], ["hierarchical", "global", "exact"])
            self.assertTrue(all(item["id_recall_at_1"] == 1.0 for item in results))


if __name__ == "__main__":
    unittest.main()
