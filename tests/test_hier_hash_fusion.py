import os
import tempfile
import unittest

import numpy as np
import torch

from hash_global import BinaryHashHead, HashGlobalConfig, HashGlobalIndex, HashHeadConfig
from hier_hash_fusion import HierHashFusionConfig, HierHashFusionIndex


class HierHashFusionIndexTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(23)
        self.embeddings = rng.normal(size=(24, 6)).astype(np.float32)
        self.trajectories = np.empty(24, dtype=object)
        for idx in range(24):
            start = idx % 9
            self.trajectories[idx] = np.asarray([start, start + 1, start + 2], dtype=np.int64)
        torch.manual_seed(23)
        self.head = BinaryHashHead(
            HashHeadConfig(input_dim=6, bits=16, hidden_dim=12, dropout=0.0)
        )
        self.head.fit_input_normalization(self.embeddings)

    @staticmethod
    def _write_hierarchy(directory):
        np.save(os.path.join(directory, "node_to_l1.npy"), np.arange(12) // 2)
        np.save(os.path.join(directory, "node_to_l2.npy"), np.arange(12) // 4)
        np.save(os.path.join(directory, "node_to_l3.npy"), np.arange(12) // 6)
        np.save(os.path.join(directory, "l3_edge_index.npy"), np.asarray([[0, 1]], dtype=np.int64))

    def test_parallel_query_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            hierarchy_dir = os.path.join(directory, "hierarchy")
            hash_dir = os.path.join(directory, "hash")
            fusion_dir = os.path.join(directory, "fusion")
            os.makedirs(hierarchy_dir)
            self._write_hierarchy(hierarchy_dir)

            hash_index = HashGlobalIndex(
                HashGlobalConfig(
                    bits=16,
                    probe_radius=1,
                    hamming_candidates=12,
                    embedding_storage="float32",
                )
            ).build(self.embeddings, self.head, encode_device="cpu")
            hash_index.save(hash_dir)
            fusion = HierHashFusionIndex(
                hash_index,
                HierHashFusionConfig(
                    struct_hamming_candidates=8,
                    global_hash_candidates=8,
                    max_rerank_candidates=16,
                ),
            ).build(self.trajectories, hierarchy_dir)

            route_ids, route_details = fusion.route_candidate_ids(self.trajectories[5])
            self.assertGreater(len(route_ids), 0)
            self.assertGreater(route_details["bucket_count"], 0)
            ids, distances, details = fusion.query(
                self.embeddings[5],
                self.trajectories[5],
                top_k=5,
                return_details=True,
                query_code=hash_index.codes[5],
            )
            self.assertEqual(int(ids[0]), 5)
            self.assertAlmostEqual(float(distances[0]), 0.0, places=6)
            self.assertGreater(details["struct_candidate_count"], 0)
            self.assertGreater(details["global_candidate_count"], 0)
            self.assertLessEqual(details["rerank_candidate_count"], 16)

            fusion.save(fusion_dir, hash_dir)
            loaded = HierHashFusionIndex.load(fusion_dir, mmap=False)
            loaded_ids, loaded_distances = loaded.query(
                self.embeddings[5], self.trajectories[5], top_k=5
            )
            np.testing.assert_array_equal(loaded_ids, ids)
            np.testing.assert_allclose(loaded_distances, distances)

    def test_exclude_self(self):
        with tempfile.TemporaryDirectory() as hierarchy_dir:
            self._write_hierarchy(hierarchy_dir)
            hash_index = HashGlobalIndex(
                HashGlobalConfig(
                    bits=16,
                    probe_radius=1,
                    hamming_candidates=12,
                    embedding_storage="float32",
                )
            ).build(self.embeddings, self.head, encode_device="cpu")
            fusion = HierHashFusionIndex(hash_index).build(self.trajectories, hierarchy_dir)
            ids, _ = fusion.query(
                self.embeddings[3], self.trajectories[3], top_k=5, exclude_ids={3}
            )
            self.assertNotIn(3, ids)

    def test_hierarchical_routing_uses_l2_postings(self):
        with tempfile.TemporaryDirectory() as directory:
            hierarchy_dir = os.path.join(directory, "hierarchy")
            hash_dir = os.path.join(directory, "hash")
            fusion_dir = os.path.join(directory, "fusion")
            os.makedirs(hierarchy_dir)
            self._write_hierarchy(hierarchy_dir)

            hash_index = HashGlobalIndex(
                HashGlobalConfig(
                    bits=16,
                    probe_radius=1,
                    hamming_candidates=12,
                    embedding_storage="float32",
                )
            ).build(self.embeddings, self.head, encode_device="cpu")
            hash_index.save(hash_dir)
            fusion = HierHashFusionIndex(
                hash_index,
                HierHashFusionConfig(
                    routing_mode="hierarchical",
                    l2_top_regions=2,
                    min_route_candidates=8,
                    max_route_candidates=16,
                    bucket_max_size=4,
                    struct_hamming_candidates=8,
                    global_hash_candidates=8,
                    max_rerank_candidates=16,
                ),
            ).build(self.trajectories, hierarchy_dir)

            self.assertGreater(len(fusion.l2_to_ids), 0)
            self.assertGreater(len(fusion.routing_bucket_to_ids), 0)
            self.assertTrue(
                all(key.startswith(("l2:", "l1:")) for key in fusion.routing_bucket_to_ids)
            )
            route_ids, details = fusion.route_candidate_ids(self.trajectories[5])
            self.assertGreater(len(route_ids), 0)
            self.assertGreater(details["l2_count"], 0)
            self.assertLessEqual(len(route_ids), 16)
            self.assertGreaterEqual(details["route_pool_before_cap"], 8)

            _, _, query_details = fusion.query(
                self.embeddings[5],
                self.trajectories[5],
                top_k=5,
                return_details=True,
                query_code=hash_index.codes[5],
            )
            self.assertFalse(
                set(query_details["struct_unique_candidate_ids"])
                & set(query_details["global_candidate_ids"])
            )
            self.assertLessEqual(query_details["struct_unique_candidate_count"], 8)

            fusion.save(fusion_dir, hash_dir)
            loaded = HierHashFusionIndex.load(fusion_dir, mmap=False)
            self.assertEqual(loaded.config.routing_mode, "hierarchical")
            loaded_ids, loaded_details = loaded.route_candidate_ids(self.trajectories[5])
            np.testing.assert_array_equal(loaded_ids, route_ids)
            self.assertEqual(loaded_details, details)

if __name__ == "__main__":
    unittest.main()
