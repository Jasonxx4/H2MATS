import os
import tempfile
import unittest
import importlib.util
from unittest.mock import patch

import numpy as np
import torch

from binary_codes import (
    BinaryCodeEncoder,
    BinaryCodeIndexConfig,
    BinaryCodeIndex,
    CodeEncoderConfig,
    CodeTrainConfig,
    _prepare_metric_lists,
    _sample_triplets,
    hamming_distances,
    load_code_encoder,
    pack_binary_codes,
    save_code_encoder,
    semantic_code_loss,
    train_code_encoder,
)
from binary_code_pipeline import load_metric_candidate_teacher


class BinaryCodeTest(unittest.TestCase):
    def test_pack_and_hamming_distance(self):
        codes = np.asarray(
            [
                [1] * 16,
                [-1] * 16,
                [1, -1] * 8,
            ],
            dtype=np.float32,
        )
        packed = pack_binary_codes(codes)
        self.assertEqual(packed.shape, (3, 2))
        np.testing.assert_array_equal(hamming_distances(packed[0], packed), [0, 16, 8])

    def test_code_encoder_checkpoint_round_trip(self):
        torch.manual_seed(7)
        head = BinaryCodeEncoder(CodeEncoderConfig(input_dim=4, bits=16, hidden_dim=8, dropout=0.0))
        head.fit_input_normalization(np.arange(24, dtype=np.float32).reshape(6, 4))
        query = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        expected = head(query).detach().numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "head.pt")
            save_code_encoder(head, path, metadata={"test": True})
            loaded, metadata = load_code_encoder(path)
            np.testing.assert_allclose(loaded(query).detach().numpy(), expected)
            self.assertTrue(metadata["test"])


class BinaryCodeIndexTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.embeddings = rng.normal(size=(40, 4)).astype(np.float32)
        torch.manual_seed(11)
        self.head = BinaryCodeEncoder(
            CodeEncoderConfig(input_dim=4, bits=16, hidden_dim=8, dropout=0.0)
        )
        self.head.fit_input_normalization(self.embeddings)

    def test_query_and_save_load(self):
        config = BinaryCodeIndexConfig(
            bits=16,
            probe_radius=0,
            hamming_candidates=20,
            metric="l2",
            embedding_storage="float32",
        )
        index = BinaryCodeIndex(config).build(self.embeddings, self.head, encode_device="cpu")
        ids, distances, details = index.query(
            self.embeddings[7], top_k=3, return_details=True
        )
        self.assertEqual(int(ids[0]), 7)
        self.assertAlmostEqual(float(distances[0]), 0.0, places=6)
        self.assertLessEqual(len(details["candidate_ids"]), config.hamming_candidates)
        precomputed_ids, _ = index.query(
            self.embeddings[7], top_k=3, query_code=index.codes[7]
        )
        np.testing.assert_array_equal(precomputed_ids, ids)

        with tempfile.TemporaryDirectory() as directory:
            index.save(directory)
            loaded = BinaryCodeIndex.load(directory, mmap=False)
            loaded_ids, loaded_distances = loaded.query(self.embeddings[7], top_k=3)
            np.testing.assert_array_equal(loaded_ids, ids)
            np.testing.assert_allclose(loaded_distances, distances)

    def test_exclude_id(self):
        index = BinaryCodeIndex(
            BinaryCodeIndexConfig(
                bits=16,
                probe_radius=1,
                hamming_candidates=20,
                embedding_storage="float32",
            )
        ).build(self.embeddings, self.head, encode_device="cpu")
        ids, _ = index.query(self.embeddings[3], top_k=5, exclude_ids={3})
        self.assertNotIn(3, ids)

    def test_flat_matches_unpacked_hamming_with_ties_and_exclusions(self):
        rng = np.random.default_rng(27)
        for bits in (16, 128):
            codes = rng.integers(0, 256, size=(137, bits // 8), dtype=np.uint8)
            codes[40:48] = codes[0]
            query = codes[0]
            expected_distances = np.count_nonzero(
                np.unpackbits(codes, axis=1) != np.unpackbits(query), axis=1
            )
            for block_size in (1, 31, 65536):
                index = BinaryCodeIndex(BinaryCodeIndexConfig(
                    bits=bits, backend="flat", flat_block_size=block_size
                ))
                index.codes = codes
                for excluded in ([], [-1, 0, 0, 40, 200], list(range(len(codes)))):
                    allowed = np.asarray(
                        [i for i in range(len(codes)) if i not in excluded], dtype=np.int64
                    )
                    ordered = allowed[np.lexsort((allowed, expected_distances[allowed]))]
                    for k in (1, 7, 200):
                        ids, distances, details = index.hamming_search(
                            query, candidate_limit=k, exclude_ids=excluded
                        )
                        np.testing.assert_array_equal(ids, ordered[:k])
                        np.testing.assert_array_equal(distances, expected_distances[ids])
                        self.assertEqual(details["flat_scan_count"], len(codes))
                        self.assertTrue(details["used_full_scan"])

    def test_flat_code_only_preserves_hamming_order_without_embedding_rerank(self):
        index = BinaryCodeIndex(BinaryCodeIndexConfig(
            bits=16, backend="flat", flat_block_size=2, hamming_candidates=4,
        ))
        # Embedding order is the reverse of Hamming order. The closest binary
        # codes are in the final block, so a truncated scan cannot pass.
        index.codes = np.asarray([[255, 255], [3, 0], [1, 0], [0, 0]], dtype=np.uint8)
        index.embeddings = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        with patch.object(
            index, "rerank_continuous", side_effect=AssertionError("unexpected rerank")
        ):
            ids, distances, details = index.query(
                np.asarray([0.0]), top_k=3, query_code=np.asarray([0, 0], dtype=np.uint8),
                rerank_mode="code-only", return_details=True,
            )
        np.testing.assert_array_equal(ids, [3, 2, 1])
        np.testing.assert_array_equal(distances, [0, 1, 2])
        self.assertEqual(details["flat_scan_count"], 4)
        self.assertTrue(details["used_full_scan"])

    def test_flat_query_and_mmap_round_trip_without_mih_tables(self):
        index = BinaryCodeIndex(BinaryCodeIndexConfig(
            bits=16, backend="flat", flat_block_size=7,
            hamming_candidates=len(self.embeddings), embedding_storage="float32",
        )).build(self.embeddings, self.head, encode_device="cpu")
        self.assertIsNone(index.table_offsets)
        self.assertIsNone(index.table_postings)
        expected_distances = np.sum((self.embeddings - self.embeddings[7]) ** 2, axis=1)
        expected = np.argsort(expected_distances)[:5]
        ids, distances = index.query(self.embeddings[7], top_k=5)
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_allclose(distances, expected_distances[expected], atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            index.save(directory)
            self.assertFalse(os.path.exists(os.path.join(directory, "table_offsets.npy")))
            loaded = BinaryCodeIndex.load(directory, mmap=True)
            try:
                self.assertEqual(loaded.config.backend, "flat")
                loaded_ids, loaded_distances = loaded.query(self.embeddings[7], top_k=5)
                np.testing.assert_array_equal(loaded_ids, ids)
                np.testing.assert_array_equal(loaded_distances, distances)
            finally:
                loaded.codes._mmap.close()
                loaded.embeddings._mmap.close()

    @unittest.skipUnless(
        importlib.util.find_spec("hnswlib") is not None,
        "hnswlib is required for the HNSW code backend",
    )
    def test_hnsw_backend_search_and_save_load(self):
        config = BinaryCodeIndexConfig(
            bits=16,
            backend="hnsw",
            hamming_candidates=20,
            hnsw_ef_search=32,
            metric="l2",
            embedding_storage="float32",
        )
        index = BinaryCodeIndex(config).build(self.embeddings, self.head, encode_device="cpu")
        ids, distances, details = index.query(
            self.embeddings[7], top_k=3, return_details=True
        )
        self.assertEqual(int(ids[0]), 7)
        self.assertEqual(int(distances[0]), 0)
        self.assertGreater(details["hnsw_candidate_count"], 0)
        with tempfile.TemporaryDirectory() as directory:
            index.save(directory)
            loaded = BinaryCodeIndex.load(directory, mmap=False)
            loaded_ids, loaded_distances = loaded.query(self.embeddings[7], top_k=3)
            np.testing.assert_array_equal(loaded_ids, ids)
            np.testing.assert_array_equal(loaded_distances, distances)

    def test_hnsw_retries_with_smaller_k_when_graph_cannot_fill_request(self):
        class SparseFakeHnsw:
            def set_ef(self, value):
                self.ef = value

            def knn_query(self, vectors, k):
                if k > 4:
                    raise RuntimeError(
                        "Cannot return the results in a contiguous 2D array. "
                        "Probably ef or M is too small"
                    )
                return (
                    np.arange(k, dtype=np.int64).reshape(1, -1),
                    np.zeros((1, k), dtype=np.float32),
                )

        index = BinaryCodeIndex(
            BinaryCodeIndexConfig(
                bits=16,
                backend="hnsw",
                hamming_candidates=8,
                hnsw_ef_search=8,
            )
        )
        index.codes = np.zeros((10, 2), dtype=np.uint8)
        index.hnsw_index = SparseFakeHnsw()
        ids, distances, details = index.hamming_search(
            index.codes[0], candidate_limit=8, minimum_candidates=3
        )
        self.assertGreaterEqual(len(ids), 3)
        self.assertLess(len(ids), 8)
        self.assertGreater(details["hnsw_retry_count"], 0)
        self.assertEqual(details["hnsw_requested_count"], 8)
        self.assertEqual(len(distances), len(ids))

class CodeTrainingTest(unittest.TestCase):
    def test_cutoff_rank_loss_penalizes_reversed_boundary(self):
        config = CodeTrainConfig(
            embedding_distance_weight=0.0,
            ranking_weight=0.0,
            quantization_weight=0.0,
            balance_weight=0.0,
            independence_weight=0.0,
            cutoff_rank_weight=1.0,
            cutoff_rank_margin=0.1,
        )
        anchor = torch.tensor([[1.0, 1.0]])
        teacher = torch.zeros((1, 2))
        common = dict(
            anchor=anchor,
            positive=anchor,
            negative=-anchor,
            teacher_anchor=teacher,
            teacher_positive=teacher,
            teacher_negative=teacher,
            train_config=config,
            l2_scale=1.0,
        )
        good, good_parts = semantic_code_loss(
            **common,
            cutoff_positive_codes=anchor.unsqueeze(1),
            cutoff_negative_codes=(-anchor).unsqueeze(1),
        )
        bad, bad_parts = semantic_code_loss(
            **common,
            cutoff_positive_codes=(-anchor).unsqueeze(1),
            cutoff_negative_codes=anchor.unsqueeze(1),
        )
        self.assertLess(float(good), float(bad))
        self.assertEqual(float(good_parts["cutoff_rank"]), 0.0)
        self.assertGreater(float(bad_parts["cutoff_rank"]), 0.0)

    def test_candidate_teacher_loader_aligns_ragged_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "teacher.npz")
            np.savez(
                path,
                query_ids=np.asarray([0, 2]),
                offsets=np.asarray([0, 2, 3]),
                candidate_ids=np.asarray([1, 2, 0]),
                tp_distances=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            )
            labels, distances, report = load_metric_candidate_teacher(path, 3)
        np.testing.assert_array_equal(labels[0], [1, 2])
        np.testing.assert_array_equal(labels[1], [-1, -1])
        np.testing.assert_allclose(distances[2, 0], 0.3)
        self.assertEqual(report["query_count"], 2)

    def test_small_training_run(self):
        rng = np.random.default_rng(19)
        embeddings = rng.normal(size=(16, 6)).astype(np.float32)
        labels = np.asarray(
            [[(idx + offset) % len(embeddings) for offset in (1, 2, 3)] for idx in range(len(embeddings))],
            dtype=np.int64,
        )
        head, report = train_code_encoder(
            embeddings,
            labels,
            CodeEncoderConfig(input_dim=6, bits=16, hidden_dim=12, dropout=0.0),
            CodeTrainConfig(
                epochs=2,
                batch_size=4,
                validation_fraction=0.25,
                device="cpu",
            ),
        )
        self.assertEqual(head.config.bits, 16)
        self.assertEqual(len(report["history"]), 2)
        self.assertTrue(np.isfinite(report["best_validation_loss"]))

    def test_metric_hard_sampling_uses_ranked_tail_and_filters_same_source(self):
        labels = np.asarray(
            [
                [1, 2, 3, 4, 5],
                [0, 2, 3, 4, 5],
                [0, 1, 3, 4, 5],
                [0, 1, 2, 4, 5],
                [0, 1, 2, 3, 5],
                [0, 1, 2, 3, 4],
            ],
            dtype=np.int64,
        )
        distances = np.tile(
            np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            (len(labels), 1),
        )
        prepared = _prepare_metric_lists(labels, len(labels), 2, distances)
        anchors, positives, ranked, positive_distances, ranked_distances = prepared
        sampled = _sample_triplets(
            np.asarray([0]),
            anchors,
            positives,
            ranked,
            positive_distances,
            ranked_distances,
            len(labels),
            np.random.default_rng(4),
            positives_per_anchor=2,
            negatives_per_anchor=4,
            negative_strategy="metric-hard",
            hard_negative_pool=3,
            source_ids=np.asarray([0, 1, 2, 3, 3, 4]),
        )
        _, positive_ids, negative_ids, positive_metric, negative_metric = sampled
        self.assertEqual(positive_ids.shape, (1, 2))
        self.assertEqual(negative_ids.shape, (1, 4))
        self.assertTrue(set(positive_ids[0]).issubset({1, 2}))
        self.assertTrue(set(negative_ids[0]).issubset({3, 4, 5}))
        self.assertTrue(np.all(np.isfinite(positive_metric)))
        self.assertTrue(np.all(np.isfinite(negative_metric)))

    def test_metric_aware_multi_sample_training(self):
        rng = np.random.default_rng(23)
        embeddings = rng.normal(size=(20, 6)).astype(np.float32)
        labels = np.asarray(
            [
                [(idx + offset) % len(embeddings) for offset in range(1, 7)]
                for idx in range(len(embeddings))
            ],
            dtype=np.int64,
        )
        metric_distances = np.tile(
            np.linspace(0.1, 0.6, labels.shape[1], dtype=np.float32),
            (len(labels), 1),
        )
        _, report = train_code_encoder(
            embeddings,
            labels,
            CodeEncoderConfig(input_dim=6, bits=16, hidden_dim=12, dropout=0.0),
            CodeTrainConfig(
                epochs=2,
                batch_size=5,
                positive_pool=2,
                positives_per_anchor=2,
                negatives_per_anchor=3,
                negative_strategy="mixed",
                hard_negative_pool=4,
                metric_distance_weight=1.0,
                embedding_distance_weight=0.25,
                in_batch_negative_weight=0.25,
                graded_rank_weight=1.0,
                graded_rank_margin=0.05,
                graded_rank_cutoffs=(2, 4, 6),
                validation_fraction=0.2,
                device="cpu",
            ),
            metric_distances=metric_distances,
        )
        self.assertEqual(report["metric_distance_count"], metric_distances.size)
        self.assertGreater(report["metric_distance_scale"], 0.0)
        self.assertEqual(report["train_config"]["graded_rank_cutoffs"], (2, 4, 6))
        self.assertTrue(np.isfinite(report["best_validation_loss"]))

    def test_metric_distances_define_the_rank_tiers(self):
        labels = np.asarray([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        distances = np.asarray(
            [[0.3, 0.1, 0.2], [0.1, 0.3, 0.2],
             [0.2, 0.1, 0.3], [0.3, 0.2, 0.1]],
            dtype=np.float32,
        )
        _, positives, ranked, positive_distances, ranked_distances = (
            _prepare_metric_lists(labels, 4, 2, distances)
        )
        np.testing.assert_array_equal(ranked[0], [2, 3, 1])
        np.testing.assert_array_equal(positives[0], [2, 3])
        np.testing.assert_allclose(ranked_distances[0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(positive_distances[0], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
