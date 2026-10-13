import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_large_scale_head_training import assemble, build_truth


class BuildLargeScaleHeadTrainingTest(unittest.TestCase):
    def test_truth_stores_gallery_relative_topk_and_distances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point_file = root / "points.npy"
            query_file = root / "queries.npy"
            gallery_file = root / "gallery.npy"
            truth_file = root / "truth.npz"
            np.save(
                point_file,
                np.asarray(
                    [[0, 1, 4], [1, 0, 2], [4, 2, 0]], dtype=np.float32
                ),
            )
            queries = np.empty(1, dtype=object)
            queries[0] = np.asarray([0])
            gallery = np.empty(3, dtype=object)
            gallery[:] = [np.asarray([2]), np.asarray([1]), np.asarray([0])]
            np.save(query_file, queries, allow_pickle=True)
            np.save(gallery_file, gallery, allow_pickle=True)
            args = type("Args", (), {
                "query_file": str(query_file),
                "gallery_file": str(gallery_file),
                "point_distance_file": str(point_file),
                "output_file": str(truth_file),
                "top_k": 2,
                "max_queries": 0,
                "num_threads": 1,
                "device": "cpu",
                "gpu_batch_size": 2,
                "cache_file": None,
                "force_recompute": False,
            })()
            build_truth(args)
            with np.load(truth_file) as data:
                np.testing.assert_array_equal(data["gallery_ids"], [[2, 1]])
                np.testing.assert_allclose(data["metric_distances"], [[0.0, 1.0]])

    def test_assemble_offsets_gallery_labels_and_keeps_only_query_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_file = root / "query.npz"
            gallery_file = root / "gallery.npz"
            truth_file = root / "truth.npz"
            output_file = root / "head.npz"

            query_embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            gallery_embeddings = np.asarray(
                [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32
            )
            query_trajectories = np.empty(2, dtype=object)
            query_trajectories[:] = [np.asarray([0]), np.asarray([1])]
            gallery_trajectories = np.empty(3, dtype=object)
            gallery_trajectories[:] = [np.asarray([2]), np.asarray([3]), np.asarray([4])]
            np.savez(
                query_file,
                embeddings=query_embeddings,
                trajectories=query_trajectories,
                source_ids=np.asarray([100, 101]),
            )
            np.savez(
                gallery_file,
                embeddings=gallery_embeddings,
                trajectories=gallery_trajectories,
                source_ids=np.asarray([200, 201, 202]),
            )
            np.savez(
                truth_file,
                gallery_ids=np.asarray([[2, 0], [1, 2]], dtype=np.int64),
                metric_distances=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            )

            args = type("Args", (), {
                "query_embedding_file": str(query_file),
                "gallery_embedding_file": str(gallery_file),
                "truth_file": str(truth_file),
                "output_file": str(output_file),
                "include_trajectories": False,
            })()
            assemble(args)

            with np.load(output_file, allow_pickle=True) as data:
                self.assertEqual(data["embeddings"].shape, (5, 2))
                np.testing.assert_array_equal(
                    data["labels"],
                    [[4, 2], [3, 4]],
                )
                np.testing.assert_allclose(
                    data["metric_distances"], [[0.1, 0.2], [0.3, 0.4]]
                )
                self.assertEqual(int(data["anchor_count"]), 2)
                np.testing.assert_array_equal(
                    data["source_ids"], [100, 101, 200, 201, 202]
                )

    def test_assemble_preserves_missing_truth_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_file = root / "query.npz"
            gallery_file = root / "gallery.npz"
            truth_file = root / "truth.npz"
            output_file = root / "head.npz"

            np.savez(query_file, embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32))
            np.savez(
                gallery_file,
                embeddings=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            )
            np.savez(
                truth_file,
                gallery_ids=np.asarray([[-1, 1]], dtype=np.int64),
                metric_distances=np.asarray([[np.inf, 0.4]], dtype=np.float32),
            )

            args = type("Args", (), {
                "query_embedding_file": str(query_file),
                "gallery_embedding_file": str(gallery_file),
                "truth_file": str(truth_file),
                "output_file": str(output_file),
                "include_trajectories": False,
            })()
            assemble(args)

            with np.load(output_file, allow_pickle=True) as data:
                np.testing.assert_array_equal(data["labels"], [[-1, 2]])
                np.testing.assert_allclose(
                    data["metric_distances"], [[np.inf, 0.4]]
                )


if __name__ == "__main__":
    unittest.main()
