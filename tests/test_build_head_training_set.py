import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_head_training_set import build_head_training_set


class BuildHeadTrainingSetTest(unittest.TestCase):
    def test_merges_rows_and_recomputes_local_topk_without_self(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point_file = root / "points.npy"
            train_file = root / "train.npz"
            validation_file = root / "validation.npz"
            output_file = root / "head.npz"

            np.save(
                point_file,
                np.asarray(
                    [
                        [0, 1, 4, 7],
                        [1, 0, 2, 5],
                        [4, 2, 0, 1],
                        [7, 5, 1, 0],
                    ],
                    dtype=np.float32,
                ),
            )
            train_trajectories = np.empty(2, dtype=object)
            train_coordinates = np.empty(2, dtype=object)
            train_trajectories[:] = [np.asarray([0]), np.asarray([1])]
            train_coordinates[:] = [np.asarray([[0.0, 0.0]]), np.asarray([[1.0, 0.0]])]
            validation_trajectories = np.empty(2, dtype=object)
            validation_coordinates = np.empty(2, dtype=object)
            validation_trajectories[:] = [np.asarray([2]), np.asarray([3])]
            validation_coordinates[:] = [np.asarray([[2.0, 0.0]]), np.asarray([[3.0, 0.0]])]
            np.savez(
                train_file,
                train_list=train_trajectories,
                train_coor=train_coordinates,
            )
            np.savez(
                validation_file,
                vali_list=validation_trajectories,
                vali_coor=validation_coordinates,
            )

            build_head_training_set(
                str(train_file),
                str(validation_file),
                str(output_file),
                str(point_file),
                top_k=2,
                gallery_batch_size=2,
            )

            with np.load(output_file, allow_pickle=True) as data:
                self.assertEqual(len(data["vali_list"]), 4)
                np.testing.assert_array_equal(data["vali_y"], [[1, 2], [0, 2], [3, 1], [2, 1]])
                for row_id, row in enumerate(data["vali_y"]):
                    self.assertNotIn(row_id, row)
                self.assertEqual(int(data["train_count"]), 2)
                self.assertEqual(int(data["validation_count"]), 2)
                self.assertEqual(str(data["training_scope"]), "train+validation")

    def test_accepts_regular_array_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point_file = root / "points.npy"
            train_file = root / "train.npz"
            validation_file = root / "validation.npz"
            output_file = root / "head.npz"
            np.save(point_file, np.zeros((2, 2), dtype=np.float32))
            np.savez(
                train_file,
                train_list=np.asarray([[0]], dtype=np.int64),
                train_coor=np.asarray([[[0.0, 0.0]]], dtype=np.float32),
            )
            np.savez(
                validation_file,
                vali_list=np.asarray([[1]], dtype=np.int64),
                vali_coor=np.asarray([[[1.0, 0.0]]], dtype=np.float32),
            )
            build_head_training_set(
                str(train_file),
                str(validation_file),
                str(output_file),
                str(point_file),
                top_k=1,
            )
            with np.load(output_file, allow_pickle=True) as data:
                self.assertEqual(len(data["vali_list"]), 2)
                np.testing.assert_array_equal(data["vali_y"], [[1], [0]])

    def test_expected_split_counts_are_upper_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point_file = root / "points.npy"
            train_file = root / "train.npz"
            validation_file = root / "validation.npz"
            output_file = root / "head.npz"
            np.save(point_file, np.zeros((2, 2), dtype=np.float32))
            np.savez(
                train_file,
                train_list=np.asarray([[0], [1]], dtype=np.int64),
                train_coor=np.asarray(
                    [[[0.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32
                ),
            )
            np.savez(
                validation_file,
                vali_list=np.asarray([[1]], dtype=np.int64),
                vali_coor=np.asarray([[[1.0, 0.0]]], dtype=np.float32),
            )
            build_head_training_set(
                str(train_file),
                str(validation_file),
                str(output_file),
                str(point_file),
                expected_train_count=15000,
                expected_validation_count=5000,
            )
            with np.load(output_file, allow_pickle=True) as data:
                self.assertEqual(int(data["train_count"]), 2)
                self.assertEqual(int(data["validation_count"]), 1)
                self.assertEqual(int(data["expected_train_count"]), 15000)
                self.assertEqual(int(data["expected_validation_count"]), 5000)

            too_small_limit_output = root / "head_too_large.npz"
            with self.assertRaisesRegex(
                ValueError, "Expected at most 1 training rows"
            ):
                build_head_training_set(
                    str(train_file),
                    str(validation_file),
                    str(too_small_limit_output),
                    str(point_file),
                    expected_train_count=1,
                    expected_validation_count=5000,
                )


if __name__ == "__main__":
    unittest.main()
