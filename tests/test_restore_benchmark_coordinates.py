import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from restore_benchmark_coordinates import _restore_rows, main


class RestoreCoordinatesTests(unittest.TestCase):
    def test_clean_subsequence_and_neighbor_replacement(self):
        source_coordinates = np.asarray(
            [[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32
        )
        source = [(np.asarray([1, 2, 3, 4]), source_coordinates)]
        rows = np.empty(3, dtype=object)
        rows[:] = [
            np.asarray([1, 2, 3, 4]),
            np.asarray([1, 3, 4]),
            np.asarray([1, 9, 3, 4]),
        ]
        restored, report = _restore_rows(
            rows,
            np.asarray([0, 0, 0]),
            np.asarray(["clean", "drop", "neighbor_replace"]),
            source,
            {9: np.asarray([9, 90], dtype=np.float32)},
        )
        np.testing.assert_array_equal(restored[0], source_coordinates)
        np.testing.assert_array_equal(restored[1], source_coordinates[[0, 2, 3]])
        np.testing.assert_array_equal(
            restored[2], [[1, 10], [9, 90], [3, 30], [4, 40]]
        )
        self.assertEqual(report["max_replacements_in_one_row"], 1)

    def test_ambiguous_repeated_node_with_different_coordinates_fails(self):
        source = [(
            np.asarray([1, 2, 2, 3]),
            np.asarray([[1, 10], [2, 20], [2, 21], [3, 30]], dtype=np.float32),
        )]
        rows = np.empty(1, dtype=object)
        rows[0] = np.asarray([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "ambiguous matching positions"):
            _restore_rows(rows, np.asarray([0]), np.asarray(["drop"]), source, None)

    def test_source_id_must_refer_to_raw_trajectory(self):
        rows = np.empty(1, dtype=object)
        rows[0] = np.asarray([1, 2])
        with self.assertRaisesRegex(ValueError, "outside the original trajectory range"):
            _restore_rows(rows, np.asarray([3]), np.asarray(["clean"]), [], None)

    def test_cli_restores_rows_and_checks_gallery_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matching = root / "matching.csv"
            with matching.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["Node_list", "Coor_list"])
                writer.writeheader()
                writer.writerow({
                    "Node_list": repr([1, 2, 3, 4]),
                    "Coor_list": repr([[1, 10], [2, 20], [3, 30], [4, 40]]),
                })
            rows = np.empty(2, dtype=object)
            rows[:] = [np.asarray([1, 2, 3, 4]), np.asarray([1, 3, 4])]
            all_coordinates = np.asarray(
                [[1, 10], [2, 20], [3, 30], [4, 40], [1, 10], [3, 30], [4, 40]],
                dtype=np.float64,
            )
            stats = np.concatenate((all_coordinates.mean(axis=0), all_coordinates.std(axis=0)))
            payload = root / "old_embeddings.npz"
            np.savez_compressed(
                payload,
                embeddings=np.zeros((2, 2), dtype=np.float32),
                trajectories=rows,
                source_ids=np.asarray([0, 0]),
                perturbation_types=np.asarray(["clean", "drop"]),
                coordinate_stats=stats,
            )
            output = root / "coordinates.npy"
            import sys
            from unittest.mock import patch

            with patch.object(sys, "argv", [
                "restore_benchmark_coordinates.py",
                "--embedding-file", str(payload),
                "--trajectory-file", str(matching),
                "--output-file", str(output),
                "--check-gallery-stats",
            ]):
                main()
            restored = np.load(output, allow_pickle=True)
            np.testing.assert_array_equal(restored[1], [[1, 10], [3, 30], [4, 40]])


if __name__ == "__main__":
    unittest.main()
