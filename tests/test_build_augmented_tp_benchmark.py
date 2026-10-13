import json
import pickle
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from build_augmented_tp_benchmark import (
    build_benchmark,
    compute_tp_table,
    _load_matching_result,
    _load_training_ids,
    make_parser,
)


class AugmentedBenchmarkHelpersTest(unittest.TestCase):
    def test_loads_torch_zip_payload_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matching_result.pt"
            records = [
                [[1, 2], [[116.0, 39.0], [116.1, 39.1]]],
                [[3, 4], [[116.2, 39.2], [116.3, 39.3]]],
            ]
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("matching_result/data.pkl", pickle.dumps(records))
            loaded = _load_matching_result(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0][0], [1, 2])

    def test_training_index_positions_map_to_filtered_raw_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shuffle_index_list.npy"
            np.save(path, np.asarray([2, 0, 1], dtype=np.int64))
            selected = _load_training_ids(path, [10, 20, 30], 2, 1953)
            np.testing.assert_array_equal(selected, [30, 10])

    def test_cli_defaults_match_requested_protocol(self):
        args = make_parser().parse_args(["build", "--output-dir", "out"])
        self.assertEqual(args.dataset, "beijing")
        self.assertEqual(args.target_size, 1_000_000)
        self.assertEqual(args.query_count, 2_000)
        self.assertEqual(args.train_count, 50_000)
        self.assertEqual(args.min_source_length, 10)

    def test_gallery_prefixes_keep_query_positives_with_sampling_or_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                [list(range(10)), [[float(idx), float(raw_id)] for idx in range(10)]]
                for raw_id in range(12)
            ]
            trajectory_file = root / "matching_result.pt"
            with zipfile.ZipFile(trajectory_file, "w") as archive:
                archive.writestr("matching_result/data.pkl", pickle.dumps(records))
            train_index_file = root / "shuffle_index_list.npy"
            np.save(train_index_file, np.asarray([0, 1], dtype=np.int64))

            for target_size in (6, 12):
                output_dir = root / f"gallery_{target_size}"
                args = make_parser().parse_args([
                    "build", "--dataset", "porto",
                    "--trajectory-file", str(trajectory_file),
                    "--train-index-file", str(train_index_file),
                    "--train-count", "2", "--query-count", "2",
                    "--target-size", str(target_size),
                    "--perturbations", "drop", "--output-dir", str(output_dir),
                ])
                build_benchmark(args)
                with np.load(output_dir / "gallery_metadata.npz") as gallery, np.load(
                    output_dir / "query_metadata.npz"
                ) as query:
                    self.assertEqual(len(gallery["source_ids"]), target_size)
                    np.testing.assert_array_equal(
                        gallery["source_ids"][:2], query["source_ids"]
                    )
                    self.assertTrue(np.all(gallery["is_initial"][:2]))
                    self.assertFalse(np.isin(gallery["source_ids"], [0, 1]).any())
                manifest = json.loads((output_dir / "manifest.json").read_text())
                self.assertEqual(manifest["real_non_training_pool_count"], 10)

    def test_tp_table_writes_query_by_gallery_memmap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gallery = np.empty(3, dtype=object)
            gallery[:] = [np.asarray([0]), np.asarray([2]), np.asarray([1, 2])]
            queries = np.empty(1, dtype=object)
            queries[0] = np.asarray([0, 1])
            np.save(root / "gallery_trajectories.npy", gallery, allow_pickle=True)
            np.save(root / "query_trajectories.npy", queries, allow_pickle=True)
            point_distances = np.asarray(
                [[0, 1, 4], [1, 0, 2], [4, 2, 0]], dtype=np.float32
            )
            point_path = root / "point_distances.npy"
            np.save(point_path, point_distances)
            output_path = root / "tp_table.npy"
            compute_tp_table(
                Namespace(
                    benchmark_dir=str(root),
                    point_distance_file=str(point_path),
                    output_file=str(output_path),
                    query_count=1,
                    num_threads=1,
                    device="cpu",
                    gpu_batch_size=16,
                    force_recompute=False,
                )
            )
            table = np.load(output_path, mmap_mode="r")
            self.assertEqual(table.shape, (1, 3))
            np.testing.assert_allclose(table[0], [1.0, 4.0, 2.0])
            report = json.loads((root / "tp_table.npy.json").read_text(encoding="utf-8"))
            self.assertEqual(report["shape"], [1, 3])
            self.assertEqual(report["gallery_order"], "gallery_trajectories.npy row order")
            table._mmap.close()

    def test_dataset_defaults_and_tdrive_csv_loader(self):
        args = make_parser().parse_args(
            ["build", "--dataset", "tdrive", "--output-dir", "out"]
        )
        self.assertEqual(args.dataset, "tdrive")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matching_result.csv"
            path.write_text(
                "Traj_id,Node_list,Coor_list\n"
                '0,"[1, 2]","[[1.0, 2.0], [1.1, 2.1]]"\n',
                encoding="utf-8",
            )
            loaded = _load_matching_result(path)
            self.assertEqual(loaded[0][0], [1, 2])
            self.assertEqual(loaded[0][1][1], [1.1, 2.1])


if __name__ == "__main__":
    unittest.main()
