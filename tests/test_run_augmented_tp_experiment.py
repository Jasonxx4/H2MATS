import unittest
from pathlib import Path

from run_augmented_tp_experiment import (
    PROJECT_ROOT,
    _hnsw_command,
    _topology_command,
    make_parser,
)


class AugmentedExperimentCommandTest(unittest.TestCase):
    def test_topology_command_forwards_online_tp_and_global_comparison(self):
        args = make_parser().parse_args(
            [
                "--benchmark-dir", "benchmark",
                "--output-dir", "output",
                "--model", "model.pt",
                "--tp-distance-table-file", "tp_table.npy",
                "--rerank-metric", "TP",
                "--tp-point-distance-file", "points.npy",
                "--tp-num-threads", "16",
                "--include-global-hash",
                "--adaptive-metric-fast-path", "metric",
                "--adaptive-preserve-global-top50",
                "--hamming-candidates", "768",
                "--min-route-candidates", "512",
                "--max-route-candidates", "2048",
                "--hash-backend", "flat",
                "--hash-flat-block-size", "32768",
                "--global-candidates", "768",
                "--structural-candidates", "192",
                "--max-rerank-candidates", "960",
                "--adaptive-max-rerank", "960",
            ]
        )
        command = _topology_command(
            args,
            Path("gallery.npz"),
            Path("query.npz"),
            Path("hash.pt"),
            Path("results"),
        )

        def value(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(value("--methods"), "adaptive-topo-hash,global-hash")
        self.assertEqual(value("--rerank-metric"), "TP")
        self.assertEqual(value("--tp-num-threads"), "16")
        self.assertEqual(value("--hamming-candidates"), "768")
        self.assertEqual(value("--min-route-candidates"), "512")
        self.assertEqual(value("--max-route-candidates"), "2048")
        self.assertEqual(value("--hash-backend"), "flat")
        self.assertEqual(value("--hash-flat-block-size"), "32768")
        self.assertEqual(value("--global-candidates"), "768")
        self.assertEqual(value("--structural-candidates"), "192")
        self.assertEqual(value("--max-rerank-candidates"), "960")
        from topo_route_hash_large_scale import parse_args, _topo_config
        _topo_config(parse_args(command[2:])).validate()
        self.assertEqual(value("--adaptive-max-tp-fraction"), "0.15")
        self.assertTrue(value("--tp-point-distance-file").endswith("points.npy"))
        self.assertIn("--adaptive-preserve-global-top50", command)

    def test_dataset_defaults_are_not_hardcoded_to_beijing(self):
        args = make_parser().parse_args(
            [
                "--benchmark-dir", "benchmark",
                "--output-dir", "output",
                "--model", "model.pt",
                "--tp-distance-table-file", "tp_table.npy",
                "--dataset", "porto",
                "--distance-type", "DITA",
            ]
        )
        command = _hnsw_command(
            args,
            Path("gallery.npz"),
            Path("query.npz"),
            Path("results"),
        )
        point_file = command[command.index("--tp-point-distance-file") + 1]
        self.assertTrue(
            point_file.replace("\\", "/").endswith(
                "ground_truth/porto/Point_dis_matrix.npy"
            )
        )

    def test_dual_metric_topohash_forwards_both_heads_and_adaptive_budgets(self):
        args = make_parser().parse_args(
            [
                "--benchmark-dir", "benchmark",
                "--output-dir", "output",
                "--model", "model.pt",
                "--tp-distance-table-file", "tp_table.npy",
                "--dataset", "porto",
                "--topology-method", "adaptive-dual-metric",
                "--rerank-metric", "discret_frechet",
                "--hash-model-file", "candidate.pt",
                "--metric-hash-model-file", "metric.pt",
                "--dual-metric-weight-file", "weights.json",
                "--dual-metric-search-candidates", "320",
                "--dual-metric-candidates", "80",
                "--metric-prefilter-hash-score", "soft",
                "--include-global-hash",
            ]
        )
        command = _topology_command(
            args,
            Path("gallery.npz"),
            Path("query.npz"),
            Path("candidate.pt"),
            Path("results"),
        )

        def value(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(
            value("--methods"),
            "adaptive-dual-metric,adaptive-dual-metric-no-rerank,global-hash",
        )
        self.assertEqual(value("--metric-hash-model-file"), str(PROJECT_ROOT / "metric.pt"))
        self.assertEqual(value("--dual-metric-weight-file"), str(PROJECT_ROOT / "weights.json"))
        self.assertEqual(value("--dual-metric-search-candidates"), "320")
        self.assertEqual(value("--dual-metric-candidates"), "80")
        self.assertEqual(value("--metric-prefilter-hash-score"), "soft")


if __name__ == "__main__":
    unittest.main()
