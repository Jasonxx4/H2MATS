import unittest

import numpy as np

from dual_metric_fusion import (
    BUCKETS,
    fit_weight_payload,
    validate_weight_payload,
)
from test_adaptive_topo_hash import _index, _FakeMetricHash


def _payload(weights):
    table = {bucket: list(weights) for bucket in BUCKETS}
    return {
        "version": 1,
        "mode": "learned-bucket",
        "metrics": {
            "TP": {
                "rerank": dict(table),
                "no_rerank": dict(table),
            }
        },
    }


class DualMetricFusionTest(unittest.TestCase):
    def test_payload_requires_simplex_compatible_weights(self):
        payload = validate_weight_payload(_payload((2.0, 1.0, 1.0)), "TP")
        self.assertEqual(payload["version"], 1)
        with self.assertRaises(ValueError):
            validate_weight_payload(_payload((1.0, -1.0, 1.0)), "TP")

    def test_index_uses_metric_bucket_and_variant_weights(self):
        index = _index()
        index.metric_hash_index = _FakeMetricHash()
        index.config.dual_metric_search_candidates = 2
        index.config.dual_metric_candidates = 2
        index.config.dual_metric_high_quality_tp_candidates = 2
        index.config.dual_metric_tp_candidates = 3
        index.config.dual_metric_hard_tp_candidates = 4
        index.config.dual_metric_head_protect = 0
        index.config.dual_metric_structural_protect = 0
        index.config.adaptive_gate = "hybrid"
        index.set_dual_metric_weight_payload(_payload((0.1, 0.8, 0.1)))

        class Metric:
            name = "TP"

            def distances(self, query, trajectories):
                raise AssertionError("no-rerank path must not call the TP backend")

        _, _, details = index.query_adaptive_dual_metric(
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0, 1, 2]),
            np.zeros(4, dtype=np.uint8),
            np.zeros(4, dtype=np.uint8),
            top_k=1,
            metric_backend=Metric(),
            dual_metric_no_rerank=True,
        )
        np.testing.assert_allclose(
            details["dual_metric_fusion_weights"],
            [0.1, 0.8, 0.1],
            atol=1e-6,
        )
        self.assertEqual(details["dual_metric_fusion_variant"], "no_rerank")
        self.assertIn(
            details["dual_metric_difficulty_bucket"], BUCKETS
        )

    def test_learning_moves_weight_toward_predictive_signal(self):
        samples = []
        for _ in range(12):
            # The first column perfectly ranks the oracle Top-1; the hash
            # column is reversed and topology carries no information.
            samples.append({
                "rank_features": np.asarray(
                    [[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]], dtype=np.float32
                ),
                "oracle_distances": np.asarray([0.0, 1.0], dtype=np.float32),
                "protected_mask": np.asarray([False, False]),
                "bucket": "ordinary",
            })
        learned = fit_weight_payload(
            {"rerank": samples, "no_rerank": samples},
            "TP",
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            per_bucket=False,
            top_k=1,
            epochs=80,
            learning_rate=0.10,
            max_pairs_per_query=8,
            seed=7,
        )
        weights = learned["metrics"]["TP"]["rerank"]["ordinary"]
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[0], 0.5)


if __name__ == "__main__":
    unittest.main()
