import unittest

import torch

from hybrid_geo_network import HybridGeoTrajSimEncoder


def build_encoder(scale_aligned=True, coord_fusion="gated_attention"):
    return HybridGeoTrajSimEncoder(
        feature_size=3,
        embedding_size=4,
        hidden_size=8,
        num_layers=1,
        dropout_rate=0.0,
        concat=False,
        device=torch.device("cpu"),
        useLSTM=True,
        curvature=1.0,
        seq_encoder="hmt",
        coord_encoder="transformer",
        hmt_backbone="conv",
        hmt_layers=1,
        hmt_kernel_size=3,
        hmt_dropout=0.0,
        hmt_transformer_heads=2,
        hmt_scale_aligned_graph=scale_aligned,
        coord_fusion=coord_fusion,
    )


class ScaleAlignedGraphFusionTest(unittest.TestCase):
    def test_builds_four_node_aligned_graph_views(self):
        torch.manual_seed(23)
        encoder = build_encoder()
        hierarchy = {
            "node_to_l1": torch.tensor([0, 0, 1, 1, 2, 2]),
            "node_to_l2": torch.tensor([0, 0, 0, 0, 1, 1]),
            "node_to_l3": torch.tensor([0, 0, 0, 0, 0, 0]),
        }
        euclidean_outputs = [
            torch.randn(6, 4),
            torch.randn(3, 4),
            torch.randn(2, 4),
            torch.randn(1, 4),
        ]
        # Zero is a valid point in both the Poincare ball and its tangent space.
        hyperbolic_outputs = [torch.zeros_like(item) for item in euclidean_outputs]
        euclidean_views, hyperbolic_views = encoder.map_levels_to_original_nodes(
            euclidean_outputs,
            hyperbolic_outputs,
            hierarchy,
        )
        global_embedding = torch.randn(6, 4)
        scale_embeddings = encoder.build_scale_aligned_node_embeddings(
            global_embedding,
            euclidean_views,
            hyperbolic_views,
        )

        self.assertEqual([tuple(item.shape) for item in scale_embeddings], [(6, 4)] * 4)
        self.assertTrue(torch.equal(euclidean_views[1][0], euclidean_views[1][1]))
        self.assertTrue(torch.equal(euclidean_views[2][0], euclidean_views[2][3]))
        self.assertTrue(torch.equal(euclidean_views[3][0], euclidean_views[3][5]))

        encoder.last_euclidean_outputs = euclidean_outputs
        encoder.last_hyperbolic_tangent_outputs = hyperbolic_outputs
        trajectories = torch.tensor([[0, 2, 5]])
        batch_global = global_embedding[trajectories]
        batch_scales = encoder.build_scale_aligned_trajectory_inputs(
            batch_global,
            trajectories,
            hierarchy,
        )
        expected_scales = [item[trajectories] for item in scale_embeddings]
        self.assertTrue(all(
            torch.allclose(actual, expected, atol=1e-6)
            for actual, expected in zip(batch_scales, expected_scales)
        ))

    def test_legacy_mode_has_no_new_graph_parameters(self):
        encoder = build_encoder(scale_aligned=False, coord_fusion="late_concat")
        keys = encoder.state_dict().keys()

        self.assertFalse(any(key.startswith("scale_geometry_gates") for key in keys))
        self.assertFalse(any(key.startswith("coord_attention") for key in keys))


class CoordinateFusionTest(unittest.TestCase):
    def test_attention_pool_ignores_padding_values(self):
        torch.manual_seed(29)
        encoder = build_encoder()
        trajectory = torch.randn(2, 8)
        coordinate_last = torch.randn(2, 8)
        coordinate_outputs = torch.randn(2, 5, 8)
        padding_mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, False, False, True],
        ])
        changed_padding = coordinate_outputs.clone()
        changed_padding[padding_mask] = 1000.0

        output_a = encoder.fuse_trajectory_coordinate(
            trajectory,
            coordinate_last,
            coordinate_outputs,
            padding_mask,
        )
        output_b = encoder.fuse_trajectory_coordinate(
            trajectory,
            coordinate_last,
            changed_padding,
            padding_mask,
        )

        self.assertEqual(tuple(output_a.shape), (2, 16))
        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-6))

    def test_legacy_coordinate_fusion_is_exact_concatenation(self):
        encoder = build_encoder(scale_aligned=False, coord_fusion="late_concat")
        trajectory = torch.randn(2, 8)
        coordinate_last = torch.randn(2, 8)
        coordinate_outputs = torch.randn(2, 5, 8)
        padding_mask = torch.zeros((2, 5), dtype=torch.bool)

        output = encoder.fuse_trajectory_coordinate(
            trajectory,
            coordinate_last,
            coordinate_outputs,
            padding_mask,
        )

        self.assertTrue(torch.equal(output, torch.cat([trajectory, coordinate_last], dim=-1)))


if __name__ == "__main__":
    unittest.main()
