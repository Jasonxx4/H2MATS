import unittest

import torch
from torch_geometric.data import Data

from graph_normalization import distance_weighted_gcn_norm
from hybrid_geo_network import (
    EuclideanLevelGNN,
    HyperbolicHierarchyGNN,
    HybridGeoTrajSimEncoder,
    NodeAdaptiveScaleFusion,
    SimpleTwoLayerGCN,
    sample_negative_parent_indices,
)
from setting import SetParameter


class DistanceWeightedGCNNormTest(unittest.TestCase):
    def test_shorter_edge_has_larger_normalized_affinity(self):
        edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
        edge_length = torch.tensor([1.0, 10.0])

        normalized_index, normalized_weight, scale = distance_weighted_gcn_norm(
            edge_index,
            edge_length,
            num_nodes=3,
            scale=1.0,
        )

        self.assertEqual(tuple(normalized_index.shape), (2, 5))
        self.assertGreater(float(normalized_weight[0]), float(normalized_weight[1]))
        self.assertTrue(torch.isfinite(normalized_weight).all())
        self.assertEqual(float(scale), 1.0)

    def test_euclidean_encoder_caches_normalization_on_legacy_graphs(self):
        graphs = []
        for _ in range(4):
            graphs.append(
                Data(
                    x=torch.randn(4, 3),
                    edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                    edge_attr=torch.tensor([1.0, 2.0, 3.0]),
                )
            )
        encoder = EuclideanLevelGNN(input_size=3, hidden_size=5, dropout=0.0)

        outputs = encoder(graphs)
        cached_pointers = [graph.gcn_edge_norm.data_ptr() for graph in graphs]
        encoder(graphs)

        self.assertEqual([tuple(item.shape) for item in outputs], [(4, 5)] * 4)
        self.assertEqual(cached_pointers, [graph.gcn_edge_norm.data_ptr() for graph in graphs])

    def test_euclidean_encoder_uses_optional_transition_edge_features(self):
        graphs = []
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        for _ in range(4):
            graphs.append(Data(
                x=torch.randn(4, 3),
                edge_index=edge_index,
                edge_attr=torch.tensor([1.0, 2.0, 3.0]),
                transition_attr=torch.tensor([
                    [0.8, 0.1, 0.7, 0.2],
                    [0.2, 0.4, 0.1, 0.3],
                    [0.5, 0.0, 0.6, 0.1],
                ]),
            ))
        encoder = EuclideanLevelGNN(input_size=3, hidden_size=5, dropout=0.0)

        outputs = encoder(graphs)

        self.assertEqual([tuple(item.shape) for item in outputs], [(4, 5)] * 4)
        self.assertTrue(all(
            tuple(graph.gcn_transition_attr.shape) == (7, 4)
            for graph in graphs
        ))
        self.assertTrue(torch.isfinite(torch.cat(outputs)).all())

    def test_simple_two_layer_gcn_uses_and_caches_l0_normalization(self):
        graph = Data(
            x=torch.randn(6, 3),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
                dtype=torch.long,
            ),
            edge_attr=torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 1.0]),
        )
        encoder = SimpleTwoLayerGCN(input_size=3, hidden_size=5, dropout=0.0)

        output = encoder(graph)
        cached_pointer = graph.gcn_edge_norm.data_ptr()
        encoder(graph)

        self.assertEqual(tuple(output.shape), (6, 5))
        self.assertEqual(cached_pointer, graph.gcn_edge_norm.data_ptr())


class SimpleGCNAblationTest(unittest.TestCase):
    @staticmethod
    def build_encoder(scale_aligned=False):
        return HybridGeoTrajSimEncoder(
            feature_size=3,
            embedding_size=4,
            hidden_size=8,
            num_layers=1,
            dropout_rate=0.0,
            concat=False,
            device=torch.device("cpu"),
            useLSTM=False,
            seq_encoder="hmt",
            hmt_backbone="conv",
            hmt_layers=1,
            hmt_kernel_size=3,
            hmt_dropout=0.0,
            hmt_transformer_heads=2,
            hmt_scale_aligned_graph=scale_aligned,
            coord_fusion="late_concat",
            graph_encoder="simple_gcn",
        )

    def test_simple_gcn_exposes_only_simple_graph_parameters(self):
        encoder = self.build_encoder()
        hierarchy = {
            "level_graphs": [
                Data(
                    x=torch.randn(6, 3),
                    edge_index=torch.tensor(
                        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
                        dtype=torch.long,
                    ),
                    edge_attr=torch.ones(6),
                )
            ]
        }

        output = encoder.encode_graph(hierarchy)
        graph_keys = set(encoder.graph_state_dict())

        self.assertEqual(tuple(output.shape), (6, 4))
        self.assertTrue(all(key.startswith("simple_gcn.") for key in graph_keys))
        self.assertFalse(hasattr(encoder, "euclidean_gnn"))
        self.assertIsNone(encoder.last_hyperbolic_outputs)

        encoder.freeze_graph_encoder(cache=False)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.graph_parameters()))
        encoder.unfreeze_graph_encoder()
        self.assertTrue(all(parameter.requires_grad for parameter in encoder.graph_parameters()))

    def test_simple_gcn_rejects_scale_aligned_graph_inputs(self):
        with self.assertRaisesRegex(ValueError, "hmt_scale_aligned_graph=False"):
            self.build_encoder(scale_aligned=True)


class NegativeParentSamplingTest(unittest.TestCase):
    def test_negative_parent_is_always_different(self):
        torch.manual_seed(19)
        true_parent_ids = torch.arange(4).repeat(250)
        negative_ids = sample_negative_parent_indices(true_parent_ids, parent_count=4)

        self.assertTrue(torch.all(negative_ids != true_parent_ids))
        self.assertGreaterEqual(int(negative_ids.min()), 0)
        self.assertLess(int(negative_ids.max()), 4)

    def test_single_parent_level_has_no_valid_negative(self):
        self.assertIsNone(sample_negative_parent_indices(torch.zeros(5, dtype=torch.long), 1))

    def test_multiple_negative_parents_are_valid(self):
        true_parent_ids = torch.tensor([0, 1, 2, 3])
        negative_ids = sample_negative_parent_indices(
            true_parent_ids,
            parent_count=4,
            num_negatives=8,
        )

        self.assertEqual(tuple(negative_ids.shape), (4, 8))
        self.assertTrue(torch.all(negative_ids != true_parent_ids.unsqueeze(-1)))


class OptimizedHierarchyGNNTest(unittest.TestCase):
    def test_bidirectional_attention_blocks_propagate_gradients(self):
        torch.manual_seed(31)
        outputs = [
            torch.randn(6, 4, requires_grad=True),
            torch.randn(3, 4, requires_grad=True),
            torch.randn(2, 4, requires_grad=True),
            torch.randn(1, 4, requires_grad=True),
        ]
        hierarchy = {
            "node_to_l1": torch.tensor([0, 0, 1, 1, 2, 2]),
            "l1_to_l2": torch.tensor([0, 0, 1]),
            "l2_to_l3": torch.tensor([0, 0]),
        }
        encoder = HyperbolicHierarchyGNN(
            hidden_size=4,
            num_blocks=2,
            dropout=0.0,
        )

        result = encoder(outputs, hierarchy)
        sum(item.sum() for item in result).backward()

        self.assertEqual([tuple(item.shape) for item in result], [(6, 4), (3, 4), (2, 4), (1, 4)])
        self.assertEqual(len(encoder.blocks), 2)
        self.assertTrue(all(item.grad is not None for item in outputs))
        self.assertIsNotNone(encoder.blocks[0].child_query[0].weight.grad)

    def test_node_adaptive_scale_fusion_returns_normalized_weights(self):
        torch.manual_seed(37)
        fusion = NodeAdaptiveScaleFusion(
            embedding_size=4,
            dropout=0.0,
            scale_dropout=0.0,
        )
        euclidean = [torch.randn(7, 4) for _ in range(4)]
        hyperbolic = [torch.randn(7, 4) for _ in range(4)]

        output, scale_views, weights = fusion(euclidean, hyperbolic)

        self.assertEqual(tuple(output.shape), (7, 4))
        self.assertEqual([tuple(item.shape) for item in scale_views], [(7, 4)] * 4)
        self.assertEqual(tuple(weights.shape), (7, 4))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(7), atol=1e-6))

    def test_graph_scale_balance_loss_is_finite_for_fp16_zero_weights(self):
        encoder = HybridGeoTrajSimEncoder(
            feature_size=3,
            embedding_size=4,
            hidden_size=8,
            num_layers=1,
            dropout_rate=0.0,
            concat=False,
            device=torch.device("cpu"),
            useLSTM=False,
            seq_encoder="hmt",
            hmt_backbone="conv",
            hmt_layers=1,
            hmt_kernel_size=3,
            hmt_dropout=0.0,
            hmt_transformer_heads=2,
        )
        encoder.last_graph_scale_weights = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float16,
        )

        loss = encoder.graph_scale_balance_loss()

        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))

    def test_default_configuration_enables_optimized_hierarchy(self):
        config = SetParameter()

        self.assertEqual(config.gtraj["graphEncoder"], "hybrid_multilevel")
        self.assertTrue(config.gtraj["hmtScaleAlignedGraph"])
        self.assertEqual(config.gtraj["hierarchyBlocks"], 2)
        self.assertGreater(config.gtraj["hierarchyLossWeight"], 0.0)


if __name__ == "__main__":
    unittest.main()
