import unittest

import torch
import torch.nn as nn

from hybrid_geo_network import HybridGeoTrajSimEncoder


def build_encoder():
    return HybridGeoTrajSimEncoder(
        feature_size=3,
        embedding_size=4,
        hidden_size=8,
        num_layers=1,
        dropout_rate=0.0,
        concat=False,
        device=torch.device("cpu"),
        useLSTM=False,
        curvature=1.0,
        seq_encoder="hmt",
        coord_encoder="lstm",
        hmt_backbone="conv",
        hmt_layers=1,
        hmt_kernel_size=3,
        hmt_dropout=0.0,
        hmt_transformer_heads=2,
        hmt_scale_aligned_graph=True,
        coord_fusion="late_concat",
    )


class CachedGraphStub(HybridGeoTrajSimEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.graph_calls = 0
        self.graph_frozen = True
        self.cache_frozen_graph = True
        self._cached_graph_node_embeddings = None

    def _graph_modules(self):
        return []

    def encode_graph(self, hierarchy_data):
        self.graph_calls += 1
        return torch.ones((16, 2))

    def encode_trajectories(
        self,
        graph_node_embeddings,
        hierarchy_data,
        traj_seqs,
        coor_seqs,
        seq_lengths,
    ):
        return graph_node_embeddings[traj_seqs[:, 0]]


class PretrainedFrozenGraphTest(unittest.TestCase):
    def test_graph_only_checkpoint_loads_and_freezes_graph_modules(self):
        torch.manual_seed(41)
        source = build_encoder()
        checkpoint = {"graph_state_dict": source.graph_state_dict()}
        target = build_encoder()

        with torch.no_grad():
            for parameter in target.euclidean_gnn.parameters():
                parameter.zero_()

        missing, unexpected = target.load_pretrained_graph(checkpoint, strict=True)
        target.freeze_graph_encoder(cache=True)
        target.train()

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertTrue(all(
            torch.equal(source.state_dict()[key], target.state_dict()[key])
            for key in source.graph_state_dict()
        ))
        self.assertTrue(all(
            not parameter.requires_grad
            for module in target._graph_modules()
            for parameter in module.parameters()
        ))
        self.assertTrue(all(not module.training for module in target._graph_modules()))
        self.assertTrue(any(parameter.requires_grad for parameter in target.seq_encoder.parameters()))

    def test_graph_encoder_can_be_unfrozen_after_cached_warmup(self):
        net = build_encoder()
        net.freeze_graph_encoder(cache=True)
        net._cached_graph_node_embeddings = torch.ones((4, 4))

        net.train()
        net.unfreeze_graph_encoder()

        graph_parameters = list(net.graph_parameters())
        graph_parameter_ids = {id(parameter) for parameter in graph_parameters}
        downstream_parameters = [
            parameter
            for parameter in net.parameters()
            if id(parameter) not in graph_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": downstream_parameters, "lr": 1e-3},
                {"params": graph_parameters, "lr": 1e-4},
            ]
        )

        self.assertFalse(net.graph_frozen)
        self.assertFalse(net.cache_frozen_graph)
        self.assertIsNone(net._cached_graph_node_embeddings)
        self.assertTrue(all(parameter.requires_grad for parameter in graph_parameters))
        self.assertTrue(all(module.training for module in net._graph_modules()))
        self.assertAlmostEqual(
            optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"],
            0.1,
        )

    def test_cached_graph_is_reused_across_triplet_calls(self):
        net = CachedGraphStub()
        batch = (
            torch.tensor([[1, 2], [3, 4]]),
            torch.ones((2, 2, 2)),
            torch.tensor([2, 2]),
        )

        net.forward_triplet({}, batch, batch, batch)
        net.forward_triplet({}, batch, batch, batch)

        self.assertEqual(net.graph_calls, 1)
        self.assertIsNotNone(net._cached_graph_node_embeddings)

    def test_default_mode_does_not_cache_graph(self):
        net = CachedGraphStub()
        net.graph_frozen = False
        batch = (
            torch.tensor([[1, 2]]),
            torch.ones((1, 2, 2)),
            torch.tensor([2]),
        )

        net.forward_triplet({}, batch, batch, batch)
        net.forward_triplet({}, batch, batch, batch)

        self.assertEqual(net.graph_calls, 2)


if __name__ == "__main__":
    unittest.main()
