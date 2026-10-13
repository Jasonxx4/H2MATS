import unittest

import torch
import torch.nn as nn

from hybrid_geo_network import HybridGeoTrajSimEncoder


class StubHybridEncoder(HybridGeoTrajSimEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.graph_calls = 0
        self.combined_shapes = None

    def encode_graph(self, hierarchy_data):
        self.graph_calls += 1
        return torch.ones((32, 1))

    def encode_trajectories(
        self,
        graph_node_embeddings,
        hierarchy_data,
        traj_seqs,
        coor_seqs,
        seq_lengths,
    ):
        self.combined_shapes = (traj_seqs.shape, coor_seqs.shape, seq_lengths.shape)
        return torch.stack(
            [traj_seqs.float().sum(dim=1), coor_seqs.float().sum(dim=(1, 2))],
            dim=1,
        )


class HybridTripletForwardTest(unittest.TestCase):
    def test_triplet_uses_one_graph_pass_and_restores_batches(self):
        net = StubHybridEncoder()
        anchor = (
            torch.tensor([[1, 2, 0], [3, 4, 5]]),
            torch.ones((2, 3, 2)),
            torch.tensor([2, 3]),
        )
        positive = (
            torch.tensor([[6, 7]]),
            torch.ones((1, 2, 2)),
            torch.tensor([2]),
        )
        negative = (
            torch.tensor([[8, 9, 10, 11], [12, 13, 0, 0]]),
            torch.ones((2, 4, 2)),
            torch.tensor([4, 2]),
        )

        outputs = net.forward_triplet({}, anchor, positive, negative)

        self.assertEqual(net.graph_calls, 1)
        self.assertTrue(net.last_triplet_combined)
        self.assertEqual(net.combined_shapes, (torch.Size([5, 4]), torch.Size([5, 4, 2]), torch.Size([5])))
        self.assertEqual([tuple(item.shape) for item in outputs], [(2, 2), (1, 2), (2, 2)])

    def test_uneven_lengths_keep_one_graph_pass_without_excessive_padding(self):
        net = StubHybridEncoder()
        short = (
            torch.ones((2, 2), dtype=torch.long),
            torch.ones((2, 2, 2)),
            torch.tensor([2, 2]),
        )
        long = (
            torch.ones((2, 20), dtype=torch.long),
            torch.ones((2, 20, 2)),
            torch.tensor([20, 20]),
        )

        outputs = net.forward_triplet({}, long, short, short)

        self.assertEqual(net.graph_calls, 1)
        self.assertFalse(net.last_triplet_combined)
        self.assertEqual([tuple(item.shape) for item in outputs], [(2, 2), (2, 2), (2, 2)])

    def test_coordinate_transformer_uses_separate_sequence_passes(self):
        net = StubHybridEncoder()
        net.coord_encoder = "transformer"
        batch = (
            torch.ones((2, 6), dtype=torch.long),
            torch.ones((2, 6, 2)),
            torch.tensor([6, 5]),
        )

        outputs = net.forward_triplet({}, batch, batch, batch)

        self.assertEqual(net.graph_calls, 1)
        self.assertFalse(net.last_triplet_combined)
        self.assertEqual([tuple(item.shape) for item in outputs], [(2, 2)] * 3)


if __name__ == "__main__":
    unittest.main()
