import unittest

import torch

from sequence_encoders import (
    GatedTemporalConvEncoder,
    HierarchyAlignedMultiScaleEncoder,
    HierarchySegmentPool,
    TransformerSeqEncoder,
)


def reference_segment_pool(pool, x, region_ids, padding_mask):
    pooled_sequences = []
    segment_lengths = []
    for batch_idx in range(x.size(0)):
        valid = ~padding_mask[batch_idx]
        valid_x = x[batch_idx, valid]
        valid_regions = region_ids[batch_idx, valid]
        if valid_x.size(0) == 0:
            pooled_sequences.append(x.new_zeros((1, x.size(-1))))
            segment_lengths.append(1)
            continue

        starts = torch.ones(valid_regions.size(0), dtype=torch.bool, device=x.device)
        starts[1:] = valid_regions[1:] != valid_regions[:-1]
        segment_ids = starts.cumsum(dim=0) - 1
        start_indices = starts.nonzero(as_tuple=False).flatten()
        end_indices = torch.cat([
            start_indices[1:] - 1,
            start_indices.new_tensor([valid_x.size(0) - 1]),
        ])
        num_segments = int(start_indices.numel())
        sums = x.new_zeros((num_segments, x.size(-1)))
        sums.index_add_(0, segment_ids, valid_x)
        counts = x.new_zeros((num_segments, 1))
        counts.index_add_(0, segment_ids, x.new_ones((valid_x.size(0), 1)))
        means = sums / counts.clamp_min(1.0)
        boundary = pool.start_proj(valid_x[start_indices]) + pool.end_proj(valid_x[end_indices])
        pooled_sequences.append(pool.norm(means + boundary))
        segment_lengths.append(num_segments)

    max_segments = max(item.size(0) for item in pooled_sequences)
    padded = x.new_zeros((x.size(0), max_segments, x.size(-1)))
    for batch_idx, sequence in enumerate(pooled_sequences):
        padded[batch_idx, : sequence.size(0)] = sequence
    lengths = torch.as_tensor(segment_lengths, device=x.device, dtype=torch.long)
    positions = torch.arange(max_segments, device=x.device).unsqueeze(0)
    return padded, positions >= lengths.unsqueeze(1), lengths


class HierarchySegmentPoolTest(unittest.TestCase):
    def test_vectorized_pool_matches_reference_output_and_gradient(self):
        torch.manual_seed(11)
        pool = HierarchySegmentPool(5)
        region_ids = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [2, 2, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        padding_mask = torch.tensor([
            [False, False, False, False, False, False],
            [False, False, False, False, True, True],
            [True, True, True, True, True, True],
        ])
        actual_x = torch.randn(3, 6, 5, requires_grad=True)
        reference_x = actual_x.detach().clone().requires_grad_(True)

        actual, actual_mask, actual_lengths = pool(actual_x, region_ids, padding_mask)
        expected, expected_mask, expected_lengths = reference_segment_pool(
            pool,
            reference_x,
            region_ids,
            padding_mask,
        )

        self.assertTrue(torch.equal(actual_mask, expected_mask))
        self.assertTrue(torch.equal(actual_lengths, expected_lengths))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        actual_grad = torch.autograd.grad(actual.square().sum(), actual_x)[0]
        expected_grad = torch.autograd.grad(expected.square().sum(), reference_x)[0]
        self.assertTrue(torch.allclose(actual_grad, expected_grad, atol=1e-6))

    def test_pool_supports_mixed_precision_projection_outputs(self):
        torch.manual_seed(13)
        pool = HierarchySegmentPool(8)
        x = torch.randn(2, 6, 8, requires_grad=True)
        region_ids = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 1, 2, 2, 2],
        ])
        padding_mask = torch.tensor([
            [False, False, False, False, False, True],
            [False, False, False, False, True, True],
        ])

        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
            output, mask, lengths = pool(x, region_ids, padding_mask)
            loss = output.float().square().mean()
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 3, 8))
        self.assertEqual(lengths.tolist(), [3, 3])
        self.assertEqual(mask.tolist(), [[False, False, False], [False, False, False]])
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class HierarchyAlignedMultiScaleEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.hierarchy = {
            "node_to_l1": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
            "node_to_l2": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            "node_to_l3": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1]),
        }
        self.encoder = HierarchyAlignedMultiScaleEncoder(
            d_model=8,
            out_dim=12,
            num_layers=1,
            backbone="conv",
            kernel_size=3,
            dropout=0.0,
        )
        self.encoder.eval()

    def test_hierarchy_boundaries_define_segment_counts(self):
        x = torch.randn(2, 6, 8)
        node_ids = torch.tensor([
            [0, 1, 2, 3, 4, 5],
            [2, 3, 4, 5, 0, 0],
        ])
        padding_mask = torch.tensor([
            [False, False, False, False, False, False],
            [False, False, False, False, True, True],
        ])

        output, details = self.encoder(
            x,
            node_ids,
            self.hierarchy,
            padding_mask=padding_mask,
            return_details=True,
        )

        self.assertEqual(tuple(output.shape), (2, 12))
        self.assertEqual(tuple(details["scale_embeddings"].shape), (2, 4, 8))
        self.assertTrue(torch.allclose(details["scale_weights"].sum(dim=1), torch.ones(2)))
        self.assertEqual(details["segment_lengths"].tolist(), [[6, 3, 2, 1], [4, 2, 2, 1]])

    def test_padding_values_do_not_change_output(self):
        x = torch.randn(1, 6, 8)
        node_ids = torch.tensor([[0, 1, 2, 3, 0, 0]])
        padding_mask = torch.tensor([[False, False, False, False, True, True]])
        changed_padding = x.clone()
        changed_padding[:, 4:] = 1000.0

        output_a = self.encoder(x, node_ids, self.hierarchy, padding_mask=padding_mask)
        output_b = self.encoder(changed_padding, node_ids, self.hierarchy, padding_mask=padding_mask)

        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-6))

    def test_gradient_reaches_valid_node_tokens(self):
        x = torch.randn(1, 5, 8, requires_grad=True)
        node_ids = torch.tensor([[0, 1, 2, 3, 4]])
        padding_mask = torch.zeros((1, 5), dtype=torch.bool)

        output = self.encoder(x, node_ids, self.hierarchy, padding_mask=padding_mask)
        output.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_scale_aligned_inputs_all_contribute_to_output(self):
        node_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        padding_mask = torch.zeros((1, 6), dtype=torch.bool)
        scale_inputs = [
            torch.randn(1, 6, 8, requires_grad=True)
            for _ in range(4)
        ]

        output = self.encoder(
            scale_inputs[0],
            node_ids,
            self.hierarchy,
            padding_mask=padding_mask,
            scale_inputs=scale_inputs,
        )
        output.square().mean().backward()

        for scale_input in scale_inputs:
            self.assertIsNotNone(scale_input.grad)
            self.assertGreater(float(scale_input.grad.abs().sum()), 0.0)

    def test_hybrid_backbone_assigns_transformers_only_to_coarse_scales(self):
        encoder = HierarchyAlignedMultiScaleEncoder(
            d_model=8,
            out_dim=12,
            num_layers=1,
            backbone="hybrid",
            kernel_size=3,
            dropout=0.0,
            transformer_heads=2,
        )
        encoder.eval()
        self.assertEqual(
            encoder.scale_backbones,
            ("conv", "conv", "transformer", "transformer"),
        )
        self.assertTrue(all(
            isinstance(item, GatedTemporalConvEncoder)
            for item in encoder.scale_encoders[:2]
        ))
        self.assertTrue(all(
            isinstance(item, TransformerSeqEncoder)
            for item in encoder.scale_encoders[2:]
        ))

        x = torch.randn(2, 6, 8, requires_grad=True)
        node_ids = torch.tensor([
            [0, 1, 2, 3, 4, 5],
            [2, 3, 4, 5, 0, 0],
        ])
        padding_mask = torch.tensor([
            [False, False, False, False, False, False],
            [False, False, False, False, True, True],
        ])
        output = encoder(x, node_ids, self.hierarchy, padding_mask=padding_mask)
        self.assertEqual(tuple(output.shape), (2, 12))
        output.square().mean().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_transformer_scale_encoder_ignores_padding_values(self):
        encoder = TransformerSeqEncoder(
            d_model=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
        encoder.eval()
        x = torch.randn(1, 5, 8)
        padding_mask = torch.tensor([[False, False, False, True, True]])
        changed = x.clone()
        changed[:, 3:] = 1000.0

        output_a = encoder(x, padding_mask)
        output_b = encoder(changed, padding_mask)

        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-6))
        self.assertTrue(torch.equal(output_a[:, 3:], torch.zeros_like(output_a[:, 3:])))


if __name__ == "__main__":
    unittest.main()
