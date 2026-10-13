"""Re-encode an existing trajectory NPZ without regenerating its queries.

This is intentionally separate from ``ann_pipeline perturb-build``.  The latter
samples new perturbations; this utility keeps the input row order and all
row-aligned metadata, and only replaces the continuous embeddings.
"""

import argparse
import os
import os.path as osp
from pathlib import Path

import numpy as np

from ann_pipeline import (
    apply_dataset,
    apply_model_overrides,
    new_default_config,
    refresh_distance_paths,
    resolve_checkpoint,
)
from trajectory_embedding import export_trajectory_embeddings


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode an existing Query/Gallery NPZ with a new checkpoint while "
            "preserving trajectories, source IDs, perturbation metadata, and row order."
        )
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default=None, choices=["beijing", "porto", "tdrive"])
    parser.add_argument(
        "--distance-type",
        default=None,
        choices=["TP", "DITA", "discret_frechet", "LCRS", "NetERP"],
    )
    parser.add_argument("--coordinate-file", default=None)
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Omit coordinates and HMT scale outputs from the re-encoded NPZ.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cuda", default=None)
    parser.add_argument("--seq-encoder", choices=["transformer", "mamba", "bimamba", "hmt"], default=None)
    parser.add_argument("--graph-encoder", choices=["hybrid_multilevel", "simple_gcn"], default=None)
    parser.add_argument("--coord-encoder", choices=["lstm", "mamba", "transformer"], default=None)
    parser.add_argument("--mamba-layers", type=int, default=None)
    parser.add_argument("--mamba-d-state", type=int, default=None)
    parser.add_argument("--mamba-d-conv", type=int, default=None)
    parser.add_argument("--mamba-expand", type=int, default=None)
    parser.add_argument("--hmt-backbone", choices=["conv", "mamba", "bimamba", "transformer", "hybrid"], default=None)
    parser.add_argument("--hmt-layers", type=int, default=None)
    parser.add_argument("--hmt-kernel-size", type=int, default=None)
    parser.add_argument("--hmt-dropout", type=float, default=None)
    parser.add_argument("--hmt-transformer-heads", type=int, default=None)
    parser.add_argument("--hmt-scale-aligned-graph", dest="hmt_scale_aligned_graph", action="store_true", default=None)
    parser.add_argument("--no-hmt-scale-aligned-graph", dest="hmt_scale_aligned_graph", action="store_false")
    parser.add_argument("--coord-fusion", choices=["gated_attention", "late_concat"], default=None)
    parser.add_argument("--use-hybrid-geo", dest="use_hybrid_geo", action="store_true", default=None)
    parser.add_argument("--no-use-hybrid-geo", dest="use_hybrid_geo", action="store_false")
    parser.add_argument(
        "--include-scale-outputs",
        action="store_true",
        help="Also regenerate HMT scale embeddings/weights when the model uses HMT.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    with np.load(args.input_file, allow_pickle=True) as data:
        if "trajectories" not in data.files:
            raise ValueError("Input NPZ must contain a trajectories array.")
        trajectories = np.asarray(data["trajectories"], dtype=object)
        if "coordinates" in data.files:
            coordinates = np.asarray(data["coordinates"], dtype=object)
        elif args.coordinate_file:
            coordinates = np.asarray(np.load(args.coordinate_file, allow_pickle=True), dtype=object)
        else:
            raise ValueError(
                "Input NPZ has no coordinates. Re-encoding requires the exact "
                "per-row coordinate sequences; provide --coordinate-file or use "
                "the original non-compact perturbation file."
            )
        if len(trajectories) != len(coordinates):
            raise ValueError(
                f"Trajectory rows ({len(trajectories)}) and coordinate rows "
                f"({len(coordinates)}) differ."
            )
        original = {key: np.asarray(data[key]) for key in data.files}

    config = apply_dataset(new_default_config(), args.dataset)
    if args.distance_type is not None:
        config.distance_type = args.distance_type
        refresh_distance_paths(config)
    config = apply_model_overrides(config, args)
    coordinate_stats = original.get("coordinate_stats")
    encoded = export_trajectory_embeddings(
        config=config,
        load_model=resolve_checkpoint(args.model),
        trajectories=trajectories,
        coordinates=coordinates,
        output_file=None,
        batch_size=args.batch_size,
        coordinate_stats=coordinate_stats,
        include_coordinates=(
            not args.compact_output
            and ("coordinates" in original or args.coordinate_file is not None)
        ),
        include_scale_outputs=args.include_scale_outputs and not args.compact_output,
    )

    output = dict(original)
    output["embeddings"] = encoded["embeddings"]
    if "coordinate_stats" not in output:
        output["coordinate_stats"] = encoded["coordinate_stats"]
    if args.compact_output:
        for key in ("coordinates", "scale_embeddings", "scale_weights", "segment_lengths"):
            output.pop(key, None)
    elif "coordinates" not in output and args.coordinate_file is not None:
        output["coordinates"] = coordinates
    if args.include_scale_outputs and not args.compact_output:
        for key in ("scale_embeddings", "scale_weights", "segment_lengths"):
            if key in encoded:
                output[key] = encoded[key]
    out_dir = osp.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.output_file, **output)
    print(
        f"Re-encoded {len(trajectories)} rows: {Path(args.input_file).resolve()} -> "
        f"{Path(args.output_file).resolve()}"
    )
    print(f"embedding_shape={output['embeddings'].shape}; source/metadata arrays preserved")


if __name__ == "__main__":
    main()
