"""Build TP supervision on validation candidates produced by Global+topology."""

import argparse
import json
import os
import os.path as osp

import numpy as np

from ann_index import load_hierarchy_arrays
from hash_global import HashGlobalConfig, HashGlobalIndex, load_hash_head
from hier_hash_large_scale import load_embedding_payload
from topo_route_hash_large_scale import AdaptiveTopoHashIndex, TopoRouteHashConfig
from trajectory_metric_backend import TrajectoryMetricBackend


def build_teacher(args):
    if osp.exists(args.output_file) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_file}; pass --overwrite to replace it."
        )
    payload = load_embedding_payload(args.embedding_file, role="query")
    embeddings = payload["embeddings"]
    trajectories = payload["trajectories"]
    source_ids = payload["source_ids"]
    head, _ = load_hash_head(args.candidate_hash_model_file)
    hash_config = HashGlobalConfig(
        bits=head.config.bits,
        probe_radius=args.probe_radius,
        hamming_candidates=args.hamming_candidates,
        metric=args.embedding_metric,
        normalize=args.normalize,
        embedding_storage="float16",
        encode_batch_size=args.encode_batch_size,
    )
    hash_index = HashGlobalIndex(hash_config).build(
        embeddings, head, encode_device=args.hash_device
    )
    topo_config = TopoRouteHashConfig(
        candidate_source="hybrid",
        qgram=args.qgram,
        minhash_size=args.minhash_size,
        postings_per_level=args.postings_per_level,
        l3_expand_hops=args.l3_expand_hops,
        min_route_candidates=args.min_route_candidates,
        max_route_candidates=args.max_route_candidates,
        structural_candidates=args.structural_candidates,
        global_candidates=args.global_candidates,
        max_rerank_candidates=args.max_rerank_candidates,
        embedding_weight=1.0,
        adaptive_initial_global=min(512, args.global_candidates),
        adaptive_initial_structural=min(192, args.structural_candidates),
        adaptive_max_global=args.global_candidates,
        adaptive_max_structural=args.structural_candidates,
        adaptive_max_rerank=args.max_rerank_candidates,
        seed=args.seed,
    )
    index = AdaptiveTopoHashIndex(
        hash_index, load_hierarchy_arrays(args.hierarchy_dir), topo_config
    ).build(trajectories, scale_embeddings=payload.get("scale_embeddings"))
    backend = TrajectoryMetricBackend(
        "TP",
        point_distance_file=args.point_distance_file,
        num_threads=args.num_threads,
    )
    query_ids = np.arange(len(embeddings), dtype=np.int64)
    offsets = [0]
    candidate_chunks = []
    distance_chunks = []
    try:
        try:
            from tqdm import tqdm

            iterator = tqdm(query_ids, desc="validation union TP teacher")
        except ImportError:
            iterator = query_ids
        for query_id in iterator:
            exclude_ids = np.flatnonzero(
                source_ids == source_ids[int(query_id)]
            ).astype(np.int64, copy=False)
            pools = index.retrieve_candidate_sets(
                trajectories[int(query_id)],
                hash_index.codes[int(query_id)],
                top_k=50,
                exclude_ids=exclude_ids,
                global_candidates=args.global_candidates,
                structural_candidates=args.structural_candidates,
            )
            candidates = np.asarray(
                pools["union_candidate_ids"], dtype=np.int64
            )
            if len(candidates) > args.max_rerank_candidates:
                candidates = candidates[: args.max_rerank_candidates]
            distances = backend.distances(
                trajectories[int(query_id)], trajectories[candidates]
            ).astype(np.float32, copy=False)
            candidate_chunks.append(candidates.astype(np.int32, copy=False))
            distance_chunks.append(distances)
            offsets.append(offsets[-1] + len(candidates))
    finally:
        backend.close()

    candidates = np.concatenate(candidate_chunks)
    distances = np.concatenate(distance_chunks)
    out_dir = osp.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    config = {
        key: getattr(args, key)
        for key in (
            "qgram", "minhash_size", "postings_per_level", "l3_expand_hops",
            "min_route_candidates", "max_route_candidates",
            "structural_candidates", "global_candidates",
            "max_rerank_candidates", "probe_radius", "hamming_candidates",
        )
    }
    np.savez_compressed(
        args.output_file,
        query_ids=query_ids,
        offsets=np.asarray(offsets, dtype=np.int64),
        candidate_ids=candidates,
        tp_distances=distances,
        embedding_count=np.asarray(len(embeddings), dtype=np.int64),
        source_embedding_file=np.asarray(osp.abspath(args.embedding_file)),
        candidate_hash_model_file=np.asarray(
            osp.abspath(args.candidate_hash_model_file)
        ),
        candidate_config=np.asarray(json.dumps(config, sort_keys=True)),
    )
    counts = np.diff(np.asarray(offsets))
    print(
        f"saved candidate TP teacher: {args.output_file}, queries={len(query_ids)}, "
        f"candidates={len(candidates)}, mean={counts.mean():.2f}, "
        f"min={counts.min()}, max={counts.max()}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-file", required=True)
    parser.add_argument("--candidate-hash-model-file", required=True)
    parser.add_argument("--hierarchy-dir", required=True)
    parser.add_argument("--point-distance-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--hash-device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--probe-radius", type=int, default=1)
    parser.add_argument("--hamming-candidates", type=int, default=768)
    parser.add_argument("--embedding-metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--qgram", type=int, default=2)
    parser.add_argument("--minhash-size", type=int, default=8)
    parser.add_argument("--postings-per-level", type=int, default=4)
    parser.add_argument("--l3-expand-hops", type=int, default=1)
    parser.add_argument("--min-route-candidates", type=int, default=96)
    parser.add_argument("--max-route-candidates", type=int, default=192)
    parser.add_argument("--structural-candidates", type=int, default=192)
    parser.add_argument("--global-candidates", type=int, default=768)
    parser.add_argument("--max-rerank-candidates", type=int, default=960)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_teacher(parse_args())
