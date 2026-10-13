import argparse
import heapq
import math
import os
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from setting import SetParameter


LEVELS = ("l3", "l2", "l1")


def min_max_norm(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if not finite.any():
        return out
    lo, hi = values[finite].min(), values[finite].max()
    if hi - lo < 1e-12:
        return out
    out[finite] = (values[finite] - lo) / (hi - lo)
    return out


def load_trajectory_node_lists(config, max_trajectories=None):
    shuffle_file = str(config.shuffle_node_file)
    if os.path.exists(shuffle_file):
        trajs = np.load(shuffle_file, allow_pickle=True)
        return trajs[:max_trajectories] if max_trajectories else trajs

    traj_file = str(config.traj_file)
    if not os.path.exists(traj_file):
        return []

    raw = torch.load(traj_file, map_location="cpu")
    trajs = []
    for item in raw[:max_trajectories] if max_trajectories else raw:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            trajs.append(item[0])
        else:
            trajs.append(item)
    return trajs


def build_transition_statistics(trajs):
    edge_counts = Counter()
    out_counts = Counter()
    node_flow = Counter()
    for traj in tqdm(trajs, desc="count trajectory transitions"):
        if len(traj) == 0:
            continue
        for node in traj:
            node_flow[int(node)] += 1
        for u, v in zip(traj[:-1], traj[1:]):
            u, v = int(u), int(v)
            if u == v:
                continue
            edge_counts[(u, v)] += 1
            out_counts[u] += 1
    return edge_counts, out_counts, node_flow


def build_transition_edge_features(edge_df, assignment, edge_counts, out_counts,
                                   transition_smooth=1.0):
    """Build [log forward count, log reverse count, P(v|u), P(u|v)] per graph edge."""
    assignment = np.asarray(assignment, dtype=np.int64)
    aggregate = defaultdict(lambda: [0.0, 0.0])
    region_out = Counter()
    region_targets = defaultdict(set)
    for node, count in out_counts.items():
        region_out[int(assignment[int(node)])] += float(count)

    for row in edge_df.itertuples(index=False):
        u, v = int(row.s_node), int(row.e_node)
        cu, cv = int(assignment[u]), int(assignment[v])
        if cu < 0 or cv < 0 or cu == cv:
            continue
        forward = float(edge_counts.get((u, v), 0))
        reverse = float(edge_counts.get((v, u), 0))
        aggregate[(cu, cv)][0] += forward
        aggregate[(cu, cv)][1] += reverse
        region_targets[cu].add(cv)

    max_count = max(
        (max(values) for values in aggregate.values()),
        default=1.0,
    )
    log_scale = max(math.log1p(max_count), 1.0)

    features = []
    for row in edge_df.itertuples(index=False):
        u, v = int(row.s_node), int(row.e_node)
        cu, cv = int(assignment[u]), int(assignment[v])
        forward, reverse = aggregate.get((cu, cv), (0.0, 0.0))
        forward_degree = max(len(region_targets.get(cu, ())), 1)
        reverse_degree = max(len(region_targets.get(cv, ())), 1)
        forward_probability = (forward + transition_smooth) / (
            region_out[cu] + transition_smooth * forward_degree
        )
        reverse_probability = (reverse + transition_smooth) / (
            region_out[cv] + transition_smooth * reverse_degree
        )
        features.append([
            math.log1p(forward) / log_scale,
            math.log1p(reverse) / log_scale,
            forward_probability,
            reverse_probability,
        ])
    return np.asarray(features, dtype=np.float32)


def read_road_tables(config):
    node_df = pd.read_csv(str(config.node_file))
    edge_df = pd.read_csv(str(config.edge_file))
    max_node = int(
        max(
            node_df["node"].max(),
            edge_df["s_node"].max(),
            edge_df["e_node"].max(),
        )
    )
    return node_df, edge_df, max_node + 1


def build_weighted_graph(edge_df, edge_counts, out_counts, args):
    graph = nx.Graph()
    edge_cost = {}
    lengths = edge_df["length"].to_numpy(dtype=np.float64)
    length_norm = min_max_norm(lengths)

    for idx, row in tqdm(edge_df.iterrows(), total=len(edge_df), desc="build affinity graph"):
        u, v = int(row["s_node"]), int(row["e_node"])
        directed_count = edge_counts.get((u, v), 0)
        reverse_count = edge_counts.get((v, u), 0)
        out_uv = out_counts.get(u, 0) + out_counts.get(v, 0)
        deg_smooth = 2.0 * args.transition_smooth
        p_trans = (directed_count + reverse_count + args.transition_smooth) / max(out_uv + deg_smooth, 1e-12)
        p_trans = min(max(p_trans, 1e-12), 1.0)

        cost = (
            args.lambda_geo * float(length_norm[idx])
            + args.lambda_time * float(length_norm[idx])
            + args.lambda_trans * float(-math.log(p_trans))
        )
        affinity = math.exp(-cost)

        if graph.has_edge(u, v):
            if affinity > graph[u][v]["weight"]:
                graph[u][v]["weight"] = affinity
                graph[u][v]["cost"] = cost
        else:
            graph.add_edge(u, v, weight=affinity, cost=cost, length=float(row["length"]))
        edge_cost[(u, v)] = cost
        edge_cost[(v, u)] = cost
    return graph, edge_cost


def compute_importance(graph, num_nodes, node_flow, args, cache_dir=None):
    centrality_cache_dir = None
    if cache_dir is not None and not args.no_centrality_cache:
        centrality_cache_dir = os.path.join(cache_dir, "centrality_cache")
        os.makedirs(centrality_cache_dir, exist_ok=True)

    degree_arr = np.zeros(num_nodes, dtype=np.float64)
    for node, deg in graph.degree():
        degree_arr[int(node)] = deg

    flow_arr = np.zeros(num_nodes, dtype=np.float64)
    for node, cnt in node_flow.items():
        if 0 <= node < num_nodes:
            flow_arr[node] = cnt

    pagerank_cache = None
    if centrality_cache_dir is not None:
        pagerank_cache = os.path.join(centrality_cache_dir, f"pagerank_iter{args.pagerank_iter}.npy")
    if pagerank_cache is not None and os.path.exists(pagerank_cache):
        print(f"load cached PageRank: {pagerank_cache}")
        pagerank_arr = np.load(pagerank_cache)
    else:
        print("compute PageRank")
        pagerank = nx.pagerank(graph, weight="weight", max_iter=args.pagerank_iter, tol=1e-6)
        pagerank_arr = np.zeros(num_nodes, dtype=np.float64)
        for node, val in pagerank.items():
            pagerank_arr[int(node)] = val
        if pagerank_cache is not None:
            np.save(pagerank_cache, pagerank_arr)

    if args.betweenness_k > 0:
        betweenness_cache = None
        if centrality_cache_dir is not None:
            betweenness_cache = os.path.join(
                centrality_cache_dir,
                f"betweenness_k{args.betweenness_k}_seed{args.seed}.npy",
            )
        if betweenness_cache is not None and os.path.exists(betweenness_cache):
            print(f"load cached betweenness: {betweenness_cache}")
            betweenness_arr = np.load(betweenness_cache)
        else:
            print(f"compute approximate betweenness: k={args.betweenness_k}")
            betweenness = nx.betweenness_centrality(
                graph,
                k=args.betweenness_k,
                weight="cost",
                seed=args.seed,
                normalized=True,
            )
            betweenness_arr = np.zeros(num_nodes, dtype=np.float64)
            for node, val in betweenness.items():
                betweenness_arr[int(node)] = val
            if betweenness_cache is not None:
                np.save(betweenness_cache, betweenness_arr)
    else:
        betweenness_arr = np.zeros(num_nodes, dtype=np.float64)

    importance = (
        args.alpha_degree * min_max_norm(degree_arr)
        + args.alpha_betweenness * min_max_norm(betweenness_arr)
        + args.alpha_flow * min_max_norm(flow_arr)
        + args.alpha_pagerank * min_max_norm(pagerank_arr)
    )
    return importance


def louvain_communities(subgraph, resolution, seed):
    if subgraph.number_of_nodes() == 0:
        return []
    if subgraph.number_of_edges() == 0:
        return [{node} for node in subgraph.nodes()]
    return nx.algorithms.community.louvain_communities(
        subgraph,
        weight="weight",
        resolution=resolution,
        seed=seed,
    )


def select_seeds(nodes, importance, seed_ratio, min_seeds):
    nodes = list(nodes)
    if not nodes:
        return []
    seed_count = max(min_seeds, int(math.ceil(len(nodes) * seed_ratio)))
    seed_count = min(seed_count, len(nodes))
    nodes.sort(key=lambda x: importance[int(x)], reverse=True)
    return nodes[:seed_count]


def select_top_k_seeds(nodes, importance, k):
    nodes = list(nodes)
    if not nodes:
        return []
    k = min(max(1, k), len(nodes))
    nodes.sort(key=lambda x: importance[int(x)], reverse=True)
    return nodes[:k]


def multi_source_assign(subgraph, seeds):
    if len(seeds) <= 1:
        seed = seeds[0] if seeds else next(iter(subgraph.nodes()))
        return {node: seed for node in subgraph.nodes()}
    heap = []
    dist = {}
    owner = {}
    for seed in seeds:
        dist[seed] = 0.0
        owner[seed] = seed
        heapq.heappush(heap, (0.0, int(seed), int(seed)))

    while heap:
        cur_dist, cur_owner, node = heapq.heappop(heap)
        if cur_dist > dist.get(node, float("inf")):
            continue
        for nbr, attrs in subgraph[node].items():
            edge_cost = float(attrs.get("cost", 1.0))
            next_dist = cur_dist + edge_cost
            old_dist = dist.get(nbr, float("inf"))
            old_owner = owner.get(nbr, None)
            if next_dist < old_dist or (abs(next_dist - old_dist) < 1e-12 and (old_owner is None or cur_owner < old_owner)):
                dist[nbr] = next_dist
                owner[nbr] = cur_owner
                heapq.heappush(heap, (next_dist, cur_owner, int(nbr)))

    fallback = min(seeds)
    for node in subgraph.nodes():
        owner.setdefault(node, fallback)
    return owner


def merge_small_regions(assign, graph, min_size):
    changed = True
    while changed:
        changed = False
        region_nodes = defaultdict(list)
        for node, cid in assign.items():
            region_nodes[cid].append(node)

        small = [cid for cid, nodes in region_nodes.items() if len(nodes) < min_size]
        if not small or len(region_nodes) <= 1:
            break

        for cid in small:
            nodes = region_nodes.get(cid, [])
            neighbor_score = defaultdict(float)
            for node in nodes:
                for nbr, attrs in graph[node].items():
                    nbr_cid = assign.get(nbr)
                    if nbr_cid is not None and nbr_cid != cid:
                        neighbor_score[nbr_cid] += attrs.get("weight", 1.0)
            if not neighbor_score:
                continue
            target = max(neighbor_score.items(), key=lambda x: x[1])[0]
            for node in nodes:
                assign[node] = target
            changed = True
    return assign


def partition_parent_region(graph, parent_nodes, importance, level_cfg, seed):
    parent_nodes = list(parent_nodes)
    if len(parent_nodes) <= level_cfg["min_size"] * 2:
        return {node: 0 for node in parent_nodes}

    subgraph = graph.subgraph(parent_nodes).copy()
    communities = louvain_communities(subgraph, level_cfg["resolution"], seed)
    seeds = set(select_seeds(parent_nodes, importance, level_cfg["seed_ratio"], level_cfg["min_seeds"]))

    assign = {}
    next_cid = 0
    for community in communities:
        community = set(community)
        community_seeds = sorted(community & seeds, key=lambda x: importance[int(x)], reverse=True)
        should_split = len(community_seeds) > 1 and len(community) >= level_cfg["split_min_size"]
        if should_split:
            owner = multi_source_assign(subgraph.subgraph(community).copy(), community_seeds)
            seed_to_cid = {}
            for node, owner_seed in owner.items():
                if owner_seed not in seed_to_cid:
                    seed_to_cid[owner_seed] = next_cid
                    next_cid += 1
                assign[node] = seed_to_cid[owner_seed]
        else:
            for node in community:
                assign[node] = next_cid
            next_cid += 1

    assign = merge_small_regions(assign, subgraph, level_cfg["min_size"])

    min_children = level_cfg.get("min_children", 1)
    enough_nodes = len(parent_nodes) >= level_cfg["min_size"] * min_children
    child_count = len(set(assign.values()))
    if min_children > 1 and enough_nodes and child_count < min_children:
        forced_seeds = select_top_k_seeds(parent_nodes, importance, min_children)
        owner = multi_source_assign(subgraph, forced_seeds)
        seed_to_cid = {seed_node: cid for cid, seed_node in enumerate(forced_seeds)}
        assign = {node: seed_to_cid[owner_seed] for node, owner_seed in owner.items()}
        assign = merge_small_regions(assign, subgraph, level_cfg["min_size"])

    return assign


def nested_partition(graph, importance, num_nodes, args):
    cfg = {
        "l3": {
            "resolution": args.l3_resolution,
            "seed_ratio": args.l3_seed_ratio,
            "min_size": args.l3_min_size,
            "split_min_size": args.l3_split_min_size,
            "min_seeds": 2,
            "min_children": args.l3_min_children,
        },
        "l2": {
            "resolution": args.l2_resolution,
            "seed_ratio": args.l2_seed_ratio,
            "min_size": args.l2_min_size,
            "split_min_size": args.l2_split_min_size,
            "min_seeds": 2,
            "min_children": args.l2_min_children,
        },
        "l1": {
            "resolution": args.l1_resolution,
            "seed_ratio": args.l1_seed_ratio,
            "min_size": args.l1_min_size,
            "split_min_size": args.l1_split_min_size,
            "min_seeds": 2,
            "min_children": args.l1_min_children,
        },
    }

    node_to_level = {}
    parent_regions = {0: list(graph.nodes())}
    for level in LEVELS:
        print(f"partition {level}")
        global_assign = np.full(num_nodes, -1, dtype=np.int64)
        child_regions = {}
        next_global_cid = 0
        for parent_id, parent_nodes in tqdm(parent_regions.items(), desc=f"{level} parent regions"):
            local_assign = partition_parent_region(
                graph,
                parent_nodes,
                importance,
                cfg[level],
                args.seed + int(parent_id),
            )
            local_to_global = {}
            for node, local_cid in local_assign.items():
                if local_cid not in local_to_global:
                    local_to_global[local_cid] = next_global_cid
                    child_regions[next_global_cid] = []
                    next_global_cid += 1
                global_cid = local_to_global[local_cid]
                global_assign[int(node)] = global_cid
                child_regions[global_cid].append(node)
        node_to_level[level] = global_assign
        parent_regions = child_regions
        print(f"{level}: {len(child_regions)} regions")
    return node_to_level


def build_super_edges(edge_df, assignment):
    edge_stats = {}
    for row in edge_df.itertuples(index=False):
        u, v = int(row.s_node), int(row.e_node)
        cu, cv = int(assignment[u]), int(assignment[v])
        if cu < 0 or cv < 0 or cu == cv:
            continue
        key = (cu, cv)
        if key not in edge_stats:
            edge_stats[key] = [0.0, 0]
        edge_stats[key][0] += float(row.length)
        edge_stats[key][1] += 1

    edges, weights = [], []
    for (cu, cv), (total_len, cnt) in edge_stats.items():
        edges.append([cu, cv])
        weights.append(total_len / max(cnt, 1))
    return np.asarray(edges, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def save_region_tables(out_dir, node_df, edge_df, node_to_level,
                       edge_counts=None, out_counts=None, transition_smooth=1.0):
    for level, assignment in node_to_level.items():
        np.save(os.path.join(out_dir, f"node_to_{level}.npy"), assignment)
        region_count = int(assignment.max()) + 1
        sizes = np.bincount(assignment[assignment >= 0], minlength=region_count)
        pd.DataFrame({"region": np.arange(region_count), "size": sizes}).to_csv(
            os.path.join(out_dir, f"{level}_regions.csv"),
            index=False,
        )
        super_edges, super_weights = build_super_edges(edge_df, assignment)
        np.save(os.path.join(out_dir, f"{level}_edge_index.npy"), super_edges)
        np.save(os.path.join(out_dir, f"{level}_edge_weight.npy"), super_weights)
        if edge_counts is not None and out_counts is not None:
            transition_features = build_transition_edge_features(
                edge_df,
                assignment,
                edge_counts,
                out_counts,
                transition_smooth=transition_smooth,
            )
            # Reorder from raw-road order to the unique super-edge order.
            feature_by_pair = {}
            for row, feature in zip(edge_df.itertuples(index=False), transition_features):
                u, v = int(row.s_node), int(row.e_node)
                cu, cv = int(assignment[u]), int(assignment[v])
                if cu >= 0 and cv >= 0 and cu != cv:
                    feature_by_pair.setdefault((cu, cv), feature)
            super_features = np.asarray([
                feature_by_pair.get(tuple(edge), np.zeros(4, dtype=np.float32))
                for edge in super_edges
            ], dtype=np.float32)
            np.save(os.path.join(out_dir, f"{level}_transition_features.npy"), super_features)

    export_df = node_df.copy()
    for level, assignment in node_to_level.items():
        export_df[level] = export_df["node"].map(lambda x: int(assignment[int(x)]))
    export_df.to_csv(os.path.join(out_dir, "node_hierarchy.csv"), index=False)


def plot_partitions(out_dir, node_df, node_to_level, args):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip hierarchy visualization")
        return

    rng = np.random.default_rng(args.seed)
    plot_df = node_df
    if args.plot_sample > 0 and len(node_df) > args.plot_sample:
        plot_df = node_df.iloc[rng.choice(len(node_df), size=args.plot_sample, replace=False)]

    for level, assignment in node_to_level.items():
        region_ids = plot_df["node"].map(lambda x: int(assignment[int(x)])).to_numpy()
        colors = (region_ids * 2654435761 % 9973).astype(np.float64)
        plt.figure(figsize=(9, 8))
        plt.scatter(
            plot_df["lng"],
            plot_df["lat"],
            c=colors,
            s=args.plot_point_size,
            cmap="tab20",
            linewidths=0,
            alpha=0.85,
        )
        plt.title(f"{args.dataset} {level.upper()} Louvain traffic hierarchy")
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{level}_partition.png"), dpi=220)
        plt.close()


def plot_road_partitions(out_dir, node_df, edge_df, node_to_level, args):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        print("matplotlib is not installed; skip road hierarchy visualization")
        return

    coord = {}
    for row in node_df.itertuples(index=False):
        coord[int(row.node)] = (float(row.lng), float(row.lat))

    plot_edges = edge_df
    if args.plot_max_edges > 0 and len(edge_df) > args.plot_max_edges:
        plot_edges = edge_df.sample(n=args.plot_max_edges, random_state=args.seed)

    for level, assignment in node_to_level.items():
        same_region_segments = []
        same_region_values = []
        boundary_segments = []

        for row in plot_edges.itertuples(index=False):
            u, v = int(row.s_node), int(row.e_node)
            if u not in coord or v not in coord:
                continue
            cu, cv = int(assignment[u]), int(assignment[v])
            if cu < 0 or cv < 0:
                continue
            segment = [coord[u], coord[v]]
            if cu == cv:
                same_region_segments.append(segment)
                same_region_values.append((cu * 2654435761) % 9973)
            else:
                boundary_segments.append(segment)

        fig, ax = plt.subplots(figsize=(10, 9))
        if same_region_segments:
            lc = LineCollection(
                same_region_segments,
                array=np.asarray(same_region_values, dtype=np.float64),
                cmap="tab20",
                linewidths=args.plot_road_width,
                alpha=0.90,
            )
            ax.add_collection(lc)

        if boundary_segments and args.plot_boundaries:
            boundary_lc = LineCollection(
                boundary_segments,
                colors="#111111",
                linewidths=args.plot_boundary_width,
                alpha=0.45,
            )
            ax.add_collection(boundary_lc)

        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{args.dataset} {level.upper()} road-network partition")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.margins(0.01)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{level}_road_partition.png"), dpi=240)
        fig.savefig(os.path.join(out_dir, f"{level}_road_partition.pdf"))
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate traffic-aware Louvain road hierarchy.")
    parser.add_argument("--dataset", default=None, choices=["beijing", "porto", "tdrive"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-trajectories", type=int, default=200000)
    parser.add_argument(
        "--transition-max-trajectories",
        type=int,
        default=None,
        help="Limit transition statistics to the training trajectory prefix.",
    )
    parser.add_argument("--seed", type=int, default=1953)

    parser.add_argument("--alpha-degree", type=float, default=0.25)
    parser.add_argument("--alpha-betweenness", type=float, default=0.25)
    parser.add_argument("--alpha-flow", type=float, default=0.30)
    parser.add_argument("--alpha-pagerank", type=float, default=0.20)
    parser.add_argument("--betweenness-k", type=int, default=512)
    parser.add_argument("--pagerank-iter", type=int, default=100)
    parser.add_argument("--no-centrality-cache", action="store_true")

    parser.add_argument("--lambda-geo", type=float, default=1.0)
    parser.add_argument("--lambda-time", type=float, default=0.0)
    parser.add_argument("--lambda-trans", type=float, default=0.35)
    parser.add_argument("--transition-smooth", type=float, default=1.0)

    parser.add_argument("--l3-resolution", type=float, default=0.35)
    parser.add_argument("--l2-resolution", type=float, default=0.80)
    parser.add_argument("--l1-resolution", type=float, default=1.40)
    parser.add_argument("--l3-seed-ratio", type=float, default=0.02)
    parser.add_argument("--l2-seed-ratio", type=float, default=0.08)
    parser.add_argument("--l1-seed-ratio", type=float, default=0.25)
    parser.add_argument("--l3-min-size", type=int, default=500)
    parser.add_argument("--l2-min-size", type=int, default=160)
    parser.add_argument("--l1-min-size", type=int, default=40)
    parser.add_argument("--l3-split-min-size", type=int, default=1200)
    parser.add_argument("--l2-split-min-size", type=int, default=400)
    parser.add_argument("--l1-split-min-size", type=int, default=100)
    parser.add_argument("--l3-min-children", type=int, default=1)
    parser.add_argument("--l2-min-children", type=int, default=2)
    parser.add_argument("--l1-min-children", type=int, default=2)

    parser.add_argument("--plot-sample", type=int, default=80000)
    parser.add_argument("--plot-point-size", type=float, default=0.8)
    parser.add_argument("--plot-max-edges", type=int, default=300000)
    parser.add_argument("--plot-road-width", type=float, default=0.18)
    parser.add_argument("--plot-boundary-width", type=float, default=0.35)
    parser.add_argument("--plot-boundaries", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SetParameter()
    if args.dataset is not None:
        config.dataset = args.dataset
        config.node_file = os.path.join("dataset", config.dataset, "node.csv")
        config.edge_file = os.path.join("dataset", config.dataset, "edge_weight.csv")
        config.traj_file = os.path.join("dataset", config.dataset, "matching_result.pt")
        config.data_file = os.path.join("data", config.dataset, "st_traj")
        config.shuffle_node_file = os.path.join(config.data_file, "shuffle_node_list.npy")

    args.dataset = config.dataset
    out_dir = args.output_dir or os.path.join("dataset", config.dataset, "hierarchy_louvain")
    os.makedirs(out_dir, exist_ok=True)

    node_df, edge_df, num_nodes = read_road_tables(config)
    trajs = load_trajectory_node_lists(config, args.max_trajectories)
    transition_trajs = (
        trajs[:args.transition_max_trajectories]
        if args.transition_max_trajectories is not None
        else trajs
    )
    edge_counts, out_counts, node_flow = build_transition_statistics(transition_trajs)
    graph, _ = build_weighted_graph(edge_df, edge_counts, out_counts, args)
    graph.add_nodes_from(node_df["node"].astype(int).tolist())

    importance = compute_importance(graph, num_nodes, node_flow, args, cache_dir=out_dir)
    np.save(os.path.join(out_dir, "node_importance.npy"), importance)

    node_to_level = nested_partition(graph, importance, num_nodes, args)
    save_region_tables(
        out_dir,
        node_df,
        edge_df,
        node_to_level,
        edge_counts=edge_counts,
        out_counts=out_counts,
        transition_smooth=args.transition_smooth,
    )
    np.save(
        os.path.join(out_dir, "l0_transition_features.npy"),
        build_transition_edge_features(
            edge_df,
            np.arange(num_nodes, dtype=np.int64),
            edge_counts,
            out_counts,
            transition_smooth=args.transition_smooth,
        ),
    )
    plot_partitions(out_dir, node_df, node_to_level, args)
    plot_road_partitions(out_dir, node_df, edge_df, node_to_level, args)
    print(f"hierarchy saved to: {out_dir}")


if __name__ == "__main__":
    main()
