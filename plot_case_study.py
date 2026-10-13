"""Pick a case-study query from exported retrieval results and draw it.

    python plot_case_study.py --input case_study/porto_TP_retrieval.npz

Reads the file written by export_case_retrieval.py, prints both models' metrics
and how their top-1 hits split, lists the queries that fit the case rule, and
draws one of them (default: the first listed) next to its results, in three
panels side by side over the same window:

  case_<query>.{pdf,png}  (a) SPGMT  (b) w/o AR: H²MATS without adaptive
                          retrieval, i.e. its embedding ranking  (c) H²MATS
                          after exact reranking of its top --rerank-pool candidates

Exact reranking sorts the candidates by their true distance. The export keeps
the ground-truth top-50, so panel (c) orders candidates by ground-truth rank;
the few outside the top-50 keep their embedding order behind them.

TP is a Hausdorff distance over the road nodes a trajectory visits, so every
trajectory with the same node set as the query sits at distance 0. get_label in
generate_trajectory.py drops the first zero-distance entry as "the query
itself", which for such queries may drop a duplicate and keep the query in its
own labels. The case rule therefore skips queries that have a duplicate in the
test set, and asks that SPGMT's miss be visible on the map:

  - H²MATS's embedding ranks the true nearest neighbour first;
  - SPGMT's top-1 lies outside the ground-truth top-10, and its ground
    Hausdorff distance to the query is at least MIN_DETOUR_KM and
    DETOUR_RATIO times that of H²MATS's top-1;
  - the query length is typical (inter-quartile range of the test set).

Candidates are listed by how many road nodes SPGMT's top-1 shares with the
query, so the default case is a distractor that follows most of the route and
then leaves it. Pass --query to draw any other test query.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_hierarchy_partitions import (
    INK, KM_PER_DEGREE, apply_paper_style, build_network, draw_panel, fit_window, save_figure,
)

PANELS = (
    ("spgmt", "SPGMT"),
    # Named as in the ablation study: H2MATS without adaptive retrieval.
    ("ours", "w/o AR"),
    ("rerank", "H²MATS"),
)
# The two retrieval runs stored in the export; "rerank" is derived from "ours".
MODELS = PANELS[:2]
MIN_DETOUR_KM = 0.5
DETOUR_RATIO = 3.0
QUERY_STYLE = dict(color=INK, alpha=0.3, solid_capstyle="round")
# The first three categorical hues stay apart pairwise, also for colour-blind readers.
RESULT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
# Lower-ranked results are wider and drawn underneath, so a road shared by several
# results shows every one of them as a nested band; width also encodes the rank.
RESULT_WIDTHS = (1.1, 2.3, 3.5)
# Fainter wide bands keep the low-ranked results from outweighing the top-1.
RESULT_ALPHAS = (1.0, 0.85, 0.7)
QUERY_EXTRA_WIDTH = 2.4
MARKER_SIZE = 4
# Panels are at least this much wider than tall, and at most PANEL_RATIO_RANGE[1].
PANEL_RATIO_RANGE = (1.2, 2.0)
# How far, as a share of the query-and-top-1 box, the frame may grow to keep other results in view.
EXTRA_FRAME = 0.3
# Inches above the panels for the shared legend and below them for the two-line captions.
LEGEND_INCHES, CAPTION_INCHES = 0.26, 0.4


def load(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def trajectory(data, index):
    offsets = data["traj_offsets"]
    return data["traj_nodes"][offsets[index]:offsets[index + 1]]


def node_overlap(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / max(len(a | b), 1)


def ground_xy(data):
    """Node coordinates in km on a local flat projection."""
    xy = data["node_xy"]
    scale = np.cos(np.radians(np.nanmean(xy[:, 1])))
    return xy * np.array([scale * KM_PER_DEGREE, KM_PER_DEGREE])


def hausdorff_km(ground, a, b):
    gap = np.linalg.norm(ground[a][:, None, :] - ground[b][None, :, :], axis=-1)
    return float(max(gap.min(axis=1).max(), gap.min(axis=0).max()))


def duplicate_counts(data):
    """For each test trajectory, how many others visit exactly the same road nodes (TP = 0)."""
    keys = [np.unique(trajectory(data, i)).tobytes() for i in range(len(data["gt_top"]))]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return counts[inverse] - 1


def ground_truth_rank(data, query, result):
    hits = np.flatnonzero(data["gt_top"][query] == result)
    return int(hits[0]) + 1 if len(hits) else None


def rank_text(rank):
    return str(rank) if rank else ">50"


def panel_results(data, query, key, args):
    """The top results one panel shows."""
    if key != "rerank":
        return data[f"{key}_top"][query, :args.top]
    pool = data["ours_top"][query, :args.rerank_pool]
    ranks = [ground_truth_rank(data, query, c) or np.inf for c in pool]
    return pool[np.argsort(ranks, kind="stable")[:args.top]]


def summarize(data, duplicates):
    for name, label in MODELS:
        values = "  ".join(f"{m}={100 * v:.2f}" for m, v in zip(data["metric_names"], data[f"{name}_metrics"]))
        print(f"{label}:\n  {values}")

    truth = data["gt_top"][:, 0]
    ours = data["ours_top"][:, 0] == truth
    spgmt = data["spgmt_top"][:, 0] == truth
    total = len(truth)
    print("\nTrue nearest neighbour ranked first (R1@1 per query, embeddings only):")
    for text, mask in (("both", ours & spgmt), ("H2MATS only", ours & ~spgmt),
                       ("SPGMT only", ~ours & spgmt), ("neither", ~ours & ~spgmt)):
        print(f"  {text:12s} {int(mask.sum()):6d}  ({100 * mask.mean():5.2f}%)")
    print(f"  total        {total:6d}")

    print("\nQueries with other test trajectories on exactly the same road nodes (TP = 0):")
    for least in (1, 10, 50):
        share = (duplicates >= least).mean()
        print(f"  >= {least:2d} such trajectories: {int((duplicates >= least).sum()):6d}  ({100 * share:5.2f}%)")


def candidates(data, ground, duplicates):
    truth = data["gt_top"]
    lengths = np.diff(data["traj_offsets"])
    low, high = np.quantile(lengths, [0.25, 0.75])
    ours_top1, spgmt_top1 = data["ours_top"][:, 0], data["spgmt_top"][:, 0]
    spgmt_miss = ~(truth[:, :10] == spgmt_top1[:, None]).any(axis=1)
    mask = (
        (ours_top1 == truth[:, 0])
        & spgmt_miss
        & (duplicates == 0)
        & (lengths >= low) & (lengths <= high)
    )
    rows = []
    for query in np.flatnonzero(mask):
        nodes = trajectory(data, query)
        spgmt_detour = hausdorff_km(ground, nodes, trajectory(data, spgmt_top1[query]))
        ours_detour = hausdorff_km(ground, nodes, trajectory(data, ours_top1[query]))
        if spgmt_detour < max(MIN_DETOUR_KM, DETOUR_RATIO * ours_detour):
            continue
        overlap = node_overlap(nodes, trajectory(data, spgmt_top1[query]))
        rows.append((int(query), int(lengths[query]), overlap, spgmt_detour, ours_detour,
                     int(data["spgmt_gt1_rank"][query])))
    rows.sort(key=lambda row: -row[2])
    print(f"\n{len(rows)} queries fit the case rule (length {low:.0f}-{high:.0f} nodes); top 10:")
    print("  query  length  SPGMT top-1: overlap  detour km   H2MATS top-1: detour km"
          "   SPGMT rank of true NN")
    for query, length, overlap, spgmt_detour, ours_detour, rank in rows[:10]:
        print(f"  {query:5d}  {length:6d}  {overlap:20.2f}  {spgmt_detour:9.2f}"
              f"   {ours_detour:23.2f}   {rank:21d}")
    return rows


def describe(data, ground, query, shown):
    distance = str(data["distance_type"])
    nodes = trajectory(data, query)
    print(f"\nquery {query}: {len(nodes)} nodes")
    for key, label in PANELS:
        print(f"  {label}:")
        for k, result in enumerate(shown[key], start=1):
            other = trajectory(data, result)
            print(f"    top-{k}: trajectory {int(result):5d}  {distance} rank "
                  f"{rank_text(ground_truth_rank(data, query, result)):>3}"
                  f"  node overlap {node_overlap(nodes, other):.2f}"
                  f"  detour {hausdorff_km(ground, nodes, other):.2f} km")


def row_figure(width, ratio, count, gap=0.1):
    """Map panels side by side under one shared legend, each with a two-line caption."""
    apply_paper_style()
    panel_width = (width - (count - 1) * gap) / count
    panel_height = panel_width / ratio
    height = LEGEND_INCHES + panel_height + CAPTION_INCHES
    fig = plt.figure(figsize=(width, height))
    axes = [fig.add_axes([i * (panel_width + gap) / width, CAPTION_INCHES / height,
                          panel_width / width, panel_height / height])
            for i in range(count)]
    return fig, axes, (CAPTION_INCHES + panel_height) / height


def case_window(net, data, query, shown, ratio):
    """Frame the query and every panel's top-1, plus whatever of the other results
    stays within EXTRA_FRAME of that core box; a far-off result may leave the frame."""
    def bounds(trajectories):
        xy = net.xy[np.concatenate([trajectory(data, t) for t in trajectories])]
        return xy.min(axis=0), xy.max(axis=0)

    core_low, core_high = bounds([query] + [results[0] for results in shown.values()])
    all_low, all_high = bounds([query] + [r for results in shown.values() for r in results])
    reach = EXTRA_FRAME * (core_high - core_low)
    (x0, y0) = np.maximum(all_low, core_low - reach)
    (x1, y1) = np.minimum(all_high, core_high + reach)
    ratio = ratio or float(np.clip((x1 - x0) / net.aspect / (y1 - y0), *PANEL_RATIO_RANGE))
    return fit_window(net, x0, x1, y0, y1, pad=0.12, ratio=ratio), ratio


def emptiest_corner(net, data, query, shown, window):
    """The panel corner crossed by the fewest drawn trajectory nodes, for the scale bar."""
    x0, x1, y0, y1 = window
    xy = net.xy[np.concatenate([trajectory(data, t) for t in
                                [query] + [r for results in shown.values() for r in results]])]
    left, low = xy[:, 0] < x0 + 0.35 * (x1 - x0), xy[:, 1] < y0 + 0.3 * (y1 - y0)
    right, high = xy[:, 0] > x1 - 0.35 * (x1 - x0), xy[:, 1] > y1 - 0.3 * (y1 - y0)
    corners = {"lower left": left & low, "lower right": right & low,
               "upper left": left & high, "upper right": right & high}
    return min(corners, key=lambda corner: int(corners[corner].sum()))


def draw_case(net, data, query, shown, args):
    distance = str(data["distance_type"])
    window, ratio = case_window(net, data, query, shown, args.panel_ratio)
    scale_bar_loc = emptiest_corner(net, data, query, shown, window)
    context = np.full(len(net.xy), -1)
    query_xy = net.xy[trajectory(data, query)]
    query_width = RESULT_WIDTHS[args.top - 1] + QUERY_EXTRA_WIDTH

    fig, axes, legend_y = row_figure(args.fig_width, ratio, len(PANELS))
    for ax, (key, label), letter in zip(axes, PANELS, "abc"):
        draw_panel(ax, net, context, window, args, scale_bar_loc=scale_bar_loc)
        ax.plot(query_xy[:, 0], query_xy[:, 1], linewidth=query_width, zorder=3, **QUERY_STYLE)
        # Top-1 is drawn last, thinnest, so wider lower-ranked results stay visible around it.
        for k in reversed(range(len(shown[key]))):
            result_xy = net.xy[trajectory(data, shown[key][k])]
            ax.plot(result_xy[:, 0], result_xy[:, 1], color=RESULT_COLORS[k],
                    linewidth=RESULT_WIDTHS[k], alpha=RESULT_ALPHAS[k],
                    solid_capstyle="round", solid_joinstyle="round",
                    zorder=4 + (len(shown[key]) - k))
        ax.plot(*query_xy[0], marker="o", color=INK, markersize=MARKER_SIZE, zorder=9)
        ax.plot(*query_xy[-1], marker="s", color=INK, markersize=MARKER_SIZE, zorder=9)
        ranks = ", ".join(rank_text(ground_truth_rank(data, query, r)) for r in shown[key])
        tops = "/".join(str(k + 1) for k in range(len(shown[key])))
        ax.set_xlabel(f"({letter}) {label}\nTop-{tops} {distance} rank: {ranks}", labelpad=4)

    handles = [
        Line2D([], [], linewidth=query_width, label="Query", **{**QUERY_STYLE, "solid_capstyle": "butt"}),
        Line2D([], [], color=INK, marker="o", markersize=MARKER_SIZE, linestyle="none", label="Query start"),
        Line2D([], [], color=INK, marker="s", markersize=MARKER_SIZE, linestyle="none", label="Query end"),
    ] + [
        Line2D([], [], color=RESULT_COLORS[k], linewidth=RESULT_WIDTHS[k], alpha=RESULT_ALPHAS[k],
               solid_capstyle="butt", label=f"Top-{k + 1}")
        for k in range(args.top)
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, legend_y), ncol=len(handles),
               frameon=False, fontsize=7, handlelength=2.0, columnspacing=1.4, handletextpad=0.6)
    save_figure(fig, args.output_dir, f"case_{query}", args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="npz written by export_case_retrieval.py")
    parser.add_argument("--query", type=int, default=None, help="Test query to draw (default: first candidate).")
    parser.add_argument("--top", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--rerank-pool", type=int, default=50,
                        help="How many H²MATS embedding candidates exact reranking re-sorts (at most 50).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fig-width", type=float, default=7.0,
                        help="Figure width in inches (default: full width of a two-column paper).")
    parser.add_argument("--panel-ratio", type=float, default=None,
                        help="Panel width / height (default: from the trajectories, 1.2-2).")
    parser.add_argument("--line-scale", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--raster", action="store_true",
                        help="Rasterize the background roads (smaller PDF).")
    return parser.parse_args()


def main():
    args = parse_args()
    args.vector = not args.raster
    args.output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(args.output_dir, exist_ok=True)

    data = load(args.input)
    ground = ground_xy(data)
    duplicates = duplicate_counts(data)
    summarize(data, duplicates)
    rows = candidates(data, ground, duplicates)
    query = args.query if args.query is not None else (rows[0][0] if rows else None)
    if query is None:
        raise SystemExit("no query fits the case rule; pass --query")

    assign = {level: data[f"node_to_{level}"] for level in ("l1", "l2", "l3")}
    net = build_network(data["node_xy"], data["edges"], assign)
    shown = {key: panel_results(data, query, key, args) for key, _ in PANELS}
    describe(data, ground, query, shown)
    draw_case(net, data, query, shown, args)
    print(f"\nfigure saved to: {os.path.join(args.output_dir, f'case_{query}')}.pdf/png")


if __name__ == "__main__":
    main()
