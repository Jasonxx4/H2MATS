"""Redraw the saved Louvain road hierarchy without re-running the partition.

    python plot_hierarchy_partitions.py --dataset beijing

Reads dataset/<dataset>/{node.csv, edge_weight.csv} and
hierarchy_louvain/node_to_{l3,l2,l1}.npy, then writes into the hierarchy dir:

  hierarchy_drilldown.{pdf,png}  L3 over the city -> L2 inside one L3 region
                                 -> L1 inside one L2 region
  hierarchy_levels.{pdf,png}     L3 / L2 / L1 over one shared window, with the
                                 L3 region borders outlined
"""
import argparse
import csv
import heapq
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba_array
from matplotlib.font_manager import FontProperties
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


LEVELS = ("l3", "l2", "l1")
PANEL_LABELS = ("(a) $L_3$ macro-regions", "(b) $L_2$ districts", "(c) $L_1$ neighborhoods")
# The first five hues stay pairwise distinguishable, also under red-green colour
# blindness. The last three are spares for a region whose neighbours hold all five.
PALETTE = ("#2a78d6", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
           "#e34948", "#1baf7a", "#eb6834")
BASE_SLOTS = 5
PALETTE_RGBA = to_rgba_array(PALETTE)
CONTEXT_COLOR = "#cfcfcf"
INK = "#202020"
SCALE_BAR_KM = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
KM_PER_DEGREE = 111.32


class RoadNetwork:
    def __init__(self, xy, edges, assign):
        self.xy = xy          # [num_nodes, 2] lng/lat, NaN when missing from node.csv
        self.edges = edges    # [num_edges, 2] undirected, deduplicated
        self.assign = assign  # level -> region id per node, -1 when unassigned
        # Degrees of longitude per degree of latitude on the ground at this city.
        self.aspect = 1.0 / math.cos(math.radians(float(np.nanmean(xy[:, 1]))))


def read_columns(path, columns):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [[row[c] for c in columns] for row in csv.DictReader(f)]
    return np.asarray(rows, dtype=np.float64)


def load_network(data_dir, hierarchy_dir):
    assign = {
        level: np.load(os.path.join(hierarchy_dir, f"node_to_{level}.npy"))
        for level in LEVELS
    }
    num_nodes = len(assign["l3"])

    nodes = read_columns(os.path.join(data_dir, "node.csv"), ("node", "lng", "lat"))
    node_ids = nodes[:, 0].astype(np.int64)
    if node_ids.max() >= num_nodes:
        raise ValueError(
            f"node.csv has node id {node_ids.max()} but the hierarchy covers "
            f"{num_nodes} nodes; do {data_dir} and {hierarchy_dir} match?"
        )
    xy = np.full((num_nodes, 2), np.nan)
    xy[node_ids] = nodes[:, 1:]

    edges = read_columns(os.path.join(data_dir, "edge_weight.csv"), ("s_node", "e_node"))
    return build_network(xy, edges, assign)


def build_network(xy, edges, assign):
    """Deduplicate the directed edge list and drop edges that cannot be drawn."""
    edges = np.unique(np.sort(np.asarray(edges, dtype=np.int64), axis=1), axis=0)
    drawable = np.isfinite(xy[:, 0]) & (assign["l3"] >= 0)
    edges = edges[
        (edges[:, 0] != edges[:, 1]) & drawable[edges[:, 0]] & drawable[edges[:, 1]]
    ]
    return RoadNetwork(xy, edges, assign)


def color_regions(num_regions, regions, pairs):
    """Give every region a palette slot that differs from its road-connected neighbours.

    DSATUR order (most constrained region first). Among the free base slots the
    least used one wins, so the hues stay balanced over the map.
    """
    neighbors = defaultdict(set)
    for a, b in pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)

    slot = np.full(num_regions, -1, dtype=np.int64)
    usage = [0] * len(PALETTE)
    taken = defaultdict(set)
    heap = [(0, -len(neighbors[region]), region) for region in regions]
    heapq.heapify(heap)
    while heap:
        neg_saturation, _, region = heapq.heappop(heap)
        if slot[region] >= 0 or -neg_saturation != len(taken[region]):
            continue
        free = [s for s in range(BASE_SLOTS) if s not in taken[region]]
        spare = [s for s in range(BASE_SLOTS, len(PALETTE)) if s not in taken[region]]
        if free:
            choice = min(free, key=lambda s: usage[s])
        elif spare:
            choice = spare[0]
        else:
            # Every slot is taken around this region: reuse the one fewest neighbours hold.
            around = [slot[other] for other in neighbors[region]]
            choice = min(range(len(PALETTE)), key=around.count)
        slot[region] = choice
        usage[choice] += 1
        for other in neighbors[region]:
            if slot[other] < 0 and choice not in taken[other]:
                taken[other].add(choice)
                heapq.heappush(heap, (-len(taken[other]), -len(neighbors[other]), other))
    return slot


def node_slots(net, level, inside=None):
    """Palette slot per node for the `level` regions; nodes outside `inside` are context (-1)."""
    assign = net.assign[level]
    keep = assign >= 0 if inside is None else inside & (assign >= 0)
    u, v = net.edges[:, 0], net.edges[:, 1]
    cross = keep[u] & keep[v] & (assign[u] != assign[v])
    pairs = np.unique(np.stack([assign[u[cross]], assign[v[cross]]], axis=1), axis=0)
    regions = np.unique(assign[keep])
    region_slot = color_regions(int(assign.max()) + 1, regions.tolist(), pairs.tolist())
    return np.where(keep, region_slot[assign], -1)


def fit_window(net, x0, x1, y0, y1, pad, ratio=1.0):
    """Grow a lng/lat box into a window `ratio` times wider than tall on the ground,
    so the panel fills its axes."""
    width = (1.0 + pad) * max((x1 - x0) / net.aspect, ratio * (y1 - y0))
    half_x, half_y = 0.5 * width * net.aspect, 0.5 * width / ratio
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return cx - half_x, cx + half_x, cy - half_y, cy + half_y


def crop_to_ratio(net, window, ratio):
    """Cut the top and bottom of `window` so it is `ratio` times wider than tall on the
    ground; the width, and with it the map scale, stays the same."""
    x0, x1, y0, y1 = window
    half_y = 0.5 * (x1 - x0) / net.aspect / ratio
    cy = 0.5 * (y0 + y1)
    return x0, x1, cy - half_y, cy + half_y


def region_window(net, mask, pad, ratio=1.0):
    xy = net.xy[mask]
    return fit_window(
        net,
        np.nanmin(xy[:, 0]), np.nanmax(xy[:, 0]),
        np.nanmin(xy[:, 1]), np.nanmax(xy[:, 1]),
        pad, ratio,
    )


def network_window(net, trim):
    """Whole network, minus the `trim` fraction of stray nodes on each side."""
    x0, x1 = np.nanquantile(net.xy[:, 0], [trim, 1.0 - trim])
    y0, y1 = np.nanquantile(net.xy[:, 1], [trim, 1.0 - trim])
    return fit_window(net, x0, x1, y0, y1, pad=0.04)


def nodes_in_window(net, window):
    x0, x1, y0, y1 = window
    x, y = net.xy[:, 0], net.xy[:, 1]
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def pick_focus(net, args):
    """Choose the L3 region, and the L2 region inside it, that the drill-down zooms into."""
    l3, l2, l1 = (net.assign[level] for level in LEVELS)
    located = np.isfinite(net.xy[:, 0]) & (l1 >= 0)
    if args.focus_l2 is not None:
        members = located & (l2 == args.focus_l2)
        if not members.any():
            raise ValueError(f"--focus-l2 {args.focus_l2} has no drawable node")
        return int(l3[members][0]), args.focus_l2

    focus_l3 = args.focus_l3
    if focus_l3 is None:
        nodes = np.flatnonzero(located)
        lng, lat = args.focus or np.median(net.xy[nodes], axis=0)
        distance = ((net.xy[nodes, 0] - lng) / net.aspect) ** 2 + (net.xy[nodes, 1] - lat) ** 2
        focus_l3 = int(l3[nodes[int(np.argmin(distance))]])

    children = np.unique(l2[located & (l3 == focus_l3)])
    if len(children) == 0:
        raise ValueError(f"--focus-l3 {focus_l3} has no drawable node")
    # The L2 child with the most L1 regions gives the last panel the most to show.
    l1_counts = [len(np.unique(l1[located & (l2 == child)])) for child in children]
    return focus_l3, int(children[int(np.argmax(l1_counts))])


def drop_short_pieces(segments, min_km):
    """Drop connected border pieces shorter than `min_km` in total.

    These are rings around a few stray nodes assigned to a neighbouring region;
    real borders join into long pieces. Shared endpoints are bit-identical, because
    both triangles compute an edge midpoint from the same two corners.
    """
    if len(segments) == 0:
        return segments
    _, point = np.unique(segments.reshape(-1, 2), axis=0, return_inverse=True)
    point = point.reshape(-1, 2)
    parent = list(range(int(point.max()) + 1))

    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for a, b in point:
        parent[root(a)] = root(b)
    piece = np.array([root(a) for a in point[:, 0]])
    length_km = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) * KM_PER_DEGREE
    total = np.bincount(piece, weights=length_km)
    return segments[total[piece] >= min_km]


def region_borders(net, level, window, max_edge_km, min_piece_km):
    """Line segments that separate the `level` regions around `window`.

    The road nodes are triangulated. In a triangle whose corners lie in two regions,
    the midpoints of its two mixed edges are joined; with three regions, all three
    midpoints meet at the centroid. The result traces the Voronoi borders between
    regions. Triangles with an edge longer than `max_edge_km` span empty land and
    are skipped, and pieces shorter than `min_piece_km` are dropped.
    """
    x0, x1, y0, y1 = window
    margin_x, margin_y = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
    around = (x0 - margin_x, x1 + margin_x, y0 - margin_y, y1 + margin_y)
    assign = net.assign[level]
    nodes = np.flatnonzero(np.isfinite(net.xy[:, 0]) & (assign >= 0) & nodes_in_window(net, around))
    # Work in ground units, so triangle shapes and edge lengths are not skewed by latitude.
    ground, first = np.unique(
        np.column_stack([net.xy[nodes, 0] / net.aspect, net.xy[nodes, 1]]),
        axis=0, return_index=True,
    )
    if len(ground) < 3:
        return np.empty((0, 2, 2))
    triangles = Triangulation(ground[:, 0], ground[:, 1]).triangles
    corners = ground[triangles]
    region = assign[nodes[first]][triangles]

    # Edge k runs from corner k to corner k + 1.
    edge_length = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2)
    middle = 0.5 * (corners + np.roll(corners, -1, axis=1))
    mixed = region != np.roll(region, -1, axis=1)
    short = edge_length.max(axis=1) * KM_PER_DEGREE <= max_edge_km
    two = short & (mixed.sum(axis=1) == 2)
    three = short & (mixed.sum(axis=1) == 3)
    centroid = corners[three].mean(axis=1)
    segments = np.concatenate(
        [middle[two][mixed[two]].reshape(-1, 2, 2)]
        + [np.stack([middle[three][:, k], centroid], axis=1) for k in range(3)]
    )
    segments = drop_short_pieces(segments, min_piece_km)
    segments[..., 0] *= net.aspect
    return segments


def draw_panel(ax, net, slots, window, args, show_nodes=False, scale_bar_loc="lower left",
               border_level=None):
    """Draw the roads inside `window`; an edge joining two colours switches at its midpoint."""
    x0, x1, y0, y1 = window
    start, end = net.xy[net.edges[:, 0]], net.xy[net.edges[:, 1]]
    visible = (
        (np.maximum(start[:, 0], end[:, 0]) >= x0)
        & (np.minimum(start[:, 0], end[:, 0]) <= x1)
        & (np.maximum(start[:, 1], end[:, 1]) >= y0)
        & (np.minimum(start[:, 1], end[:, 1]) <= y1)
    )
    u, v = net.edges[visible, 0], net.edges[visible, 1]
    start, end = start[visible], end[visible]
    split = slots[u] != slots[v]
    middle = 0.5 * (start[split] + end[split])
    segments = np.concatenate([
        np.stack([start[~split], end[~split]], axis=1),
        np.stack([start[split], middle], axis=1),
        np.stack([end[split], middle], axis=1),
    ])
    segment_slots = np.concatenate([slots[u[~split]], slots[u[split]], slots[v[split]]])
    colored = segment_slots >= 0

    # Fewer roads in view leave room for thicker lines.
    width = args.line_scale * float(np.clip(2.2 - 0.38 * math.log10(max(len(u), 1)), 0.3, 1.1))
    style = dict(capstyle="round", rasterized=not args.vector)
    if (~colored).any():
        ax.add_collection(LineCollection(
            segments[~colored], colors=CONTEXT_COLOR, linewidths=0.7 * width, zorder=1, **style))
    if colored.any():
        ax.add_collection(LineCollection(
            segments[colored], colors=PALETTE_RGBA[segment_slots[colored]],
            linewidths=width, zorder=2, **style))

    if show_nodes:
        nodes = np.flatnonzero((slots >= 0) & nodes_in_window(net, window))
        ax.scatter(
            net.xy[nodes, 0], net.xy[nodes, 1],
            s=(2.6 * width) ** 2, c=PALETTE_RGBA[slots[nodes]],
            edgecolors="white", linewidths=0.25, zorder=3,
        )

    if border_level is not None:
        ax.add_collection(LineCollection(
            region_borders(net, border_level, window,
                           args.border_max_edge_km, args.border_min_km),
            colors=INK, linewidths=args.border_width, capstyle="round",
            joinstyle="round", zorder=4,
        ))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(net.aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.6)
    add_scale_bar(ax, net, window, scale_bar_loc)


def quieter_bottom_corner(net, slots, window):
    """'lower left' or 'lower right', whichever overlaps fewer coloured nodes."""
    x0, x1, y0, y1 = window
    width, height = 0.4 * (x1 - x0), 0.2 * (y1 - y0)
    colored = slots >= 0
    counts = {
        loc: int((colored & nodes_in_window(net, box)).sum())
        for loc, box in (("lower left", (x0, x0 + width, y0, y0 + height)),
                         ("lower right", (x1 - width, x1, y0, y0 + height)))
    }
    return min(counts, key=counts.get)


def add_scale_bar(ax, net, window, loc="lower left"):
    x0, x1, y0, y1 = window
    km_per_lng = KM_PER_DEGREE / net.aspect
    limit = 0.3 * (x1 - x0) * km_per_lng
    length = max((km for km in SCALE_BAR_KM if km <= limit), default=SCALE_BAR_KM[0])
    label = f"{length:g} km" if length >= 1 else f"{length * 1000:g} m"
    bar = AnchoredSizeBar(
        ax.transData, length / km_per_lng, label, loc=loc,
        pad=0.25, borderpad=0.4, sep=2, color=INK, frameon=True,
        fontproperties=FontProperties(size=6.5),
    )
    bar.patch.set(facecolor="white", edgecolor="none", alpha=0.85)
    bar.set_zorder(6)
    ax.add_artist(bar)


def mark_zoom(fig, ax_from, ax_to, window):
    """Box `window` in ax_from and tie its right corners to the left corners of ax_to."""
    # Clip to the source view; an --overview-bbox may leave the focus region partly outside.
    (view_x0, view_x1), (view_y0, view_y1) = ax_from.get_xlim(), ax_from.get_ylim()
    x0, x1 = max(window[0], view_x0), min(window[1], view_x1)
    y0, y1 = max(window[2], view_y0), min(window[3], view_y1)
    if x0 >= x1 or y0 >= y1:
        return
    ax_from.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor=INK, linewidth=0.8, zorder=5))
    for corner, target in (((x1, y1), (0, 1)), ((x1, y0), (0, 0))):
        fig.add_artist(ConnectionPatch(
            xyA=corner, coordsA=ax_from.transData,
            xyB=target, coordsB=ax_to.transAxes,
            color=INK, linewidth=0.5, alpha=0.6,
        ))


def apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def new_figure(args):
    apply_paper_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(args.fig_width, args.fig_width / (3 * args.panel_ratio) + 0.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.1, wspace=0.1)
    return fig, axes


def save_figure(fig, out_dir, name, args):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=args.dpi,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_drilldown(net, focus_l3, focus_l2, out_dir, args):
    in_l3 = net.assign["l3"] == focus_l3
    in_l2 = net.assign["l2"] == focus_l2
    if args.overview_bbox:
        overview = fit_window(net, *args.overview_bbox, pad=0.0)
    else:
        overview = network_window(net, args.trim)
    # The overview may lose its outskirts, but the zoomed regions must stay whole,
    # so their windows widen instead.
    windows = (
        crop_to_ratio(net, overview, args.panel_ratio),
        region_window(net, in_l3, pad=0.08, ratio=args.panel_ratio),
        region_window(net, in_l2, pad=0.08, ratio=args.panel_ratio),
    )
    # Each panel colours only the children of the region boxed in the previous one.
    parents = (None, in_l3, in_l2)

    fig, axes = new_figure(args)
    for ax, window, level, inside, label in zip(axes, windows, LEVELS, parents, PANEL_LABELS):
        slots = node_slots(net, level, inside)
        draw_panel(ax, net, slots, window, args, show_nodes=level == "l1",
                   scale_bar_loc=quieter_bottom_corner(net, slots, window))
        ax.set_xlabel(label, labelpad=4)
    mark_zoom(fig, axes[0], axes[1], windows[1])
    mark_zoom(fig, axes[1], axes[2], windows[2])
    save_figure(fig, out_dir, "hierarchy_drilldown", args)


def plot_levels(net, focus_l3, out_dir, args):
    if args.levels_bbox == "full":
        window = network_window(net, args.trim)
    elif args.levels_bbox:
        window = fit_window(net, *args.levels_bbox, pad=0.0)
    else:
        window = region_window(net, net.assign["l3"] == focus_l3, pad=0.6)
    window = crop_to_ratio(net, window, args.panel_ratio)

    fig, axes = new_figure(args)
    for ax, level, label in zip(axes, LEVELS, PANEL_LABELS):
        # Only the coarsest level is outlined; finer borders would bury the roads.
        border_level = level if level == "l3" and args.border_width > 0 else None
        draw_panel(ax, net, node_slots(net, level), window, args, border_level=border_level)
        ax.set_xlabel(label, labelpad=4)
    save_figure(fig, out_dir, "hierarchy_levels", args)


def float_list(count):
    def parse(text):
        values = [float(part) for part in text.split(",")]
        if len(values) != count:
            raise argparse.ArgumentTypeError(f"expected {count} comma-separated numbers")
        return values
    return parse


def levels_bbox(text):
    return text if text == "full" else float_list(4)(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the saved road hierarchy as readable paper figures.",
        epilog="Write values that start with a minus sign as --focus=-8.61,41.15.",
    )
    parser.add_argument("--dataset", default="beijing", choices=["beijing", "porto", "tdrive"])
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--data-dir", default=None,
                        help="Folder with node.csv and edge_weight.csv "
                             "(default: <data-root>/<dataset>).")
    parser.add_argument("--hierarchy-dir", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--focus", type=float_list(2), default=None, metavar="LNG,LAT",
                        help="Zoom into the L3 region at this point (default: median node).")
    parser.add_argument("--focus-l3", type=int, default=None)
    parser.add_argument("--focus-l2", type=int, default=None,
                        help="L2 region of the last panel "
                             "(default: the child of the focus L3 region with the most L1 regions).")
    parser.add_argument("--overview-bbox", type=float_list(4), default=None,
                        metavar="LNG_MIN,LNG_MAX,LAT_MIN,LAT_MAX",
                        help="Window of drill-down panel (a) (default: whole network).")
    parser.add_argument("--levels-bbox", type=levels_bbox, default=None,
                        metavar="LNG_MIN,LNG_MAX,LAT_MIN,LAT_MAX|full",
                        help="Shared window of the per-level figure "
                             "(default: around the focus L3 region).")
    parser.add_argument("--trim", type=float, default=0.005,
                        help="Fraction of stray nodes cut from each side of the whole-network window.")

    parser.add_argument("--fig-width", type=float, default=7.0)
    parser.add_argument("--panel-ratio", type=float, default=1.5,
                        help="Width over height of every panel (1 gives square panels).")
    parser.add_argument("--line-scale", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--vector", action="store_true",
                        help="Keep roads as vector paths (much larger PDF).")
    parser.add_argument("--border-width", type=float, default=0.8,
                        help="Line width of the L3 region borders in the per-level figure "
                             "(0 turns them off).")
    parser.add_argument("--border-max-edge-km", type=float, default=3.0,
                        help="Draw no border across empty land wider than this.")
    parser.add_argument("--border-min-km", type=float, default=1.0,
                        help="Hide border pieces shorter than this, such as rings "
                             "around a few stray nodes.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir or os.path.join(args.data_root, args.dataset)
    hierarchy_dir = args.hierarchy_dir or os.path.join(data_dir, "hierarchy_louvain")
    out_dir = args.output_dir or hierarchy_dir
    os.makedirs(out_dir, exist_ok=True)

    net = load_network(data_dir, hierarchy_dir)
    focus_l3, focus_l2 = pick_focus(net, args)
    print(f"focus regions: l3={focus_l3}, l2={focus_l2}")
    plot_drilldown(net, focus_l3, focus_l2, out_dir, args)
    plot_levels(net, focus_l3, out_dir, args)
    print(f"figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
