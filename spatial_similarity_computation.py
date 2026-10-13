import argparse
from STmatching_distribution_ver import network_data
from multiprocessing import Pool
import numpy as np
import networkx as nx
import numba
import pandas as pd
import os
from setting import SetParameter
config = SetParameter()

dataset = str(config.dataset)
dataset_point = config.pointnum[str(config.dataset)]
true_point = config.truenum[str(config.dataset)]

distance_matrix = None
hot_node_id = None
node_edge_keys = None
longest_traj_len = None
_worker_trajectories = None
_worker_output_root = None
_worker_split = None
_worker_distance_type = None


def generate_node_edge_interation(count):
    edge = pd.read_csv(str(config.edge_file))
    node_s = edge.s_node.to_numpy(dtype=np.int64)
    node_e = edge.e_node.to_numpy(dtype=np.int64)
    diagonal = np.arange(count, dtype=np.int64)
    edge_keys = np.concatenate([
        diagonal * count + diagonal,
        node_s * count + node_e,
        node_e * count + node_s,
    ])
    return np.unique(edge_keys)

def _point_batch_ranges(batch_size=1000):
    return [
        (start, min(start + batch_size, dataset_point))
        for start in range(0, dataset_point, batch_size)
    ]


def batch_Point_distance(workers=30):
    with Pool(processes=max(1, workers)) as pool:
        results = []
        for batch_start, batch_end in _point_batch_ranges():
            print('batch Point distance:', batch_end)
            results.append(pool.apply_async(
                parallel_point_com,
                (batch_end, list(range(batch_start, batch_end))),
            ))
        for result in results:
            result.get()


def merge_Point_distance():
    output_file = point_matrix_path()
    ranges = _point_batch_ranges()
    for batch_start, batch_end in ranges:
        batch_file = os.path.join(
            'ground_truth',
            dataset,
            f'Point_dis_matrix_{batch_end}.npy',
        )
        batch = np.load(batch_file, mmap_mode='r')
        expected_shape = (batch_end - batch_start, dataset_point)
        if batch.shape != expected_shape:
            raise ValueError(
                f'Point-distance batch {batch_file} has shape {batch.shape}; '
                f'expected {expected_shape}.'
            )

    merged = np.lib.format.open_memmap(
        output_file,
        mode='w+',
        dtype=np.float32,
        shape=(dataset_point, dataset_point),
    )
    for batch_start, batch_end in ranges:
        batch_file = os.path.join(
            'ground_truth',
            dataset,
            f'Point_dis_matrix_{batch_end}.npy',
        )
        batch = np.load(batch_file, mmap_mode='r')
        merged[batch_start:batch_end] = batch
    merged.flush()
    del merged
    print(f'success merge Point distance: {output_file}')


def parallel_point_com(i, id_list=None):
    if id_list is None:
        id_list = []
    batch_list = []

    for k in id_list:
        one_list = []
        if k in roadnetwork.nodes():
            length_list = nx.shortest_path_length(roadnetwork, source=k, weight='distance')
            for j in range(dataset_point):
                if (j in length_list.keys()) == True:
                    one_list.append(length_list[j])
                else:
                    one_list.append(-1)
            batch_list.append(np.array(one_list, dtype=np.float32))
        else:
            one_list = [-1 for j in range(dataset_point)]
            batch_list.append(np.array(one_list, dtype=np.float32))

    batch_list = np.array(batch_list, dtype=np.float32)
    p = './ground_truth/{}/'.format(dataset)
    if not os.path.exists(p):
        os.makedirs(p)
    np.save('./ground_truth/{}/Point_dis_matrix_{}.npy'.format(dataset, str(i)), batch_list)


def point_matrix_path():
    return os.path.join('ground_truth', dataset, 'Point_dis_matrix.npy')


def generate_point_matrix():
    return np.load(point_matrix_path(), mmap_mode='r')


def get_similarity_output_root():
    return str(config.ground_truth_dir)


def get_split_bounds(total_size, split):
    return config.get_split_bounds(total_size, split)


def _batch_ranges(sample_len, batch_size):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return [
        (start, min(start + batch_size, sample_len))
        for start in range(0, sample_len, batch_size)
    ]


def _batch_file_path(output_root, split, batch_end):
    return os.path.join(
        output_root,
        f'{split}_batch',
        f'{config.distance_type}_spatial_distance_{batch_end}.npy',
    )


def _batch_file_is_complete(path, row_count, column_count):
    if not os.path.exists(path):
        return False
    try:
        batch = np.load(path, mmap_mode='r')
        return batch.shape == (row_count, column_count)
    except (OSError, ValueError):
        return False


def _initialize_similarity_worker(
        trajectory_file, split_start, split_end, output_root, split,
        distance_type, edge_keys, longest_length, neterp_hot_node):
    global _worker_trajectories, _worker_output_root, _worker_split, _worker_distance_type
    global distance_matrix, node_edge_keys, longest_traj_len, hot_node_id

    # Forked Linux workers inherit this array without serializing it. Spawned
    # workers fall back to loading the trajectory file once per worker.
    if _worker_trajectories is None:
        all_trajectories = np.load(trajectory_file, allow_pickle=True)
        _worker_trajectories = all_trajectories[split_start:split_end]

    _worker_output_root = output_root
    _worker_split = split
    _worker_distance_type = distance_type
    node_edge_keys = edge_keys
    longest_traj_len = longest_length
    hot_node_id = neterp_hot_node
    if distance_type == 'LCRS':
        distance_matrix = None
    else:
        distance_matrix = np.load(point_matrix_path(), mmap_mode='r')


def Traj_distance(batch_start, batch_end):
    test_list = _worker_trajectories
    sample_list = test_list[batch_start:batch_end]
    all_dis_list = np.empty((len(sample_list), len(test_list)), dtype=np.float32)

    for sample_idx, sample in enumerate(sample_list):
        for traj_idx, traj in enumerate(test_list):
            if _worker_distance_type == 'TP':
                distance = TP_dis(sample, traj, distance_matrix)
            elif _worker_distance_type == 'DITA':
                distance = DITA_dis(sample, traj, distance_matrix)
            elif _worker_distance_type == 'discret_frechet':
                distance = frechet_dis(sample, traj, distance_matrix)
            elif _worker_distance_type == 'LCRS':
                distance = LCRS_dis(
                    sample,
                    traj,
                    node_edge_keys,
                    true_point,
                    longest_traj_len,
                )
            elif _worker_distance_type == 'NetERP':
                distance = NetERP_dis(sample, traj, distance_matrix, hot_node_id)
            else:
                raise ValueError(f"Unknown distance type: {_worker_distance_type}.")
            all_dis_list[sample_idx, traj_idx] = distance

    output_file = _batch_file_path(_worker_output_root, _worker_split, batch_end)
    np.save(output_file, all_dis_list)
    print(f'complete: {_worker_split} [{batch_start}:{batch_end})')
    return output_file


def batch_similarity_ground_truth(
        valiortest=None, all_node_list_int=None, batch_size=50,
        workers=19, overwrite=False):
    global _worker_trajectories

    if all_node_list_int is None:
        all_node_list_int = np.load(str(config.shuffle_node_file), allow_pickle=True)
    split_start, split_end = get_split_bounds(len(all_node_list_int), valiortest)
    node_list_int = all_node_list_int[split_start:split_end]
    sample_len = len(node_list_int)
    if sample_len == 0:
        raise ValueError(f"Split {valiortest} contains no trajectories.")

    output_root = get_similarity_output_root()
    batch_dir = os.path.join(output_root, f'{valiortest}_batch')
    os.makedirs(batch_dir, exist_ok=True)
    ranges = _batch_ranges(sample_len, batch_size)
    pending = []
    for batch_start, batch_end in ranges:
        output_file = _batch_file_path(output_root, valiortest, batch_end)
        row_count = batch_end - batch_start
        if overwrite or not _batch_file_is_complete(output_file, row_count, sample_len):
            pending.append((batch_start, batch_end))

    print(
        f'{valiortest} spatial split: [{split_start}:{split_end}) count={sample_len}, '
        f'pending_batches={len(pending)}/{len(ranges)}'
    )
    if not pending:
        return sample_len

    workers = max(1, int(workers))
    _worker_trajectories = node_list_int
    initializer_args = (
        str(config.shuffle_node_file),
        split_start,
        split_end,
        output_root,
        valiortest,
        str(config.distance_type),
        node_edge_keys,
        longest_traj_len,
        hot_node_id,
    )
    try:
        if workers == 1:
            _initialize_similarity_worker(*initializer_args)
            for batch_start, batch_end in pending:
                Traj_distance(batch_start, batch_end)
        else:
            with Pool(
                    processes=workers,
                    initializer=_initialize_similarity_worker,
                    initargs=initializer_args) as pool:
                results = [
                    pool.apply_async(Traj_distance, (batch_start, batch_end))
                    for batch_start, batch_end in pending
                ]
                for result in results:
                    result.get()
    finally:
        _worker_trajectories = None
    return sample_len


def merge_similarity_ground_truth(sample_len, valiortest, batch_size=50):
    output_root = get_similarity_output_root()
    output_size_tag = config.similarity_output_size_tag(sample_len)
    output_file = os.path.join(
        output_root,
        f'{valiortest}_spatial_distance_{output_size_tag}.npy',
    )
    os.makedirs(output_root, exist_ok=True)
    ranges = _batch_ranges(sample_len, batch_size)
    for batch_start, batch_end in ranges:
        batch_file = _batch_file_path(output_root, valiortest, batch_end)
        expected_shape = (batch_end - batch_start, sample_len)
        if not _batch_file_is_complete(batch_file, *expected_shape):
            raise FileNotFoundError(
                f"Missing or incomplete batch {batch_file}; expected shape {expected_shape}."
            )

    merged = np.lib.format.open_memmap(
        output_file,
        mode='w+',
        dtype=np.float32,
        shape=(sample_len, sample_len),
    )
    for batch_start, batch_end in ranges:
        batch_file = _batch_file_path(output_root, valiortest, batch_end)
        batch = np.load(batch_file, mmap_mode='r')
        merged[batch_start:batch_end] = batch
    merged.flush()
    del merged
    print(f'success merge similarity ground truth: {output_file}')
    return output_file


@numba.jit(nopython=True, fastmath=True)
def TP_dis(list_a, list_b, point_distances):
    tr1 = np.array(list_a)
    tr2 = np.array(list_b)
    M, N = len(tr1), len(tr2)
    max1 = -1
    for i in range(M):
        mindis = np.inf
        for j in range(N):
            if point_distances[tr1[i]][tr2[j]] != -1:
                temp = point_distances[tr1[i]][tr2[j]]
                if temp < mindis:
                    mindis = temp
            else:
                return -1
        if mindis != np.inf and mindis > max1:
            max1 = mindis

    max2 = -1
    for i in range(N):
        mindis = np.inf
        for j in range(M):
            if point_distances[tr2[i]][tr1[j]] != -1:
                temp = point_distances[tr2[i]][tr1[j]]
                if temp < mindis:
                    mindis = temp
            else:
                return -1
        if mindis != np.inf and mindis > max2:
            max2 = mindis

    return int(max(max1, max2))


@numba.jit(nopython=True, fastmath=True)
def DITA_dis(list_a, list_b, point_distances):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    cost = np.zeros((M, N))
    tp = point_distances[tr1[0]][tr2[0]]
    if tp == -1:
        return -1
    cost[0, 0] = tp
    for i in range(1, M):
        tp = point_distances[tr1[i]][tr2[0]]
        if tp == -1:
            return -1
        cost[i, 0] = cost[i - 1, 0] + tp
    for i in range(1, N):
        tp = point_distances[tr1[0]][tr2[i]]
        if tp == -1:
            return -1
        cost[0, i] = cost[0, i - 1] + tp
    for i in range(1, M):
        for j in range(1, N):
            small = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            tp = point_distances[tr1[i]][tr2[j]]
            if tp == -1:
                return -1
            cost[i, j] = min(small) + tp
    return int(cost[M - 1, N - 1])


@numba.jit(nopython=True, fastmath=True)
def frechet_dis(list_a, list_b, point_distances):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    c = np.zeros((M + 1, N + 1))
    tp = point_distances[tr1[0]][tr2[0]]
    if tp == -1:
        return -1
    c[0, 0] = tp
    for i in range(1, M):
        tp = point_distances[tr1[i]][tr2[0]]
        if tp == -1:
            return -1
        temp = tp
        if temp > c[i - 1][0]:
            c[i][0] = temp
        else:
            c[i][0] = c[i - 1][0]
    for i in range(1, N):
        tp = point_distances[tr1[0]][tr2[i]]
        if tp == -1:
            return -1
        temp = tp
        if temp > c[0][i - 1]:
            c[0][i] = temp
        else:
            c[0][i] = c[0][i - 1]
    for i in range(1, M):
        for j in range(1, N):
            tp = point_distances[tr1[i]][tr2[j]]
            if tp == -1:
                return -1
            c[i, j] = max(tp, min(c[i - 1][j - 1], c[i - 1][j], c[i][j - 1]))

    return int(c[M - 1, N - 1])


@numba.jit(nopython=True, fastmath=True)
def LCRS_dis(list_a, list_b, edge_keys, node_count, max_trajectory_length):
    lena = len(list_a)
    lenb = len(list_b)
    c = np.zeros((lena + 1, lenb + 1))
    for i in range(lena):
        for j in range(lenb):
            edge_key = int(list_a[i]) * node_count + int(list_b[j])
            edge_idx = np.searchsorted(edge_keys, edge_key)
            if edge_idx < len(edge_keys) and edge_keys[edge_idx] == edge_key:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    if c[-1][-1] == 0:
        return max_trajectory_length * 2
    else:
        return (lena + lenb - c[-1][-1]) / float(c[-1][-1])


def hot_node(point_distances):
    max_num = 0
    max_idx = 0
    for idx, nodes_interaction in enumerate(point_distances):
        nodes_interaction = np.array(nodes_interaction)
        x = len(nodes_interaction[nodes_interaction != -1])
        if x > max_num:
            max_num = x
            max_idx = idx
    print(max_num, max_idx)
    return max_idx


@numba.jit(nopython=True, fastmath=True)
def NetERP_dis(list_a, list_b, point_distances, anchor_node_id):
    lena = len(list_a)
    lenb = len(list_b)

    edit = np.zeros((lena + 1, lenb + 1))
    for i in range(1, lena + 1):
        tp = point_distances[anchor_node_id][list_a[i - 1]]
        if tp == -1:
            return -1
        edit[i][0] = edit[i - 1][0] + tp
    for i in range(1, lenb + 1):
        tp = point_distances[anchor_node_id][list_b[i - 1]]
        if tp == -1:
            return -1
        edit[0][i] = edit[0][i - 1] + tp

    for i in range(1, lena + 1):
        for j in range(1, lenb + 1):
            tp1 = point_distances[anchor_node_id][list_a[i - 1]]
            tp2 = point_distances[anchor_node_id][list_b[j - 1]]
            tp3 = point_distances[list_a[i - 1]][list_b[j - 1]]
            if tp1 == -1 or tp2 == -1 or tp3 == -1:
                return -1
            edit[i][j] = min(edit[i - 1][j] + tp1, edit[i][j - 1] + tp2, edit[i - 1][j - 1] + tp3)

    return edit[-1][-1]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute standard, 50w, or full-dataset ground-truth trajectory similarity matrices.'
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--dataset-mode',
        choices=['standard', '50w', 'full'],
        default=None,
        help='Select the 5w sample, fixed 50w sample, or every valid trajectory.',
    )
    mode_group.add_argument(
        '--full-dataset',
        action='store_true',
        help=(
            'Read data/<dataset>/st_traj_full, use ratio-based splits, and write '
            'results under ground_truth_full/<dataset>/<distance_type>. '
            'Alias for --dataset-mode full.'
        ),
    )
    parser.add_argument(
        '--split',
        choices=['all', 'train', 'vali', 'test'],
        default='all',
        help='Compute one split or all train/validation/test splits.',
    )
    parser.add_argument('--workers', type=int, default=19, help='Number of similarity worker processes.')
    parser.add_argument('--batch-size', type=int, default=50, help='Rows stored in each batch file.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Recompute complete batch files instead of resuming from them.',
    )
    return parser.parse_args()


def prepare_distance_resources(all_trajectories, workers=30):
    global roadnetwork, distance_matrix, hot_node_id, node_edge_keys, longest_traj_len

    if config.distance_type == 'LCRS':
        longest_traj_len = max((len(traj) for traj in all_trajectories), default=0)
        node_edge_keys = generate_node_edge_interation(true_point)
        print(f'LCRS sparse adjacency entries: {len(node_edge_keys)}')
        return

    if not os.path.exists(point_matrix_path()):
        roadnetwork = network_data()
        batch_Point_distance(workers=workers)
        merge_Point_distance()

    if config.distance_type == 'NetERP':
        distance_matrix = generate_point_matrix()
        hot_node_id = hot_node(distance_matrix)


def main():
    args = parse_args()
    if args.full_dataset:
        config.set_dataset_mode('full')
    elif args.dataset_mode is not None:
        config.set_dataset_mode(args.dataset_mode)
    if args.workers <= 0:
        raise ValueError(f'workers must be positive, got {args.workers}.')
    if args.batch_size <= 0:
        raise ValueError(f'batch-size must be positive, got {args.batch_size}.')
    if not os.path.exists(str(config.shuffle_node_file)):
        raise FileNotFoundError(
            f'Missing trajectory file: {config.shuffle_node_file}. '
            'Run spatial_preprocess.py with the matching dataset mode first.'
        )

    print('dataset:{},distance_type:{}'.format(dataset, str(config.distance_type)))
    print(f'mode:{config.dataset_mode}, matrix size:{dataset_point}, num point:{true_point}')
    print(f'trajectory input:{config.shuffle_node_file}')
    print(f'similarity output:{get_similarity_output_root()}')

    all_trajectories = np.load(str(config.shuffle_node_file), allow_pickle=True)
    prepare_distance_resources(all_trajectories, workers=args.workers)

    splits = ['vali', 'train', 'test'] if args.split == 'all' else [args.split]
    for split in splits:
        split_start, split_end = get_split_bounds(len(all_trajectories), split)
        split_size = split_end - split_start
        matrix_gib = split_size * split_size * np.dtype(np.float32).itemsize / (1024 ** 3)
        print(
            f'{split} matrix shape=({split_size}, {split_size}), '
            f'final_size={matrix_gib:.2f} GiB (batch files require similar additional space)'
        )
        sample_len = batch_similarity_ground_truth(
            valiortest=split,
            all_node_list_int=all_trajectories,
            batch_size=args.batch_size,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        merge_similarity_ground_truth(
            sample_len=sample_len,
            valiortest=split,
            batch_size=args.batch_size,
        )

    print('successful!!!')


if __name__ == '__main__':
    main()
