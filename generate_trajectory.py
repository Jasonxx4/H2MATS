import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from setting import SetParameter
import random
config = SetParameter()
random.seed(1933)
np.random.seed(1933)
if config.dataset == 'beijing':
    point_num = 112557  # tdrive: 74671; beijing: 112557; porto: 128466
    matrix_num = 113000  # tdrive: 75000; beijing: 113000; porto: 129000
    LCRS_num = 7250
elif config.dataset == 'porto':
    point_num = 128466
    matrix_num = 129000
    LCRS_num = 3286


def re_matrix(traj_list, coor_list, input_dis_matrix, l):
    idx = []
    for i in tqdm(range(len(input_dis_matrix))):
        if np.sum(input_dis_matrix[i] >= 0) > l:
            idx.append(i)
    idx = np.array(idx)
    out_traj_list = traj_list[idx]
    out_coor_list = coor_list[idx]
    out_dis_matrix = input_dis_matrix[idx.reshape(-1, 1), idx.reshape(1, -1)]

    return out_traj_list, out_coor_list, out_dis_matrix


def get_label(input_dis_matrix, count):
    label = []
    for i in tqdm(range(len(input_dis_matrix))):
        input_r = np.array(input_dis_matrix[i])
        idx = np.argsort(input_r)
        val = input_r[idx]
        idx = idx[val != -1]
        if len(idx) < 51:
            print(idx)
        label.append(idx[1:count+1])
    return np.array(label)


def get_train_label(input_dis_matrix, count):
    label = []
    neg_label = []

    label_dis = []
    neg_label_dis = []
    for i in tqdm(range(len(input_dis_matrix))):
        input_r = np.array(input_dis_matrix[i])
        idx = np.argsort(input_r)
        val = input_r[idx]
        re_idx = idx[val != -1]
        re_val = val[val != -1]

        label.append(re_idx[1:count+1])
        label_dis.append(re_val[1:count+1])
        neg_label.append(re_idx[count+1:])
        neg_label_dis.append(re_val[count+1:])

    label = np.array(label, dtype=object)
    neg_label = np.array(neg_label, dtype=object)
    label_dis = np.array(label_dis, dtype=object)
    neg_label_dis = np.array(neg_label_dis, dtype=object)
    return label, neg_label, label_dis, neg_label_dis


def distance_matrix_path(split, split_size):
    size_tag = config.similarity_output_size_tag(split_size)
    return osp.join(config.ground_truth_dir, f'{split}_spatial_distance_{size_tag}.npy')


def valid_distance_mask(row):
    mask = row >= 0
    if config.distance_type == 'LCRS':
        mask &= row != LCRS_num
    return mask


def eligible_indices(matrix, minimum_neighbors, split):
    eligible = []
    for row_idx in tqdm(range(len(matrix)), desc=f'filter {split}'):
        if np.count_nonzero(valid_distance_mask(np.asarray(matrix[row_idx]))) > minimum_neighbors:
            eligible.append(row_idx)
    return np.asarray(eligible, dtype=np.int64)


def nearest_positions(row, self_position, count):
    valid = valid_distance_mask(row)
    valid[self_position] = False
    candidates = np.flatnonzero(valid)
    if len(candidates) < count:
        raise ValueError(
            f'Only {len(candidates)} valid candidates remain; {count} are required.'
        )
    candidate_distances = row[candidates]
    nearest_offset = np.argpartition(candidate_distances, count - 1)[:count]
    nearest = candidates[nearest_offset]
    nearest = nearest[np.argsort(row[nearest], kind='stable')]
    return nearest, candidates


def build_large_eval_set(traj_list, coor_list, matrix, output_file, split):
    eligible = eligible_indices(matrix, 50, split)
    labels = np.empty((len(eligible), 50), dtype=np.int64)
    for new_idx, source_idx in enumerate(tqdm(eligible, desc=f'label {split}')):
        row = np.asarray(matrix[source_idx, eligible])
        labels[new_idx], _ = nearest_positions(row, new_idx, 50)

    np.savez(
        output_file,
        vali_list=traj_list[eligible],
        vali_coor=coor_list[eligible],
        vali_y=labels,
    )
    print(f'saved {split} set: {output_file}, trajectories={len(eligible)}')


def build_large_train_set(traj_list, coor_list, matrix):
    eligible = eligible_indices(matrix, config.pos_num * 2, 'train')
    sample_count = len(eligible)
    negative_count = min(
        config.large_negative_pool_size,
        max(0, sample_count - config.pos_num - 1),
    )
    if negative_count == 0:
        raise ValueError('No negative candidates remain in the 50w training split.')

    labels = np.empty((sample_count, config.pos_num), dtype=np.int64)
    label_distances = np.empty((sample_count, config.pos_num), dtype=np.float32)
    negative_labels = np.empty((sample_count, negative_count), dtype=np.int64)
    negative_distances = np.empty((sample_count, negative_count), dtype=np.float32)
    rng = np.random.default_rng(1933)
    max_distance = 0.0

    for new_idx, source_idx in enumerate(tqdm(eligible, desc='label train')):
        row = np.asarray(matrix[source_idx, eligible])
        nearest, candidates = nearest_positions(row, new_idx, config.pos_num)
        labels[new_idx] = nearest
        label_distances[new_idx] = row[nearest]

        remaining_mask = np.ones(len(row), dtype=bool)
        remaining_mask[nearest] = False
        remaining = candidates[remaining_mask[candidates]]
        if len(remaining) < negative_count:
            raise ValueError(
                f'Train row {source_idx} has only {len(remaining)} negative candidates; '
                f'{negative_count} are required by large_negative_pool_size.'
            )
        selected = rng.choice(remaining, size=negative_count, replace=False)
        negative_labels[new_idx] = selected
        negative_distances[new_idx] = row[selected]
        row_valid = row[valid_distance_mask(row)]
        if len(row_valid):
            max_distance = max(max_distance, float(np.max(row_valid)))

    if max_distance <= 0:
        raise ValueError(f'Invalid maximum training distance: {max_distance}.')
    label_distances = label_distances / max_distance * config.coe
    negative_distances = negative_distances / max_distance * config.coe

    np.savez(
        config.spatial_train_set,
        train_list=traj_list[eligible],
        train_coor=coor_list[eligible],
        train_y=labels,
        train_neg_y=negative_labels,
        train_dis=label_distances,
        train_neg_dis=negative_distances,
        coe=config.coe,
    )
    print(
        f'saved train set: {config.spatial_train_set}, trajectories={sample_count}, '
        f'negative_pool={negative_count}'
    )


def generate_500k_sets(train_list, train_coor, vali_list, vali_coor, test_list, test_coor):
    train_matrix = np.load(distance_matrix_path('train', len(train_list)), mmap_mode='r')
    vali_matrix = np.load(distance_matrix_path('vali', len(vali_list)), mmap_mode='r')
    test_matrix = np.load(distance_matrix_path('test', len(test_list)), mmap_mode='r')
    build_large_train_set(train_list, train_coor, train_matrix)
    build_large_eval_set(vali_list, vali_coor, vali_matrix, config.spatial_vali_set, 'vali')
    build_large_eval_set(test_list, test_coor, test_matrix, config.spatial_test_set, 'test')


def main():
    all_node_list_int = np.load(config.shuffle_node_file, allow_pickle=True)
    all_coor_list_int = np.load(config.shuffle_coor_file, allow_pickle=True)
    print(config.dataset)
    print(config.distance_type)
    print('dataset mode:', config.dataset_mode)

    train_start, train_end = config.get_split_bounds(len(all_node_list_int), 'train')
    vali_start, vali_end = config.get_split_bounds(len(all_node_list_int), 'vali')
    test_start, test_end = config.get_split_bounds(len(all_node_list_int), 'test')

    train_list = all_node_list_int[train_start:train_end]
    vali_list = all_node_list_int[vali_start:vali_end]
    test_list = all_node_list_int[test_start:test_end]
    train_coor = all_coor_list_int[train_start:train_end]
    vali_coor = all_coor_list_int[vali_start:vali_end]
    test_coor = all_coor_list_int[test_start:test_end]

    os.makedirs(config.ground_truth_dir, exist_ok=True)
    if config.use_500k_dataset:
        generate_500k_sets(
            train_list, train_coor, vali_list, vali_coor, test_list, test_coor
        )
        return

    train_dis_matrix = np.load(distance_matrix_path('train', len(train_list)))
    vali_dis_matrix = np.load(distance_matrix_path('vali', len(vali_list)))
    test_dis_matrix = np.load(distance_matrix_path('test', len(test_list)))

    np.fill_diagonal(train_dis_matrix, 0)
    np.fill_diagonal(vali_dis_matrix, 0)
    np.fill_diagonal(test_dis_matrix, 0)
    if config.distance_type == 'LCRS':
        train_dis_matrix[train_dis_matrix == LCRS_num] = -1
    re_train_list, re_train_coor, re_train_dis_matrix = re_matrix(
        train_list, train_coor, train_dis_matrix, config.pos_num * 2)
    re_vali_list, re_vali_coor, re_vali_dis_matrix = re_matrix(
        vali_list, vali_coor, vali_dis_matrix, 50)
    re_test_list, re_test_coor, re_test_dis_matrix = re_matrix(
        test_list, test_coor, test_dis_matrix, 50)

    norm_num = np.max(re_train_dis_matrix)
    re_train_dis_matrix = re_train_dis_matrix / norm_num * config.coe
    re_train_dis_matrix[re_train_dis_matrix < 0] = -1

    train_y, train_neg_y, train_dis, train_neg_dis = get_train_label(
        re_train_dis_matrix, config.pos_num)
    vali_y = get_label(re_vali_dis_matrix, 50)
    test_y = get_label(re_test_dis_matrix, 50)

    np.savez(config.spatial_train_set,
             train_list=re_train_list,
             train_coor=re_train_coor,
             train_y=train_y,
             train_neg_y=train_neg_y,
             train_dis=train_dis,
             train_neg_dis=train_neg_dis,
             coe=config.coe)
    np.savez(config.spatial_vali_set,
             vali_list=re_vali_list, vali_coor=re_vali_coor, vali_y=vali_y)
    np.savez(config.spatial_test_set,
             vali_list=re_test_list, vali_coor=re_test_coor, vali_y=test_y)


if __name__ == '__main__':
    main()
