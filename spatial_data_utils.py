import numpy as np
from setting import SetParameter
import random
import os.path as osp
import torch
from torch_geometric.data import Data
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging
from graph_normalization import distance_weighted_gcn_norm
random.seed(1933)
np.random.seed(1933)
config = SetParameter()


def sample_region_node(dataset):
    if dataset == 'beijing':
        node_file = str(config.node_file)
        df_node = pd.read_csv(node_file, sep=',')
        all_node, all_lng, all_lat = df_node.node, df_node.lng, df_node.lat
        all_node = np.array(all_node)
        all_lng = np.array(all_lng)
        all_lat = np.array(all_lat)
        point_dis = np.load('./ground_truth/{}/Point_dis_matrix.npy'.format(dataset))
        cnt_matrix = np.zeros((7, 8))
        region_node = [[[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []],
                       [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []],
                       [[], [], [], [], [], [], [], []]]
        for i in range(112557):
            if 115.5 < all_lng[i] < 117.5 and 39 < all_lat[i] < 40.75:
                node_id = all_node[i]
                node_lng = int((all_lng[i] - 115.5) / 0.25)
                node_lat = int((all_lat[i] - 39) / 0.25)
                region_node[node_lat][node_lng].append(node_id)
                cnt_matrix[node_lat][node_lng] += 1

        selected_node_set = []
        for i in range(7):
            for j in range(8):
                node_list = region_node[i][j]
                if len(node_list) == 1:
                    selected_node_set.append(node_list[0])
                elif len(node_list) == 2:
                    selected_node_set.append(node_list[0])
                    selected_node_set.append(node_list[1])
                elif len(node_list) > 2:
                    node_ids = list(np.random.choice(len(node_list), 2, replace=False))
                    selected_node_set.append(node_ids[0])
                    selected_node_set.append(node_ids[1])

        all_distance_to_node = []
        for item in selected_node_set:
            tmp = point_dis[:112557, item]
            ids = np.where(tmp != -1)
            for idx in ids:
                tmp[idx] = np.exp(-(tmp[idx]/10000.0))
            all_distance_to_node.append(tmp)

        all_distance_to_node = np.array(all_distance_to_node).T
        distance_to_anchor_node = torch.tensor(all_distance_to_node, dtype=torch.float)
        print("distance to anchor nodes shape: ", distance_to_anchor_node.shape)
        torch.save(distance_to_anchor_node, './dataset/'+dataset+'/distance_to_anchor_node.pt')
        return distance_to_anchor_node

    elif dataset == 'tdrive':
        node_file = str(config.node_file)
        df_node = pd.read_csv(node_file, sep=',')
        all_node, all_lng, all_lat = df_node.node, df_node.lng, df_node.lat
        all_node = np.array(all_node)
        all_lng = np.array(all_lng)
        all_lat = np.array(all_lat)
        point_dis = np.load('./ground_truth/{}/Point_dis_matrix.npy'.format(dataset))

        cnt_matrix = np.zeros((8, 7))
        region_node = [[[], [], [], [], [], [], []], [[], [], [], [], [], [], []], [[], [], [], [], [], [], []],
                       [[], [], [], [], [], [], []], [[], [], [], [], [], [], []], [[], [], [], [], [], [], []],
                       [[], [], [], [], [], [], []], [[], [], [], [], [], [], []]]
        for i in range(74671):
            if 116.1 < all_lng[i] < 116.8 and 39.5 < all_lat[i] < 40.3:
                node_id = all_node[i]
                node_lng = int((all_lng[i] - 116.1) / 0.1)
                node_lat = int((all_lat[i] - 39.5) / 0.1)
                region_node[node_lat][node_lng].append(node_id)
                cnt_matrix[node_lat][node_lng] += 1

        selected_node_set = []
        for i in range(8):
            for j in range(7):
                node_list = region_node[i][j]
                if len(node_list) == 1:
                    selected_node_set.append(node_list[0])
                elif len(node_list) == 2:
                    selected_node_set.append(node_list[0])
                    selected_node_set.append(node_list[1])
                elif len(node_list) > 2:
                    node_ids = list(np.random.choice(len(node_list), 2, replace=False))
                    selected_node_set.append(node_ids[0])
                    selected_node_set.append(node_ids[1])

        all_distance_to_node = []
        for item in selected_node_set:
            tmp = point_dis[:74671, item]
            ids = np.where(tmp != -1)
            for idx in ids:
                tmp[idx] = np.exp(-(tmp[idx] / 100.0))
            all_distance_to_node.append(tmp)

        all_distance_to_node = np.array(all_distance_to_node).T
        distance_to_anchor_node = torch.tensor(all_distance_to_node, dtype=torch.float)
        print("distance to anchor nodes shape: ", distance_to_anchor_node.shape)
        torch.save(distance_to_anchor_node, './dataset/'+dataset+'/distance_to_anchor_node.pt')
        return distance_to_anchor_node

    elif dataset == 'porto':
        node_file = str(config.node_file)
        df_node = pd.read_csv(node_file, sep=',')
        all_node, all_lng, all_lat = df_node.node, df_node.lng, df_node.lat
        all_node = np.array(all_node)
        all_lng = np.array(all_lng)
        all_lat = np.array(all_lat)
        point_dis = np.load('./ground_truth/{}/Point_dis_matrix.npy'.format(dataset))
        cnt_matrix = np.zeros((6, 6))
        region_node = [[[], [], [], [], [], []], [[], [], [], [], [], []], [[], [], [], [], [], []],
                       [[], [], [], [], [], [], []], [[], [], [], [], [], []], [[], [], [], [], [], []]]
        for i in range(128466):
            if -8.8 < all_lng[i] < -8.2 and 40.9 < all_lat[i] < 41.5:
                node_id = all_node[i]
                node_lng = int((all_lng[i] + 8.8) / 0.1)
                node_lat = int((all_lat[i] - 40.9) / 0.1)
                region_node[node_lat][node_lng].append(node_id)
                cnt_matrix[node_lat][node_lng] += 1

        selected_node_set = []
        for i in range(6):
            for j in range(6):
                node_list = region_node[i][j]
                if len(node_list) == 1:
                    if np.sum(point_dis[:128466, node_list[0]]) != -128465:
                        selected_node_set.append(node_list[0])
                elif len(node_list) == 2:
                    if np.sum(point_dis[:128466, node_list[0]]) != -128465:
                        selected_node_set.append(node_list[0])
                    if np.sum(point_dis[:128466, node_list[1]]) != -128465:
                        selected_node_set.append(node_list[1])
                elif len(node_list) > 2:
                    node_ids = list(np.random.choice(len(node_list), 2, replace=False))
                    flag = True
                    while flag:
                        if np.sum(point_dis[:128466, node_ids[0]]) != -128465 and np.sum(
                                point_dis[:128466, node_ids[1]]) != -128465:
                            selected_node_set.append(node_ids[0])
                            selected_node_set.append(node_ids[1])
                            flag = False
                        else:
                            node_ids = list(np.random.choice(len(node_list), 2, replace=False))

        all_distance_to_node = []
        for item in selected_node_set:
            tmp = point_dis[:128466, item]
            ids = np.where(tmp != -1)
            for idx in ids:
                tmp[idx] = np.exp(-(tmp[idx] / 10000.0))
            all_distance_to_node.append(tmp)

        all_distance_to_node = np.array(all_distance_to_node).T
        distance_to_anchor_node = torch.tensor(all_distance_to_node, dtype=torch.float)
        print("distance to anchor nodes shape: ", distance_to_anchor_node.shape)
        torch.save(distance_to_anchor_node, './dataset/'+dataset+'/distance_to_anchor_node.pt')
        return distance_to_anchor_node


class TrainData(Dataset):
    def __init__(self, data, coor, label, neg_label, dis, neg_dis):
        self.data = data
        self.coor = coor
        self.label = label
        self.neg_label = neg_label
        self.dis = dis
        self.neg_dis = neg_dis

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.coor[idx], self.label[idx], self.neg_label[idx], self.dis[idx], self.neg_dis[idx], idx


def load_traindata(train_file):
    data = np.load(train_file, allow_pickle=True)
    x_data = data["train_list"]
    coor = data["train_coor"]

    x, y = [], []
    for traj in coor:
        for r in traj:
            x.append(r[0])
            y.append(r[1])
    meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
    coor = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
             for r in t] for t in coor]

    y_idx = data["train_y"]
    y_neg_idx = data["train_neg_y"]
    y_dis = data["train_dis"]
    y_neg_dis = data["train_neg_dis"]
    logging.info(f'coe: {data["coe"]}')
    return x_data, coor, y_idx, y_neg_idx,  y_dis, y_neg_dis


def train_data_loader(train_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_aco = []
        coor_aco = []
        data_pos = []
        coor_pos = []
        data_neg = []
        coor_neg = []
        data_pos_dis = []
        data_neg_dis = []
        for i, (data, coor, label, neg_label, dis, neg_dis, idx) in (enumerate(data_tuple)):

            for j in range(len(label)):
                data_aco.append(torch.LongTensor(data))
                coor_aco.append(torch.tensor(coor, dtype=torch.float32))

                data_pos.append(torch.LongTensor(train_x[label[j]]))
                coor_pos.append(torch.tensor(coor_x[label[j]], dtype=torch.float32))
                data_pos_dis.append(dis[j])

                neg_idx_random = np.random.randint(len(neg_label))
                neg_idx = neg_label[neg_idx_random]
                data_neg.append(torch.LongTensor(train_x[neg_idx]))
                coor_neg.append(torch.tensor(coor_x[neg_idx], dtype=torch.float32))
                data_neg_dis.append(neg_dis[neg_idx_random])

        data_pos_dis = torch.tensor(data_pos_dis)
        data_neg_dis = torch.tensor(data_neg_dis)
        aco_length = torch.tensor(list(map(len, data_aco)))
        pos_length = torch.tensor(list(map(len, data_pos)))
        neg_length = torch.tensor(list(map(len, data_neg)))

        data_aco = rnn_utils.pad_sequence(data_aco, batch_first=True, padding_value=0)
        data_pos = rnn_utils.pad_sequence(data_pos, batch_first=True, padding_value=0)
        data_neg = rnn_utils.pad_sequence(data_neg, batch_first=True, padding_value=0)
        coor_aco = rnn_utils.pad_sequence(coor_aco, batch_first=True, padding_value=0)
        coor_pos = rnn_utils.pad_sequence(coor_pos, batch_first=True, padding_value=0)
        coor_neg = rnn_utils.pad_sequence(coor_neg, batch_first=True, padding_value=0)

        return (data_aco, coor_aco, data_pos, coor_pos, data_neg, coor_neg,
                data_pos_dis, data_neg_dis,
                aco_length, pos_length, neg_length)

    train_x, coor_x, train_y, train_neg_y, train_dis, train_neg_dis = load_traindata(train_file)
    logging.info(f'train size:{len(train_x)}')
    data_ = TrainData(train_x, coor_x, train_y, train_neg_y, train_dis, train_neg_dis)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    return dataset


def load_validata(vali_file):
    data = np.load(vali_file, allow_pickle=True)
    x_data = data["vali_list"]
    coor = data["vali_coor"]
    y_idx = data["vali_y"]

    x, y = [], []
    for traj in coor:
        for r in traj:
            x.append(r[0])
            y.append(r[1])
    meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
    coor = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
             for r in t] for t in coor]

    return x_data, coor, y_idx


class ValiData(Dataset):
    def __init__(self, data, coor, label):
        self.data = data
        self.coor = coor
        self.label = label
        self.index = list(range(len(data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tuple_ = (self.data[idx], self.coor[idx], self.label[idx], self.index[idx])
        return tuple_


def vali_data_loader(vali_file, batchsize):
    def collate_fn_neg(data_tuple):
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        coor = [torch.tensor(sq[1], dtype=torch.float32) for sq in data_tuple]
        label = [sq[2] for sq in data_tuple]
        idx = [sq[3] for sq in data_tuple]
        data_length = torch.tensor(list(map(len, data)))
        label = torch.tensor(np.array(label))
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        coor = rnn_utils.pad_sequence(coor, batch_first=True, padding_value=0)
        return data, coor, label, data_length, idx

    val_x, coor_x, val_y = load_validata(vali_file)
    data_ = ValiData(val_x, coor_x, val_y)
    dataset = DataLoader(
        data_,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate_fn_neg,
        drop_last=False,
    )

    return dataset, len(val_x)


def load_region_node(dataset):
    region_node = torch.load('./dataset/'+dataset+'/distance_to_anchor_node.pt')
    return region_node


def load_netowrk(dataset):
    edge_path = str(config.edge_file)
    node_embedding_path = "./data/" + dataset + "/node_features.npy"

    node_embeddings = np.load(node_embedding_path)
    df_dege = pd.read_csv(edge_path, sep=',')

    edge_index = df_dege[["s_node", "e_node"]].to_numpy()
    edge_attr = df_dege["length"].to_numpy()
    if str(config.dataset) == "beijing" or "porto":
        edge_attr = edge_attr / 100.0

    edge_index = torch.LongTensor(edge_index).t().contiguous()
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    logging.info(f"node embeddings shape: {node_embeddings.shape}")
    logging.info(f"edge_index shap: {edge_index.shape}")
    logging.info(f"edge_attr shape: {edge_attr.shape}")

    road_network = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)

    return road_network


def _load_hierarchy_array(hierarchy_dir, filename):
    path = osp.join(hierarchy_dir, filename)
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"Missing hierarchy file: {path}. "
            "Run generate_louvain_hierarchy.py before loading the hierarchy."
        )
    return np.load(path)


def _validate_node_assignment(assignment, name, num_nodes):
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {assignment.shape}.")
    if assignment.shape[0] != num_nodes:
        raise ValueError(
            f"{name} contains {assignment.shape[0]} nodes, but node_features.npy "
            f"contains {num_nodes} nodes. Regenerate the hierarchy and node features "
            "from the same road-network data."
        )
    if np.any(assignment < 0):
        missing_count = int(np.count_nonzero(assignment < 0))
        raise ValueError(f"{name} contains {missing_count} unassigned nodes.")

    region_count = int(assignment.max()) + 1 if assignment.size else 0
    if region_count == 0:
        raise ValueError(f"{name} does not contain any regions.")
    present = np.bincount(assignment, minlength=region_count) > 0
    if not np.all(present):
        missing_regions = np.flatnonzero(~present).tolist()
        raise ValueError(f"{name} has non-contiguous region IDs; missing {missing_regions[:10]}.")
    return assignment


def _aggregate_region_features(node_features, assignment):
    region_count = int(assignment.max()) + 1
    region_features = np.zeros(
        (region_count, node_features.shape[1]),
        dtype=np.float32,
    )
    np.add.at(region_features, assignment, node_features)
    counts = np.bincount(assignment, minlength=region_count).astype(np.float32)
    region_features /= counts[:, None]
    return region_features


def _derive_parent_assignment(child_assignment, parent_assignment, child_name, parent_name):
    child_count = int(child_assignment.max()) + 1
    min_parent = np.full(child_count, np.iinfo(np.int64).max, dtype=np.int64)
    max_parent = np.full(child_count, -1, dtype=np.int64)
    np.minimum.at(min_parent, child_assignment, parent_assignment)
    np.maximum.at(max_parent, child_assignment, parent_assignment)

    inconsistent = np.flatnonzero(min_parent != max_parent)
    if inconsistent.size:
        raise ValueError(
            f"Hierarchy is not nested: {child_name} regions "
            f"{inconsistent[:10].tolist()} map to multiple {parent_name} regions."
        )
    return min_parent


def _as_edge_index(edges, name):
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if edges.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional edge array, got {edges.shape}.")
    if edges.shape[1] == 2:
        edges = edges.T
    elif edges.shape[0] != 2:
        raise ValueError(f"{name} must have shape [E, 2] or [2, E], got {edges.shape}.")
    return torch.as_tensor(edges, dtype=torch.long).contiguous()


def _build_hierarchy_graph(features, edges, weights, level_name, transition_features=None):
    edge_index = _as_edge_index(edges, f"{level_name}_edge_index")
    edge_attr = torch.as_tensor(np.asarray(weights), dtype=torch.float).reshape(-1)
    if edge_index.shape[1] != edge_attr.numel():
        raise ValueError(
            f"{level_name} has {edge_index.shape[1]} edges but "
            f"{edge_attr.numel()} edge weights."
        )
    if transition_features is not None:
        transition_features = torch.as_tensor(
            np.asarray(transition_features), dtype=torch.float
        )
        if transition_features.ndim != 2 or transition_features.size(0) != edge_index.size(1):
            raise ValueError(
                f"{level_name} transition features must have shape "
                f"[E, F] with E={edge_index.size(1)}, got {tuple(transition_features.shape)}."
            )
    if edge_index.numel() and int(edge_index.max()) >= features.shape[0]:
        raise ValueError(
            f"{level_name}_edge_index references node {int(edge_index.max())}, "
            f"but the level contains only {features.shape[0]} nodes."
        )
    gcn_edge_index, gcn_edge_norm, affinity_scale = distance_weighted_gcn_norm(
        edge_index,
        edge_attr,
        features.shape[0],
    )
    graph = Data(
        x=torch.as_tensor(features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        gcn_edge_index=gcn_edge_index,
        gcn_edge_norm=gcn_edge_norm,
        gcn_affinity_scale=affinity_scale,
    )
    if transition_features is not None:
        graph.transition_attr = transition_features
    return graph


def load_hierarchy_multilevel(dataset):
    hierarchy_dir = osp.join("dataset", dataset, "hierarchy_louvain")
    node_embedding_path = osp.join("data", dataset, "node_features.npy")
    edge_path = osp.join("dataset", dataset, "edge_weight.csv")

    if not osp.isfile(node_embedding_path):
        raise FileNotFoundError(
            f"Missing node features: {node_embedding_path}. Run spatial_preprocess.py first."
        )
    if not osp.isfile(edge_path):
        raise FileNotFoundError(f"Missing road-network edge file: {edge_path}.")

    node_features = np.asarray(np.load(node_embedding_path), dtype=np.float32)
    if node_features.ndim != 2:
        raise ValueError(
            f"node_features.npy must be two-dimensional, got shape {node_features.shape}."
        )

    num_nodes = node_features.shape[0]
    node_to_l1 = _validate_node_assignment(
        _load_hierarchy_array(hierarchy_dir, "node_to_l1.npy"),
        "node_to_l1",
        num_nodes,
    )
    node_to_l2 = _validate_node_assignment(
        _load_hierarchy_array(hierarchy_dir, "node_to_l2.npy"),
        "node_to_l2",
        num_nodes,
    )
    node_to_l3 = _validate_node_assignment(
        _load_hierarchy_array(hierarchy_dir, "node_to_l3.npy"),
        "node_to_l3",
        num_nodes,
    )

    l1_to_l2 = _derive_parent_assignment(node_to_l1, node_to_l2, "l1", "l2")
    l2_to_l3 = _derive_parent_assignment(node_to_l2, node_to_l3, "l2", "l3")

    level_features = [
        node_features,
        _aggregate_region_features(node_features, node_to_l1),
        _aggregate_region_features(node_features, node_to_l2),
        _aggregate_region_features(node_features, node_to_l3),
    ]

    edge_df = pd.read_csv(edge_path, sep=",")
    required_columns = {"s_node", "e_node", "length"}
    missing_columns = required_columns.difference(edge_df.columns)
    if missing_columns:
        raise ValueError(f"{edge_path} is missing columns: {sorted(missing_columns)}.")
    level_edges = [edge_df[["s_node", "e_node"]].to_numpy(dtype=np.int64)]
    level_weights = [edge_df["length"].to_numpy(dtype=np.float32)]
    for level_name in ("l1", "l2", "l3"):
        level_edges.append(_load_hierarchy_array(hierarchy_dir, f"{level_name}_edge_index.npy"))
        level_weights.append(_load_hierarchy_array(hierarchy_dir, f"{level_name}_edge_weight.npy"))

    if dataset in {"beijing", "porto"}:
        level_weights = [weights / 100.0 for weights in level_weights]

    transition_features = []
    for level_name in ("l0", "l1", "l2", "l3"):
        transition_path = osp.join(hierarchy_dir, f"{level_name}_transition_features.npy")
        transition_features.append(
            _load_hierarchy_array(hierarchy_dir, f"{level_name}_transition_features.npy")
            if osp.isfile(transition_path) else None
        )
    if any(item is None for item in transition_features):
        logging.warning(
            "Transition edge features are unavailable for %s; using length-only graph messages. "
            "Regenerate hierarchy_louvain to enable the flow channel.",
            dataset,
        )

    level_graphs = [
        _build_hierarchy_graph(features, edges, weights, level_name, transition_attr)
        for features, edges, weights, level_name, transition_attr in zip(
            level_features,
            level_edges,
            level_weights,
            ("l0", "l1", "l2", "l3"),
            transition_features,
        )
    ]

    logging.info(
        "hierarchy graph sizes: %s",
        [int(graph.x.shape[0]) for graph in level_graphs],
    )
    logging.info(
        "distance-affinity scales: %s",
        [float(graph.gcn_affinity_scale) for graph in level_graphs],
    )
    return {
        "level_graphs": level_graphs,
        "node_to_l1": torch.as_tensor(node_to_l1, dtype=torch.long),
        "node_to_l2": torch.as_tensor(node_to_l2, dtype=torch.long),
        "node_to_l3": torch.as_tensor(node_to_l3, dtype=torch.long),
        "l1_to_l2": torch.as_tensor(l1_to_l2, dtype=torch.long),
        "l2_to_l3": torch.as_tensor(l2_to_l3, dtype=torch.long),
    }


def load_neighbor(dataset, knn):
    k5_neighbor = np.load(f'./dataset/{dataset}/{str(knn)}/k{str(knn)}_neighbor.npy')
    k5_distance = np.load(f'./dataset/{dataset}/{str(knn)}/k{str(knn)}_distance.npy')
    k10_neighbor = np.load(f'./dataset/{dataset}/{str(knn)}/k{str(knn*2)}_neighbor.npy')
    k10_distance = np.load(f'./dataset/{dataset}/{str(knn)}/k{str(knn*2)}_distance.npy')
    k15_neighbor = np.load(f'./dataset/{dataset}/{str(knn)}/k{str(knn*3)}_neighbor.npy')
    k15_distance = np.load(f'./dataset/{dataset}/{str(knn)}/k{str(knn*3)}_distance.npy')
    node_embedding_path = "./data/" + dataset + "/node_features.npy"
    degree_encodings = np.load(node_embedding_path)

    edge_index_l0 = torch.LongTensor(k5_neighbor).t().contiguous()
    edge_attr_l0 = torch.tensor(k5_distance, dtype=torch.float)
    edge_index_l1 = torch.LongTensor(k10_neighbor).t().contiguous()
    edge_attr_l1 = torch.tensor(k10_distance, dtype=torch.float)
    edge_index_l2 = torch.LongTensor(k15_neighbor).t().contiguous()
    edge_attr_l2 = torch.tensor(k15_distance, dtype=torch.float)
    node_embeddings = torch.tensor(degree_encodings, dtype=torch.float)

    logging.info(f"node embeddings shape: {node_embeddings.shape}")
    logging.info(f"edge_index_l0 shape: {edge_index_l0.shape}")
    logging.info(f"edge_attr_l0 shape: {edge_attr_l0.shape}")
    logging.info(f"edge_index_l1 shape: {edge_index_l1.shape}")
    logging.info(f"edge_attr_l1 shape: {edge_attr_l1.shape}")
    logging.info(f"edge_index_l2 shape: {edge_index_l2.shape}")
    logging.info(f"edge_attr_l2 shape: {edge_attr_l2.shape}")

    edge_index_l0 = edge_index_l0[[1, 0], :]
    edge_index_l1 = edge_index_l1[[1, 0], :]
    edge_index_l2 = edge_index_l2[[1, 0], :]

    road_network_l0 = Data(x=node_embeddings, edge_index=edge_index_l0, edge_attr=edge_attr_l0)
    road_network_l1 = Data(x=[], edge_index=edge_index_l1, edge_attr=edge_attr_l1)
    road_network_l2 = Data(x=[], edge_index=edge_index_l2, edge_attr=edge_attr_l2)

    return [road_network_l0, road_network_l1, road_network_l2]


if __name__ == "__main__":
    sample_region_node(config.dataset)
