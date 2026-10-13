import logging
import os.path as osp
import torch
import os


class SetParameter:
    def __init__(self):
        self.dataset = 'porto'  # beijing,porto,tdrive
        self.distance_type = 'NetERP'  # 'TP','DITA','discret_frechet','LCRS','NetERP'
        requested_mode = os.getenv('H2MATS_DATASET_MODE', '').strip().lower()
        if not requested_mode:
            requested_mode = (
                'full'
                if os.getenv('H2MATS_FULL_DATASET', '0').lower() in {'1', 'true', 'yes', 'on'}
                else 'standard'
            )
        self.dataset_mode = 'standard'
        self.use_full_dataset = False
        self.use_500k_dataset = False
        self.similarity_split = 'standard'  # train, vali, test, all, standard
        self.train_ratio = 0.3
        self.vali_ratio = 0.1
        self.test_ratio = 0.6

        self.kseg = 5
        self.pos_num = 10
        self.large_negative_pool_size = 1024

        self.node_file = osp.join('dataset', self.dataset, 'node.csv')
        self.edge_file = osp.join('dataset', self.dataset, 'edge_weight.csv')
        self.traj_file = osp.join('dataset', self.dataset, 'matching_result.pt')

        self.tdrive_traj_file = osp.join('dataset', self.dataset, 'matching_result.csv')
        self.tdrive_time_file = osp.join('dataset', self.dataset, 'time_drop_list.csv')

        self.set_dataset_mode(requested_mode)
        self.spatial_long_test_set = osp.join('ground_truth', self.dataset,
                                              self.distance_type, 'spatial_long_test_set.npz')
        self.spatial_num_test_set = osp.join('ground_truth', self.dataset,
                                             self.distance_type, 'spatial_num_test_set.npz')

        # The number of nodes in the network
        self.pointnum = {
            'beijing': 113000,
            'porto': 129000,
            'tdrive': 75000
        }
        self.truenum = {
            'beijing': 112557,
            'porto': 128466,
            'tdrive': 74671
        }

        self.alpha1 = 0.75  # 0.55,0.65,0.75,0.85,0.95
        self.alpha2 = 0.25  # 0.45,0.35,0.25,0.15,0.05

        self.num_knn = 5  # 3,4,5,6,7
        self.hierarchy_dir = osp.join('dataset', self.dataset, 'hierarchy_louvain')

        self.save_folder = None
        self.feature_size = 64  # node2vec feature size
        self.embedding_size = 64  # GNN embedding size
        self.date2vec_size = 64  # TRM hidden size
        self.hidden_size = 128
        self.num_layers = 1
        self.dropout_rate = 0
        self.learning_rate = 0.001
        self.concat = False
        self.epochs = 50
        self.early_stop = 75

        self.gtraj = {
            'train_batch': 20,
            'test_batch': 128,
            'usePE': False,
            'useSI': False,
            'useLSTM': True,
            'useHybridGeo': True,
            'graphEncoder': 'hybrid_multilevel',  # hybrid_multilevel, simple_gcn
            'hybridCurvature': 1.0,
            'hierarchyBlocks': 2,
            'hierarchyDropout': 0.1,
            'graphScaleDropout': 0.2,
            'useTransitionFeatures': True,
            'transitionResidualInit': -2.0,
            'hierarchyLossWeight': 0.05,
            'hierarchyLossWarmupEpochs': 10,
            'graphScaleBalanceWeight': 0.01,
            'graphScaleBalanceEpochs': 20,
            'seqEncoder': 'hmt',  # hmt, mamba, bimamba, transformer
            'coordEncoder': 'transformer',  #lstm, mamba, transformer
            'mambaLayers': 2,
            'mambaDState': 16,
            'mambaDConv': 4,
            'mambaExpand': 2,
            'hmtBackbone': 'hybrid',  # conv, mamba, bimamba, transformer, hybrid
            'hmtLayers': 2,
            'hmtKernelSize': 5,
            'hmtDropout': 0.1,
            'hmtTransformerHeads': 4,
            'hmtScaleAlignedGraph': True,
            'coordFusion': 'gated_attention',  # gated_attention, late_concat
            'coordTransformerCheckpoint': True,
            'useAMP': False,
            # Optional graph-pretraining path. False preserves end-to-end training.
            'usePretrainedFrozenGraph': False,
            'pretrainedGraphCheckpoint': None,
            'pretrainedGraphStrict': True,
            'cacheFrozenGraph': True,
            # Optional staged fine-tuning: freeze for N epochs, then use a lower graph LR.
            'fineTunePretrainedGraph': False,
            'pretrainedGraphFreezeEpochs': 1,
            'pretrainedGraphLrRatio': 1,
        }

        self.node2vec = {
            'walk_length': 20,
            'context_size': 10,
            'walks_per_node': 10,
            'num_neg_samples': 1,
            'p': 1,
            'q': 1
        }
        self.cuda = str('0')
        self.device = torch.device('cuda:' + self.cuda) if torch.cuda.is_available() else torch.device('cpu')

        if str(self.distance_type) == "TP":
            self.coe = 8*4
        elif str(self.distance_type) == "DITA":
            if str(self.dataset) == "beijing":
                self.coe = 32*8*2
            if str(self.dataset) == "porto":
                self.coe = 32*4
        elif str(self.distance_type) == "LCRS":
            self.coe = 4*16
        elif str(self.distance_type) == "discret_frechet":
            self.coe = 8*2

        allowed_dis = {'TP', 'DITA', 'discret_frechet', 'LCRS', 'NetERP'}
        if self.distance_type not in allowed_dis:
            raise ValueError(f"Invalid value for distance_type. Allowed values are: {allowed_dis}")
        allowed_data = {'beijing', 'porto', 'tdrive'}
        if self.dataset not in allowed_data:
            raise ValueError(f"Invalid value for distance_type. Allowed values are: {allowed_data}")

    def _set_standard_split_sizes(self):
        if self.dataset == 'tdrive':
            self.dataset_size = 30000
            self.train_set_size = 10000
            self.vali_set_size = 14000
            self.test_set_size = 30000
        elif self.dataset in {'beijing', 'porto'}:
            self.dataset_size = 50000
            self.train_set_size = 15000
            self.vali_set_size = 20000
            self.test_set_size = 50000
        else:
            raise ValueError(f'Unsupported dataset: {self.dataset}')

    def set_dataset_mode(self, mode):
        aliases = {
            'sampled': 'standard',
            '500k': '50w',
            '500000': '50w',
            'all': 'full',
        }
        mode = aliases.get(str(mode).strip().lower(), str(mode).strip().lower())
        if mode not in {'standard', '50w', 'full'}:
            raise ValueError(
                f"Invalid dataset mode: {mode}. Allowed values are: standard, 50w, full."
            )

        self.dataset_mode = mode
        self.use_full_dataset = mode == 'full'
        self.use_500k_dataset = mode == '50w'
        self.full_ground_truth_dir = osp.join('ground_truth_full', self.dataset, self.distance_type)
        self.ground_truth_500k_dir = osp.join('ground_truth_50w', self.dataset, self.distance_type)

        if self.use_500k_dataset:
            self.dataset_size = 500000
            self.train_set_size = 150000
            # These two fields are cumulative split endpoints.
            self.vali_set_size = 200000
            self.test_set_size = 500000
            data_dir_name = 'st_traj_50w'
            ground_truth_dir = self.ground_truth_500k_dir
        elif self.use_full_dataset:
            self._set_standard_split_sizes()
            data_dir_name = 'st_traj_full'
            ground_truth_dir = self.full_ground_truth_dir
        else:
            self._set_standard_split_sizes()
            data_dir_name = 'st_traj'
            ground_truth_dir = osp.join('ground_truth', self.dataset, self.distance_type)

        self.data_file = osp.join('data', self.dataset, data_dir_name)
        self.shuffle_node_file = osp.join(self.data_file, 'shuffle_node_list.npy')
        self.shuffle_coor_file = osp.join(self.data_file, 'shuffle_coor_list.npy')
        self.shuffle_kseg_file = osp.join(self.data_file, 'shuffle_kseg_list.npy')
        self.shuffle_index_file = osp.join(self.data_file, 'shuffle_index_list.npy')
        self.shuffle_time_file = osp.join(self.data_file, 'shuffle_time_list.npy')
        self.shuffle_d2vec_file = osp.join(self.data_file, 'shuffle_d2vec_list.npy')

        self.ground_truth_dir = ground_truth_dir
        self.spatial_train_set = osp.join(ground_truth_dir, 'spatial_train_set.npz')
        self.spatial_vali_set = osp.join(ground_truth_dir, 'spatial_vali_set.npz')
        self.spatial_test_set = osp.join(ground_truth_dir, 'spatial_test_set.npz')

    def set_full_dataset(self, enabled):
        self.set_dataset_mode('full' if enabled else 'standard')

    def set_500k_dataset(self, enabled):
        self.set_dataset_mode('50w' if enabled else 'standard')

    def get_split_bounds(self, total_size, split):
        if split not in {'train', 'vali', 'test'}:
            raise ValueError(f'Unknown split: {split}.')

        if self.use_full_dataset:
            ratio_sum = self.train_ratio + self.vali_ratio + self.test_ratio
            if abs(ratio_sum - 1.0) > 1e-9:
                raise ValueError(f'Full-dataset split ratios must sum to 1.0, got {ratio_sum}.')
            train_end = int(total_size * self.train_ratio)
            vali_end = train_end + int(total_size * self.vali_ratio)
            bounds = {
                'train': (0, train_end),
                'vali': (train_end, vali_end),
                'test': (vali_end, total_size),
            }
        else:
            bounds = {
                'train': (0, self.train_set_size),
                'vali': (self.train_set_size, self.vali_set_size),
                'test': (self.vali_set_size, self.test_set_size),
            }

        start, end = bounds[split]
        if end > total_size:
            raise ValueError(
                f'Dataset mode {self.dataset_mode} requires trajectories through index {end}, '
                f'but only {total_size} trajectories are available in {self.shuffle_node_file}.'
            )
        return start, end

    def similarity_output_size_tag(self, split_size):
        # Keep the original standard filenames for backward compatibility.
        return self.dataset_size if self.dataset_mode == 'standard' else split_size


def setup_logger(fname=None):
    if not logging.root.hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=fname,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)
