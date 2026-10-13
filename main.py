import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from Trainer import GTrajsim_Trainer
from setting import SetParameter, setup_logger, zipdir
import argparse
import datetime
import os.path as osp
import zipfile
import pathlib
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Train the H2MATS trajectory encoder.')
    parser.add_argument('--dataset', choices=['beijing', 'porto', 'tdrive'], default=None)
    parser.add_argument(
        '--distance-type',
        choices=['TP', 'DITA', 'discret_frechet', 'LCRS', 'NetERP'],
        default=None,
    )
    parser.add_argument('--dataset-mode', choices=['standard', '50w', 'full'], default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--cuda', default=None)
    parser.add_argument('--train-batch', type=int, default=None)
    parser.add_argument('--test-batch', type=int, default=None)
    parser.add_argument('--use-hybrid-geo', dest='use_hybrid_geo', action='store_true', default=None)
    parser.add_argument('--no-use-hybrid-geo', dest='use_hybrid_geo', action='store_false')
    parser.add_argument('--seq-encoder', choices=['transformer', 'mamba', 'bimamba', 'hmt'], default=None)
    parser.add_argument('--graph-encoder', choices=['hybrid_multilevel', 'simple_gcn'], default=None)
    parser.add_argument('--coord-encoder', choices=['lstm', 'mamba', 'transformer'], default=None)
    parser.add_argument('--hmt-backbone', choices=['conv', 'mamba', 'bimamba', 'transformer', 'hybrid'], default=None)
    parser.add_argument('--hmt-layers', type=int, default=None)
    parser.add_argument('--hmt-kernel-size', type=int, default=None)
    parser.add_argument('--hmt-dropout', type=float, default=None)
    parser.add_argument('--hmt-transformer-heads', type=int, default=None)
    parser.add_argument('--hmt-scale-aligned-graph', dest='hmt_scale_aligned_graph', action='store_true', default=None)
    parser.add_argument('--no-hmt-scale-aligned-graph', dest='hmt_scale_aligned_graph', action='store_false')
    parser.add_argument('--coord-fusion', choices=['gated_attention', 'late_concat'], default=None)
    parser.add_argument('--save-folder', default=None)
    return parser.parse_args()


def apply_overrides(config, args):
    if args.dataset is not None:
        config.dataset = args.dataset
        config.node_file = osp.join('dataset', config.dataset, 'node.csv')
        config.edge_file = osp.join('dataset', config.dataset, 'edge_weight.csv')
        config.traj_file = osp.join('dataset', config.dataset, 'matching_result.pt')
        config.tdrive_traj_file = osp.join('dataset', config.dataset, 'matching_result.csv')
        config.tdrive_time_file = osp.join('dataset', config.dataset, 'time_drop_list.csv')
        config.hierarchy_dir = osp.join('dataset', config.dataset, 'hierarchy_louvain')
    if args.distance_type is not None:
        config.distance_type = args.distance_type
    config.set_dataset_mode(args.dataset_mode or config.dataset_mode)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError('--epochs must be positive.')
        config.epochs = args.epochs
    if args.cuda is not None:
        config.cuda = str(args.cuda)
    overrides = {
        'train_batch': args.train_batch,
        'test_batch': args.test_batch,
        'useHybridGeo': args.use_hybrid_geo,
        'seqEncoder': args.seq_encoder,
        'graphEncoder': args.graph_encoder,
        'coordEncoder': args.coord_encoder,
        'hmtBackbone': args.hmt_backbone,
        'hmtLayers': args.hmt_layers,
        'hmtKernelSize': args.hmt_kernel_size,
        'hmtDropout': args.hmt_dropout,
        'hmtTransformerHeads': args.hmt_transformer_heads,
        'hmtScaleAlignedGraph': args.hmt_scale_aligned_graph,
        'coordFusion': args.coord_fusion,
    }
    config.gtraj.update({key: value for key, value in overrides.items() if value is not None})
    if config.gtraj['seqEncoder'] == 'hmt' and not config.gtraj['useHybridGeo']:
        raise ValueError('--seq-encoder hmt requires --use-hybrid-geo.')
    if config.gtraj['graphEncoder'] == 'simple_gcn' and config.gtraj['hmtScaleAlignedGraph']:
        raise ValueError('simple_gcn requires --no-hmt-scale-aligned-graph.')
    return config


config = SetParameter()
if __name__ == '__main__':
    args = parse_args()
    config = apply_overrides(config, args)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(osp.join('log', config.dataset), exist_ok=True)
    config.save_folder = args.save_folder or osp.join(
        'saved_models', config.dataset, config.distance_type + '_' + str(current_time)
    )
    os.makedirs(config.save_folder, exist_ok=True)
    setup_logger(osp.join('log', config.dataset, config.distance_type + '_' + str(current_time) + '_train.log'))
    zipf = zipfile.ZipFile(os.path.join(config.save_folder, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    GTrajSim = GTrajsim_Trainer(config)
    logging.info(f'weights for different neighbors (alpha1,alpha2):({config.alpha1},{config.alpha2})')
    logging.info(f'number of nearest neighbors:{config.num_knn}')
    logging.info(f'hidden size:{config.hidden_size}')
    GTrajSim.Spa_train()
