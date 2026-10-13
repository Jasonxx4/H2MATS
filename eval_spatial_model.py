import argparse
import logging
import os
import os.path as osp
import re
from pathlib import Path


def refresh_config_paths(config):
    config.node_file = osp.join('dataset', config.dataset, 'node.csv')
    config.edge_file = osp.join('dataset', config.dataset, 'edge_weight.csv')
    config.traj_file = osp.join('dataset', config.dataset, 'matching_result.pt')
    config.tdrive_traj_file = osp.join('dataset', config.dataset, 'matching_result.csv')
    config.tdrive_time_file = osp.join('dataset', config.dataset, 'time_drop_list.csv')
    config.set_dataset_mode(config.dataset_mode)
    config.spatial_long_test_set = osp.join(
        'ground_truth', config.dataset, config.distance_type, 'spatial_long_test_set.npz'
    )
    config.spatial_num_test_set = osp.join(
        'ground_truth', config.dataset, config.distance_type, 'spatial_num_test_set.npz'
    )
    config.hierarchy_dir = osp.join('dataset', config.dataset, 'hierarchy_louvain')


def resolve_checkpoint(args, config, original_cwd):
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_absolute():
            cwd_path = original_cwd / path
            path = cwd_path if cwd_path.exists() else path
    elif args.run_time and args.epoch is not None:
        path = Path('saved_models') / config.dataset / f'{config.distance_type}_{args.run_time}' / f'epoch_{args.epoch}.pt'
    else:
        raise ValueError('Pass --checkpoint, or pass both --run-time and --epoch.')

    if path.is_dir():
        checkpoints = sorted(
            path.glob('epoch_*.pt'),
            key=lambda p: int(re.search(r'epoch_(\d+)\.pt$', p.name).group(1))
            if re.search(r'epoch_(\d+)\.pt$', p.name)
            else -1,
        )
        if not checkpoints:
            raise FileNotFoundError(f'No epoch_*.pt checkpoint found in {path}')
        path = checkpoints[-1]

    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return str(path)


def apply_args_to_config(config, args):
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.distance_type is not None:
        config.distance_type = args.distance_type
    if args.dataset_mode is not None:
        config.dataset_mode = args.dataset_mode
    refresh_config_paths(config)

    if args.cuda is not None:
        config.cuda = str(args.cuda)
    if args.test_batch is not None:
        config.gtraj['test_batch'] = args.test_batch
    if args.use_hybrid_geo is not None:
        config.gtraj['useHybridGeo'] = args.use_hybrid_geo
    if args.graph_encoder is not None:
        config.gtraj['graphEncoder'] = args.graph_encoder
        if args.graph_encoder == 'simple_gcn':
            config.gtraj['hmtScaleAlignedGraph'] = False
            config.gtraj['hierarchyLossWeight'] = 0.0
    if args.hierarchy_blocks is not None:
        config.gtraj['hierarchyBlocks'] = args.hierarchy_blocks
    if args.hierarchy_dropout is not None:
        config.gtraj['hierarchyDropout'] = args.hierarchy_dropout
    if args.graph_scale_dropout is not None:
        config.gtraj['graphScaleDropout'] = args.graph_scale_dropout
    if args.use_transition_features is not None:
        config.gtraj['useTransitionFeatures'] = args.use_transition_features
    if args.seq_encoder is not None:
        config.gtraj['seqEncoder'] = args.seq_encoder
    if args.coord_encoder is not None:
        config.gtraj['coordEncoder'] = args.coord_encoder
    if args.mamba_layers is not None:
        config.gtraj['mambaLayers'] = args.mamba_layers
    if args.mamba_d_state is not None:
        config.gtraj['mambaDState'] = args.mamba_d_state
    if args.mamba_d_conv is not None:
        config.gtraj['mambaDConv'] = args.mamba_d_conv
    if args.mamba_expand is not None:
        config.gtraj['mambaExpand'] = args.mamba_expand
    if args.hmt_backbone is not None:
        config.gtraj['hmtBackbone'] = args.hmt_backbone
    if args.hmt_layers is not None:
        config.gtraj['hmtLayers'] = args.hmt_layers
    if args.hmt_kernel_size is not None:
        config.gtraj['hmtKernelSize'] = args.hmt_kernel_size
    if args.hmt_dropout is not None:
        config.gtraj['hmtDropout'] = args.hmt_dropout
    if args.hmt_transformer_heads is not None:
        config.gtraj['hmtTransformerHeads'] = args.hmt_transformer_heads
    if args.hmt_scale_aligned_graph is not None:
        config.gtraj['hmtScaleAlignedGraph'] = args.hmt_scale_aligned_graph
    if args.coord_fusion is not None:
        config.gtraj['coordFusion'] = args.coord_fusion


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluate a trained trajectory encoder checkpoint.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to epoch_*.pt or a checkpoint directory.')
    parser.add_argument('--run-time', type=str, default=None, help='Run timestamp used under saved_models/<dataset>.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch number used with --run-time.')
    parser.add_argument('--dataset', choices=['beijing', 'porto', 'tdrive'], default=None)
    parser.add_argument('--dataset-mode', choices=['standard', '50w', 'full'], default=None)
    parser.add_argument('--distance-type', choices=['TP', 'DITA', 'discret_frechet', 'LCRS', 'NetERP'], default=None)
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device id, for example 0 or 1.')
    parser.add_argument('--test-batch', type=int, default=None)
    parser.add_argument('--seq-encoder', choices=['transformer', 'mamba', 'bimamba', 'hmt'], default=None)
    parser.add_argument(
        '--graph-encoder',
        choices=['hybrid_multilevel', 'simple_gcn'],
        default=None,
    )
    parser.add_argument('--hierarchy-blocks', type=int, default=None)
    parser.add_argument('--hierarchy-dropout', type=float, default=None)
    parser.add_argument('--graph-scale-dropout', type=float, default=None)
    parser.add_argument('--use-transition-features', dest='use_transition_features',
                        action='store_true', default=None)
    parser.add_argument('--no-transition-features', dest='use_transition_features',
                        action='store_false')
    parser.add_argument('--coord-encoder', choices=['lstm', 'mamba', 'transformer'], default=None)
    parser.add_argument('--mamba-layers', type=int, default=None)
    parser.add_argument('--mamba-d-state', type=int, default=None)
    parser.add_argument('--mamba-d-conv', type=int, default=None)
    parser.add_argument('--mamba-expand', type=int, default=None)
    parser.add_argument(
        '--hmt-backbone',
        choices=['conv', 'mamba', 'bimamba', 'transformer', 'hybrid'],
        default=None,
    )
    parser.add_argument('--hmt-layers', type=int, default=None)
    parser.add_argument('--hmt-kernel-size', type=int, default=None)
    parser.add_argument('--hmt-dropout', type=float, default=None)
    parser.add_argument('--hmt-transformer-heads', type=int, default=None)
    parser.add_argument(
        '--hmt-scale-aligned-graph',
        dest='hmt_scale_aligned_graph',
        action='store_true',
        default=None,
    )
    parser.add_argument(
        '--no-hmt-scale-aligned-graph',
        dest='hmt_scale_aligned_graph',
        action='store_false',
    )
    parser.add_argument(
        '--coord-fusion',
        choices=['gated_attention', 'late_concat'],
        default=None,
    )
    parser.add_argument('--use-hybrid-geo', dest='use_hybrid_geo', action='store_true', default=None)
    parser.add_argument('--no-use-hybrid-geo', dest='use_hybrid_geo', action='store_false')
    parser.add_argument('--long-traj', action='store_true', help='Evaluate spatial_long_test_set.npz.')
    parser.add_argument('--log-file', type=str, default=None)
    return parser


def main():
    original_cwd = Path.cwd()
    os.chdir(Path(__file__).resolve().parent)

    args = build_parser().parse_args()

    from Trainer import GTrajsim_Trainer
    from setting import SetParameter, setup_logger

    config = SetParameter()
    apply_args_to_config(config, args)
    checkpoint = resolve_checkpoint(args, config, original_cwd)

    log_file = args.log_file
    if log_file is None:
        checkpoint_tag = Path(checkpoint).stem
        split_tag = 'long_test' if args.long_traj else 'test'
        log_file = osp.join(
            'log',
            config.dataset,
            f'{config.distance_type}_{checkpoint_tag}_{split_tag}.log',
        )
    log_dir = osp.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_file)

    logging.info(f'checkpoint: {checkpoint}')
    logging.info(
        f'dataset: {config.dataset}, distance_type: {config.distance_type}, '
        f'dataset_mode: {config.dataset_mode}'
    )
    logging.info(
        'useHybridGeo: %s, graphEncoder: %s, seqEncoder: %s, coordEncoder: %s',
        config.gtraj.get('useHybridGeo'),
        config.gtraj.get('graphEncoder'),
        config.gtraj.get('seqEncoder'),
        config.gtraj.get('coordEncoder'),
    )
    logging.info(
        'hmtScaleAlignedGraph: %s, coordFusion: %s',
        config.gtraj.get('hmtScaleAlignedGraph'),
        config.gtraj.get('coordFusion'),
    )
    logging.info(f'cuda: {config.cuda}, test_batch: {config.gtraj["test_batch"]}')

    trainer = GTrajsim_Trainer(config)
    trainer.Spa_eval(load_model=checkpoint, is_long_traj=args.long_traj)


if __name__ == '__main__':
    main()
