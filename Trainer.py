from setting import SetParameter
from model_network import GraphTrajSimEncoder
from hybrid_geo_network import HybridGeoTrajSimEncoder
import spatial_data_utils
import torch
from lossfun import SpaLossFun
import time
from tqdm import tqdm
import numpy as np
import test_method
import logging
import os.path as osp
import os


def move_hierarchy_data_to_device(hierarchy_data, device):
    hierarchy_data["level_graphs"] = [graph.to(device) for graph in hierarchy_data["level_graphs"]]
    for key in ["node_to_l1", "node_to_l2", "node_to_l3", "l1_to_l2", "l2_to_l3"]:
        hierarchy_data[key] = hierarchy_data[key].to(device)
    return hierarchy_data


class GTrajsim_Trainer(object):
    def __init__(self, config):
        self.feature_size = config.feature_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate
        self.concat = config.concat
        self.device = "cuda:" + str(config.cuda)
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs

        self.train_batch = config.gtraj["train_batch"]
        self.test_batch = config.gtraj["test_batch"]
        self.traj_file = str(config.traj_file)
        self.usePE = config.gtraj["usePE"]
        self.useSI = config.gtraj["useSI"]
        self.useLSTM = config.gtraj["useLSTM"]
        self.useHybridGeo = config.gtraj.get("useHybridGeo", False)
        self.graphEncoder = config.gtraj.get("graphEncoder", "hybrid_multilevel")
        self.hybridCurvature = config.gtraj.get("hybridCurvature", 1.0)
        self.hierarchyBlocks = config.gtraj.get("hierarchyBlocks", 2)
        self.hierarchyDropout = config.gtraj.get("hierarchyDropout", 0.1)
        self.graphScaleDropout = config.gtraj.get("graphScaleDropout", 0.2)
        self.useTransitionFeatures = config.gtraj.get("useTransitionFeatures", True)
        self.transitionResidualInit = config.gtraj.get("transitionResidualInit", -2.0)
        self.hierarchyLossWeight = config.gtraj.get("hierarchyLossWeight", 0.0)
        self.hierarchyLossWarmupEpochs = config.gtraj.get("hierarchyLossWarmupEpochs", 0)
        self.graphScaleBalanceWeight = config.gtraj.get("graphScaleBalanceWeight", 0.0)
        self.graphScaleBalanceEpochs = config.gtraj.get("graphScaleBalanceEpochs", 0)
        self.seqEncoder = config.gtraj.get("seqEncoder", "transformer")
        self.coordEncoder = config.gtraj.get("coordEncoder", "lstm")
        self.mambaLayers = config.gtraj.get("mambaLayers", 2)
        self.mambaDState = config.gtraj.get("mambaDState", 16)
        self.mambaDConv = config.gtraj.get("mambaDConv", 4)
        self.mambaExpand = config.gtraj.get("mambaExpand", 2)
        self.hmtBackbone = config.gtraj.get("hmtBackbone", "conv")
        self.hmtLayers = config.gtraj.get("hmtLayers", 2)
        self.hmtKernelSize = config.gtraj.get("hmtKernelSize", 5)
        self.hmtDropout = config.gtraj.get("hmtDropout", 0.1)
        self.hmtTransformerHeads = config.gtraj.get("hmtTransformerHeads", 4)
        self.hmtScaleAlignedGraph = config.gtraj.get("hmtScaleAlignedGraph", False)
        self.coordFusion = config.gtraj.get("coordFusion", "late_concat")
        self.coordTransformerCheckpoint = config.gtraj.get("coordTransformerCheckpoint", True)
        self.useAMP = config.gtraj.get("useAMP", False) and torch.cuda.is_available()
        self.usePretrainedFrozenGraph = config.gtraj.get("usePretrainedFrozenGraph", False)
        self.pretrainedGraphCheckpoint = config.gtraj.get("pretrainedGraphCheckpoint")
        self.pretrainedGraphStrict = config.gtraj.get("pretrainedGraphStrict", True)
        self.cacheFrozenGraph = config.gtraj.get("cacheFrozenGraph", True)
        self.fineTunePretrainedGraph = config.gtraj.get("fineTunePretrainedGraph", False)
        self.pretrainedGraphFreezeEpochs = config.gtraj.get("pretrainedGraphFreezeEpochs", 30)
        self.pretrainedGraphLrRatio = config.gtraj.get("pretrainedGraphLrRatio", 0.1)
        if self.fineTunePretrainedGraph and not self.usePretrainedFrozenGraph:
            raise ValueError(
                "fineTunePretrainedGraph requires usePretrainedFrozenGraph=True."
            )
        if (
            not isinstance(self.pretrainedGraphFreezeEpochs, int)
            or self.pretrainedGraphFreezeEpochs < 0
        ):
            raise ValueError("pretrainedGraphFreezeEpochs must be a non-negative integer.")
        if self.pretrainedGraphLrRatio <= 0:
            raise ValueError("pretrainedGraphLrRatio must be positive.")
        if self.hierarchyBlocks < 1:
            raise ValueError("hierarchyBlocks must be positive.")
        if not 0 <= self.hierarchyDropout < 1 or not 0 <= self.graphScaleDropout < 1:
            raise ValueError("Hierarchy and graph scale dropout must be in [0, 1).")
        if self.hierarchyLossWarmupEpochs < 0 or self.graphScaleBalanceEpochs < 0:
            raise ValueError("Hierarchy loss schedule epochs must be non-negative.")
        if self.graphEncoder == "simple_gcn":
            if self.hmtScaleAlignedGraph:
                raise ValueError(
                    "graphEncoder='simple_gcn' requires hmtScaleAlignedGraph=False."
                )
            if self.hierarchyLossWeight > 0:
                raise ValueError(
                    "graphEncoder='simple_gcn' requires hierarchyLossWeight=0."
                )

        self.dataset = str(config.dataset)
        self.distance_type = str(config.distance_type)
        self.early_stop = config.early_stop
        self.alpha1 = config.alpha1
        self.alpha2 = config.alpha2
        self.num_knn = config.num_knn
        self.config = config

    def Spa_eval(self, load_model=None, is_long_traj=False):
        logging.info('Run on ' + self.dataset + ' with ' + self.distance_type)
        logging.info('sequence encoder: %s', self.seqEncoder)
        logging.info('graph encoder: %s', self.graphEncoder)
        if self.graphEncoder == 'hybrid_multilevel':
            logging.info(
                'hierarchy blocks=%s, dropout=%s, scale_dropout=%s, transition_features=%s, hierarchy_loss=%s',
                self.hierarchyBlocks,
                self.hierarchyDropout,
                self.graphScaleDropout,
                self.useTransitionFeatures,
                self.hierarchyLossWeight,
            )
        if self.seqEncoder == 'hmt':
            logging.info(
                'HMT backbone=%s, layers=%s, kernel=%s, dropout=%s, transformer_heads=%s',
                self.hmtBackbone,
                self.hmtLayers,
                self.hmtKernelSize,
                self.hmtDropout,
                self.hmtTransformerHeads,
            )
            logging.info(
                'HMT scale-aligned graph=%s, coordinate fusion=%s',
                self.hmtScaleAlignedGraph,
                self.coordFusion,
            )
        if self.useHybridGeo:
            net = HybridGeoTrajSimEncoder(feature_size=self.feature_size,
                                          embedding_size=self.embedding_size,
                                          hidden_size=self.hidden_size,
                                          num_layers=self.num_layers,
                                          dropout_rate=self.dropout_rate,
                                          concat=self.concat,
                                          device=self.device,
                                          useLSTM=self.useLSTM,
                                          curvature=self.hybridCurvature,
                                          seq_encoder=self.seqEncoder,
                                          mamba_layers=self.mambaLayers,
                                          mamba_d_state=self.mambaDState,
                                          mamba_d_conv=self.mambaDConv,
                                          mamba_expand=self.mambaExpand,
                                          coord_encoder=self.coordEncoder,
                                          hmt_backbone=self.hmtBackbone,
                                          hmt_layers=self.hmtLayers,
                                          hmt_kernel_size=self.hmtKernelSize,
                                          hmt_dropout=self.hmtDropout,
                                          hmt_transformer_heads=self.hmtTransformerHeads,
                                          hmt_scale_aligned_graph=self.hmtScaleAlignedGraph,
                                          coord_fusion=self.coordFusion,
                                          coord_transformer_checkpoint=self.coordTransformerCheckpoint,
                                          graph_encoder=self.graphEncoder,
                                          hierarchy_blocks=self.hierarchyBlocks,
                                          hierarchy_dropout=self.hierarchyDropout,
                                          graph_scale_dropout=self.graphScaleDropout,
                                          use_transition_features=self.useTransitionFeatures,
                                          transition_residual_init=self.transitionResidualInit)
        else:
            net = GraphTrajSimEncoder(feature_size=self.feature_size,
                                      embedding_size=self.embedding_size,
                                      hidden_size=self.hidden_size,
                                      num_layers=self.num_layers,
                                      dropout_rate=self.dropout_rate,
                                      concat=self.concat,
                                      device=self.device,
                                      usePE=self.usePE,
                                      useSI=self.useSI,
                                      useLSTM=self.useLSTM,
                                      dataset=self.dataset,
                                      alpha1=self.alpha1,
                                      alpha2=self.alpha2,
                                      seq_encoder=self.seqEncoder,
                                      mamba_layers=self.mambaLayers,
                                      mamba_d_state=self.mambaDState,
                                      mamba_d_conv=self.mambaDConv,
                                      mamba_expand=self.mambaExpand,
                                      coord_encoder=self.coordEncoder)
        if load_model != None:
            logging.info(load_model)
            net.load_state_dict(torch.load(load_model))
            net.to(self.device)
            if self.usePretrainedFrozenGraph:
                if not self.useHybridGeo:
                    raise ValueError("usePretrainedFrozenGraph currently requires useHybridGeo=True.")
                net.freeze_graph_encoder(cache=self.cacheFrozenGraph)
            if is_long_traj:
                test_data_loader, test_lenth = spatial_data_utils.vali_data_loader(
                    self.config.spatial_long_test_set, self.test_batch)
            else:
                test_data_loader, test_lenth = spatial_data_utils.vali_data_loader(
                    self.config.spatial_test_set, self.test_batch)

            logging.info(f'test size:{test_lenth}')
            if self.useHybridGeo:
                road_network = move_hierarchy_data_to_device(
                    spatial_data_utils.load_hierarchy_multilevel(self.dataset),
                    self.device
                )
            elif self.useSI:
                road_network = spatial_data_utils.load_neighbor(self.dataset, self.num_knn)
                for item in road_network:
                    item = item.to(self.device)
            else:
                road_network = spatial_data_utils.load_netowrk(self.dataset).to(self.device)
            if self.useHybridGeo:
                network_input = road_network
            else:
                distance_to_anchor_node = spatial_data_utils.load_region_node(self.dataset).to(self.device)
                network_input = [road_network, distance_to_anchor_node]

            if self.usePretrainedFrozenGraph and self.cacheFrozenGraph:
                net.prepare_frozen_graph(network_input)
                logging.info("prepared frozen graph cache for trajectory evaluation")

            net.eval()
            with torch.no_grad():
                start_test_epoch = time.time()
                if self.useLSTM:
                    test_embedding = torch.zeros((test_lenth, self.hidden_size*2),
                                                 device=self.device, requires_grad=False)
                else:
                    test_embedding = torch.zeros((test_lenth, self.hidden_size),
                                                 device=self.device, requires_grad=False)
                test_label = torch.zeros((test_lenth, 50), requires_grad=False, dtype=torch.long)

                for batch in tqdm(test_data_loader):
                    (data, coor, label, data_length, idx) = batch
                    a_embedding = net(network_input, data, coor, data_length)
                    test_embedding[idx] = a_embedding.to(
                        device=test_embedding.device,
                        dtype=test_embedding.dtype,
                    )
                    test_label[idx] = label
                end_test_epoch = time.time()
                acc = test_method.test_spa_model(test_embedding, test_label, self.device)
                acc = acc.mean(axis=0)  # HR-10 HR-50 R10@50 R1@1 R1@10 R1@50
                acc[0] = acc[0] / 10.0
                acc[1] = acc[1] / 50.0
                acc[2] = acc[2] / 10.0
                logging.info('Dataset: {}, Distance type: {}, f_num is {}'.format(
                    self.dataset, self.distance_type, test_lenth))
                end_test = time.time()
                logging.info(f"epoch test time: {end_test_epoch - start_test_epoch}")
                logging.info(f"all test time: {end_test - start_test_epoch}")
                logging.info(acc)

    def Spa_train(self):
        logging.info('Run on ' + self.dataset + ' with ' + self.distance_type)
        logging.info('positive num: ' + str(self.config.pos_num))

        logging.info('GPU: '+str(self.config.cuda))
        logging.info('automatic mixed precision: %s', self.useAMP)
        logging.info(
            'coordinate Transformer activation checkpointing: %s',
            self.coordTransformerCheckpoint,
        )
        logging.info('sequence encoder: %s', self.seqEncoder)
        logging.info('graph encoder: %s', self.graphEncoder)
        if self.graphEncoder == 'hybrid_multilevel':
            logging.info(
                'hierarchy blocks=%s, dropout=%s, scale_dropout=%s, transition_features=%s, hierarchy_loss=%s, '
                'scale_balance=%s for %s epochs',
                self.hierarchyBlocks,
                self.hierarchyDropout,
                self.graphScaleDropout,
                self.useTransitionFeatures,
                self.hierarchyLossWeight,
                self.graphScaleBalanceWeight,
                self.graphScaleBalanceEpochs,
            )
        if self.seqEncoder == 'hmt':
            logging.info(
                'HMT backbone=%s, layers=%s, kernel=%s, dropout=%s, transformer_heads=%s',
                self.hmtBackbone,
                self.hmtLayers,
                self.hmtKernelSize,
                self.hmtDropout,
                self.hmtTransformerHeads,
            )
            logging.info(
                'HMT scale-aligned graph=%s, coordinate fusion=%s',
                self.hmtScaleAlignedGraph,
                self.coordFusion,
            )
        if self.useHybridGeo:
            net = HybridGeoTrajSimEncoder(feature_size=self.feature_size,
                                          embedding_size=self.embedding_size,
                                          hidden_size=self.hidden_size,
                                          num_layers=self.num_layers,
                                          dropout_rate=self.dropout_rate,
                                          concat=self.concat,
                                          device=self.device,
                                          useLSTM=self.useLSTM,
                                          curvature=self.hybridCurvature,
                                          seq_encoder=self.seqEncoder,
                                          mamba_layers=self.mambaLayers,
                                          mamba_d_state=self.mambaDState,
                                          mamba_d_conv=self.mambaDConv,
                                          mamba_expand=self.mambaExpand,
                                          coord_encoder=self.coordEncoder,
                                          hmt_backbone=self.hmtBackbone,
                                          hmt_layers=self.hmtLayers,
                                          hmt_kernel_size=self.hmtKernelSize,
                                          hmt_dropout=self.hmtDropout,
                                          hmt_transformer_heads=self.hmtTransformerHeads,
                                          hmt_scale_aligned_graph=self.hmtScaleAlignedGraph,
                                          coord_fusion=self.coordFusion,
                                          coord_transformer_checkpoint=self.coordTransformerCheckpoint,
                                          graph_encoder=self.graphEncoder,
                                          hierarchy_blocks=self.hierarchyBlocks,
                                          hierarchy_dropout=self.hierarchyDropout,
                                          graph_scale_dropout=self.graphScaleDropout,
                                          use_transition_features=self.useTransitionFeatures,
                                          transition_residual_init=self.transitionResidualInit).to(self.device)
        else:
            net = GraphTrajSimEncoder(feature_size=self.feature_size,
                                      embedding_size=self.embedding_size,
                                      hidden_size=self.hidden_size,
                                      num_layers=self.num_layers,
                                      dropout_rate=self.dropout_rate,
                                      concat=self.concat,
                                      device=self.device,
                                      usePE=self.usePE,
                                      useSI=self.useSI,
                                      useLSTM=self.useLSTM,
                                      dataset=self.dataset,
                                      alpha1=self.alpha1,
                                      alpha2=self.alpha2,
                                      seq_encoder=self.seqEncoder,
                                      mamba_layers=self.mambaLayers,
                                      mamba_d_state=self.mambaDState,
                                      mamba_d_conv=self.mambaDConv,
                                      mamba_expand=self.mambaExpand,
                                      coord_encoder=self.coordEncoder).to(self.device)

        if self.usePretrainedFrozenGraph:
            if not self.useHybridGeo:
                raise ValueError("usePretrainedFrozenGraph currently requires useHybridGeo=True.")
            if not self.pretrainedGraphCheckpoint:
                raise ValueError(
                    "Set gtraj['pretrainedGraphCheckpoint'] when "
                    "gtraj['usePretrainedFrozenGraph'] is enabled."
                )
            missing, unexpected = net.load_pretrained_graph(
                self.pretrainedGraphCheckpoint,
                map_location=self.device,
                strict=self.pretrainedGraphStrict,
            )
            if not self.fineTunePretrainedGraph or self.pretrainedGraphFreezeEpochs > 0:
                net.freeze_graph_encoder(cache=self.cacheFrozenGraph)
                logging.info(
                    "loaded and froze pretrained graph: %s (cache=%s, missing=%d, unexpected=%d)",
                    self.pretrainedGraphCheckpoint,
                    self.cacheFrozenGraph,
                    len(missing),
                    len(unexpected),
                )
            else:
                logging.info(
                    "loaded pretrained graph for immediate fine-tuning: %s "
                    "(missing=%d, unexpected=%d)",
                    self.pretrainedGraphCheckpoint,
                    len(missing),
                    len(unexpected),
                )

        if self.fineTunePretrainedGraph:
            graph_parameters = list(net.graph_parameters())
            graph_parameter_ids = {id(parameter) for parameter in graph_parameters}
            downstream_parameters = [
                parameter
                for parameter in net.parameters()
                if id(parameter) not in graph_parameter_ids
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": downstream_parameters, "lr": self.learning_rate},
                    {
                        "params": graph_parameters,
                        "lr": self.learning_rate * self.pretrainedGraphLrRatio,
                    },
                ],
                weight_decay=0.0001,
            )
        else:
            optimizer = torch.optim.AdamW(
                [parameter for parameter in net.parameters() if parameter.requires_grad],
                lr=self.learning_rate,
                weight_decay=0.0001,
            )
        scaler = torch.amp.GradScaler('cuda', enabled=self.useAMP)
        milestones_list = [15, 30, 45, 60, 75, 90, 115]
        logging.info(milestones_list)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_list, gamma=0.2)

        lossfunction = SpaLossFun(self.train_batch, self.distance_type).to(self.device)

        if self.useHybridGeo:
            road_network = move_hierarchy_data_to_device(
                spatial_data_utils.load_hierarchy_multilevel(self.dataset),
                self.device
            )
        elif self.useSI:
            road_network = spatial_data_utils.load_neighbor(self.dataset, self.num_knn)
            for item in road_network:
                item = item.to(self.device)
        else:
            road_network = spatial_data_utils.load_netowrk(self.dataset).to(self.device)
        if self.useHybridGeo:
            network_input = road_network
        else:
            distance_to_anchor_node = spatial_data_utils.load_region_node(self.dataset).to(self.device)
            network_input = [road_network, distance_to_anchor_node]

        if self.usePretrainedFrozenGraph and net.graph_frozen and self.cacheFrozenGraph:
            net.prepare_frozen_graph(network_input)
            logging.info("prepared frozen graph cache for trajectory training")

        train_data_loader = spatial_data_utils.train_data_loader(self.config.spatial_train_set, self.train_batch)
        vali_data_loader,  vali_lenth = spatial_data_utils.vali_data_loader(
            self.config.spatial_vali_set, self.train_batch)

        best_epoch = 0
        best_hr10 = 0
        for epoch in range(0, self.epochs):
            if (
                self.fineTunePretrainedGraph
                and net.graph_frozen
                and epoch == self.pretrainedGraphFreezeEpochs
            ):
                net.unfreeze_graph_encoder()
                logging.info(
                    "unfroze pretrained graph at epoch %d; graph learning rate=%g "
                    "(ratio=%g)",
                    epoch + 1,
                    optimizer.param_groups[1]["lr"],
                    self.pretrainedGraphLrRatio,
                )
            net.train()
            losses = []
            start_train = time.time()
            for batch in tqdm(train_data_loader):
                optimizer.zero_grad(set_to_none=True)
                (data, coor, data_pos, coor_pos, data_neg, coor_neg,
                 data_pos_dis, data_neg_dis,
                 data_length, pos_length, neg_length) = batch
                with torch.amp.autocast('cuda', enabled=self.useAMP):
                    if self.useHybridGeo:
                        a_embedding, p_embedding, n_embedding = net.forward_triplet(
                            network_input,
                            (data, coor, data_length),
                            (data_pos, coor_pos, pos_length),
                            (data_neg, coor_neg, neg_length),
                        )
                    else:
                        a_embedding = net(network_input, data, coor, data_length)
                        p_embedding = net(network_input, data_pos, coor_pos, pos_length)
                        n_embedding = net(network_input, data_neg, coor_neg, neg_length)

                    loss = lossfunction(
                        a_embedding,
                        p_embedding,
                        n_embedding,
                        data_pos_dis,
                        data_neg_dis,
                        self.device,
                    )
                    if (
                        self.useHybridGeo
                        and not net.graph_frozen
                        and self.hierarchyLossWeight > 0
                    ):
                        if self.hierarchyLossWarmupEpochs > 0:
                            hierarchy_ramp = min(
                                1.0,
                                float(epoch + 1) / float(self.hierarchyLossWarmupEpochs),
                            )
                        else:
                            hierarchy_ramp = 1.0
                        loss = loss + (
                            self.hierarchyLossWeight
                            * hierarchy_ramp
                            * net.hierarchy_loss()
                        )
                    if (
                        self.useHybridGeo
                        and not net.graph_frozen
                        and self.graphScaleBalanceWeight > 0
                        and (
                            self.graphScaleBalanceEpochs <= 0
                            or epoch < self.graphScaleBalanceEpochs
                        )
                    ):
                        loss = loss + self.graphScaleBalanceWeight * net.graph_scale_balance_loss()
                losses.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            end_train = time.time()
            scheduler.step()
            current_lrs = scheduler.get_last_lr()
            if self.fineTunePretrainedGraph:
                logging.info(
                    "Used learning rates: downstream=%g, graph=%g",
                    current_lrs[0],
                    current_lrs[1],
                )
            else:
                logging.info(f'Used learning rate:{current_lrs[0]}')
            if self.useHybridGeo and net.last_graph_scale_weights is not None:
                graph_scale_usage = (
                    net.last_graph_scale_weights.detach().float().mean(dim=0).cpu().tolist()
                )
                logging.info(
                    'Graph scale usage L0/L1/L2/L3: %s',
                    [round(value, 4) for value in graph_scale_usage],
                )
            logging.info('Epoch {}: Training time is {}, loss is {}'.format(
                epoch, (end_train - start_train), np.mean(losses)))
            if epoch % 1 == 0:
                net.eval()
                with torch.no_grad():
                    start_vali = time.time()
                    if self.useLSTM:
                        vali_embedding = torch.zeros((vali_lenth, self.hidden_size*2),
                                                     device=self.device, requires_grad=False)
                    else:
                        vali_embedding = torch.zeros((vali_lenth, self.hidden_size),
                                                     device=self.device, requires_grad=False)
                    vali_label = torch.zeros((vali_lenth, 50), requires_grad=False, dtype=torch.long)

                    for batch in tqdm(vali_data_loader):
                        (data, coor, label, data_length, idx) = batch
                        with torch.amp.autocast('cuda', enabled=self.useAMP):
                            a_embedding = net(network_input, data, coor, data_length)
                        vali_embedding[idx] = a_embedding.to(
                            device=vali_embedding.device,
                            dtype=vali_embedding.dtype,
                        )
                        vali_label[idx] = label

                    acc = test_method.test_spa_model(vali_embedding, vali_label, self.device)
                    acc = acc.mean(axis=0)  # HR-10 HR-50 R10@50 R1@1 R1@10 R1@50
                    acc[0] = acc[0] / 10.0
                    acc[1] = acc[1] / 50.0
                    acc[2] = acc[2] / 10.0
                    logging.info('Dataset: {}, Distance type: {}, f_num is {}'.format(
                        self.dataset, self.distance_type, vali_lenth))
                    end_vali = time.time()
                    logging.info(f"vali time: {end_vali - start_vali}")
                    logging.info(acc)
                    logging.info(" ")
                    save_modelname = self.config.save_folder + "/epoch_%d.pt" % epoch
                    if not os.path.exists(self.config.save_folder):
                        os.makedirs(self.config.save_folder)
                    torch.save(net.state_dict(), save_modelname)

                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                        logging.info(f'best epoch: {best_epoch}')
                    if epoch - best_epoch >= self.early_stop:
                        logging.info(save_modelname)
                        break
