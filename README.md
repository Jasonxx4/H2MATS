# H²MATS

Source code for **H²MATS: Hybrid-Geometry and Hierarchy-Aligned Multi-Scale Adaptive Trajectory Similarity Learning over Road Networks**.

H²MATS learns trajectory embeddings that approximate a road-network trajectory distance (TP, DITA, LCRS, NetERP, or Seg-Fréchet) and answers top-k similarity queries with a hierarchy-aware adaptive candidate selection stage.

- **Road hierarchy.** Road nodes (L0) are grouped into neighborhoods (L1), districts (L2), and macro-regions (L3) by weighted Louvain partitioning with refinement.
- **Hybrid-geometry graph encoder.** A Euclidean branch models the semantics within each level and a hyperbolic (Poincaré ball) branch models the containment between levels.
- **Hierarchy-aligned multi-scale trajectory encoder (HMT).** A trajectory is encoded at each hierarchy level, with segments split at region boundaries.
- **Adaptive candidate selection.** A 128-bit base code scan answers most queries. Queries with a small normalized Hamming margin are augmented with candidates from a metric hash head and from L2/L3 region-transition postings, and their candidates are reranked by the exact distance under a query budget.

## Repository layout

| Stage | Files |
| --- | --- |
| Configuration | `setting.py` |
| Preprocessing | `spatial_preprocess.py`, `spatial_similarity_computation.py`, `generate_trajectory.py`, `generate_louvain_hierarchy.py`, `generate_node_knn.py`, `STmatching_distribution_ver.py`, `Model.py` |
| Encoder | `hybrid_geo_network.py`, `hyperbolic_utils.py`, `graph_normalization.py`, `sequence_encoders.py`, `model_network.py`, `spatial_data_utils.py` |
| Training and evaluation | `main.py`, `Trainer.py`, `lossfun.py`, `test_method.py`, `eval_spatial_model.py` |
| Hash heads | `hash_global.py`, `hash_global_pipeline.py`, `run_hash_global_all.py`, `trajectory_embedding.py`, `build_head_training_set.py`, `build_large_scale_head_training.py`, `build_validation_metric_distances.py`, `build_validation_union_tp_teacher.py` |
| Candidate selection | `run_adaptive_topo_hash_checkpoint.py`, `topo_route_hash_large_scale.py`, `dual_metric_fusion.py`, `hier_hash_large_scale.py`, `hier_hash_fusion.py`, `trajectory_metric_backend.py`, `tp_retrieval_evaluation.py`, `ann_index.py`, `ann_pipeline.py`, `paired_retrieval_experiment.py` |
| Large-scale benchmark | `build_augmented_tp_benchmark.py`, `restore_benchmark_coordinates.py`, `reencode_embedding_payload.py`, `run_augmented_tp_experiment.py`, `hnsw_large_scale.py` |
| Experiment scripts | `scripts/*.sh` |
| Unit tests | `tests/` |

## Requirements

Python 3.9 or later and a CUDA GPU are recommended.

```bash
pip install -r requirements.txt
```

Install PyTorch and PyTorch Geometric builds that match your CUDA version. `mamba-ssm` is optional and only needed for the Mamba sequence backbones.

## Data

The experiments use the Beijing and Porto road networks and trajectories released with SPGMT. Place each city under `dataset/<city>/`:

```text
dataset/<city>/node.csv            # node, lng, lat
dataset/<city>/edge_weight.csv     # s_node, e_node, length
dataset/<city>/matching_result.pt  # trajectories as road-node sequences with coordinates
```

All generated files are written under `data/`, `ground_truth/`, `dataset/<city>/hierarchy_louvain/`, `saved_models/`, and `ann_artifacts/`, which are ignored by git.

## Pipeline

The preprocessing scripts read the city and the distance from `dataset` and `distance_type` in `setting.py` (`TP`, `DITA`, `LCRS`, `NetERP`, or `discret_frechet`, which is Seg-Fréchet in the paper). Run every command from the repository root.

1. **Preprocess trajectories.** This writes `data/<city>/st_traj/shuffle_*.npy` and the node2vec features.

   ```bash
   python spatial_preprocess.py
   ```

2. **Compute ground-truth distances.** This writes `ground_truth/<city>/Point_dis_matrix.npy` and the pairwise trajectory distances. Batches are resumable.

   ```bash
   python spatial_similarity_computation.py --workers 16 --batch-size 50
   ```

3. **Build the training, validation, and test sets.** This writes `ground_truth/<city>/<distance>/spatial_{train,vali,test}_set.npz`.

   ```bash
   python generate_trajectory.py
   ```

4. **Build the road hierarchy.** This writes `node_to_l{1,2,3}.npy` and the transition features to `dataset/<city>/hierarchy_louvain/`. Use `--transition-max-trajectories <train_size>` to compute the transition statistics from training trajectories only.

   ```bash
   python generate_louvain_hierarchy.py --dataset porto
   ```

5. **(SPGMT baseline only)** Build the kNN structure graphs and the anchor distances.

   ```bash
   python generate_node_knn.py
   python spatial_data_utils.py
   ```

## Training and embedding evaluation

The default configuration in `setting.py` trains the full model (hybrid-geometry graph encoder, HMT with the hybrid backbone, and gated coordinate fusion).

```bash
python main.py --dataset porto --distance-type TP --cuda 0
bash scripts/eval_embedding.sh saved_models/porto/TP_<timestamp>/epoch_49.pt
```

Checkpoints are saved to `saved_models/<city>/<distance>_<timestamp>/`. The evaluation reports HR-10, HR-50, R10@50, R1@1, R1@10, and R1@50 on the test set.

## Adaptive candidate selection

```bash
H2MATS_DATASET=porto H2MATS_DISTANCE_TYPE=TP \
  bash scripts/run_candidate_selection.sh saved_models/porto/TP_<timestamp>/epoch_49.pt
```

The first run exports the embeddings and trains the base and metric hash heads into `<output-dir>/hash_assets`. Later runs reuse them unless `--retrain-hash` is passed. The script evaluates three methods.

| Method name in the code | Method in the paper |
| --- | --- |
| `global-hash` | Code-Scan |
| `adaptive-dual-metric-no-rerank` | Code-Scan+Dual |
| `adaptive-dual-metric` | H²MATS |

`scripts/run_candidate_selection_hnsw.sh` replaces the code scan with a Hamming HNSW index. The main hyperparameters are listed below.

| Option | Value | Meaning |
| --- | --- | --- |
| `--hamming-candidates` | 768 | base candidates from the code scan |
| `--adaptive-max-tp-fraction` | 0.1 | query budget for candidate augmentation and exact reranking |
| `--max-route-candidates` | 192 | cap on hierarchical candidates |

## Large-scale experiments

The large-scale benchmark keeps the trajectories outside the training selection, adds perturbed distractors to reach galleries of up to one million trajectories, and uses perturbed queries whose clean source stays in the gallery.

```bash
# Build the benchmark, export embeddings, and evaluate at 500K and 1M
bash scripts/run_large_scale_pipeline.sh saved_models/porto/TP_<timestamp>/epoch_49.pt

# Evaluate exported Query/Gallery embeddings at 500K-1M
bash scripts/run_large_scale_flat.sh       # code scan
bash scripts/run_large_scale_hnsw.sh       # Ours-HNSW
bash scripts/run_large_scale_emb_hnsw.sh   # Emb-HNSW baseline
```

All scripts read their paths and devices from environment variables such as `H2MATS_DATASET`, `H2MATS_DISTANCE_TYPE`, `H2MATS_BENCHMARK_DIR`, `H2MATS_GALLERY_FILE`, `H2MATS_QUERY_FILE`, `H2MATS_HASH_ASSETS_DIR`, `H2MATS_HASH_CUDA`, and `H2MATS_MODEL_CUDA`. Set `H2MATS_DRY_RUN=1` to print a command without running it. The exact distance table for 2,000 queries over one million trajectories takes about 8 GB of disk space and is cached after the first run.

## Tests

```bash
python -m pytest tests
```

## Acknowledgement

The data preprocessing, ground-truth computation, and training framework build on the code of SPGMT, which is also used as a baseline in our experiments.
