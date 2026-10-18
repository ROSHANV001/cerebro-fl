# cerebro-fl

Federated 3D brain-tumor segmentation on multi-modal MRI (BraTS), built as the ML node for a **Federated Zero-Knowledge Neuroimaging Diagnostic Network**.

This repo contains the **ML side only**: local training on simulated hospital nodes, weight-delta export, FedAvg aggregation, evaluation and an inference API. The zk-proof layer, on-chain registry, IPFS storage and the web dashboard are separate workstreams (see [Roadmap](#roadmap)).

## What's in here

| Component | Details |
|---|---|
| Model | MONAI 3D U-Net, channels `(16, 32, 64, 128, 256)`, instance norm, 4 input channels (FLAIR, T1, T1ce, T2), 4 output classes |
| Training | 96³ patches, DiceCE loss, AdamW (lr 2e-4), AMP, gradient accumulation, 60 local epochs per node |
| Federation | 3 simulated hospitals (30 patients each), shared initial weights, FedAvg over weight deltas |
| Export per node | `local_model.pth`, `weight_delta.pth`, `zk_metadata.json` (scaled L2 norm of the delta, for the zk layer) |
| Serving | FastAPI `POST /predict`: 4 NIfTI volumes in, segmentation mask out |

InstanceNorm is used on purpose: batch size is small and it keeps no running statistics, so weight deltas average cleanly in FedAvg.

## Results

Dice on **9 held-out patients** (verified to have no overlap with the three training nodes). Predictions are zeroed outside the brain (see [Limitations](#limitations)). One FedAvg round after 60 local epochs per node.

| Model | WT | TC | ET | Mean |
|---|---|---|---|---|
| Node A (local) | 0.797 | 0.776 | 0.739 | 0.771 |
| Node B (local) | 0.759 | 0.761 | 0.734 | 0.751 |
| Node C (local) | 0.796 | 0.734 | 0.713 | 0.748 |
| **FedAvg (global)** | **0.809** | **0.778** | **0.747** | **0.778** |

WT = whole tumor, TC = tumor core, ET = enhancing tumor. The FedAvg model is equal to or slightly better than the best single node and clearly better than the weakest ones, but with only 9 patients the gap to Node A is within noise.

## Repo layout

```
config.py          hyperparameters, patch size, device
model.py           3D U-Net definition
dataset.py         BraTS loading + MONAI transforms
trainer.py         local training, weight-delta + L2-norm export
main.py            CLI entry point (train / infer)
inference.py       sliding-window inference on whole volumes
serve.py           FastAPI inference server
split_data.py      splits BraTS into 3 nodes + holdout
init_global.py     creates the shared initial weights
fedavg.py          averages node deltas into a new global model
eval_holdout.py    Dice (WT/TC/ET) on the holdout set
view_result.py     quick matplotlib overlay of a prediction
model_final.pth    final FedAvg model (the only weights file tracked)
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
# install PyTorch for your CUDA/CPU build first: https://pytorch.org
python -m pip install -r requirements.txt
python -m pip install matplotlib    # only needed for view_result.py
```

Trained and tested on a laptop RTX 3050 (4 GB), roughly 25 s per epoch per node.

## Data

BraTS 2021 is **not included** (size and license). Download it yourself, then:

```bash
export BRATS_TAR=/path/to/BraTS2021_Training_Data.tar    # PowerShell: $env:BRATS_TAR = "..."
python split_data.py
```

This creates `data/node_A`, `data/node_B`, `data/node_C` and `data/holdout_test` (seed 42). If you change the holdout, check that no patient ID appears in both a node folder and the holdout.

## Run the pipeline

```bash
# 1. shared starting point for every node
python init_global.py                                   # -> global_round0.pth

# 2. local training on each node (60 epochs each)
python main.py --mode train --data ./data/node_A --out ./handoff_A --weights global_round0.pth
python main.py --mode train --data ./data/node_B --out ./handoff_B --weights global_round0.pth
python main.py --mode train --data ./data/node_C --out ./handoff_C --weights global_round0.pth

# 3. aggregate
python fedavg.py                                        # -> global_round1.pth

# 4. evaluate on the holdout
python eval_holdout.py

# 5. use the best model
cp global_round1.pth model_final.pth
```

All nodes must start from the same weights, otherwise their deltas are not comparable and averaging them is meaningless.

## Inference

Batch, on a folder of patient directories:

```bash
python main.py --mode infer --data ./data/holdout_test --out ./masks --weights model_final.pth
```

API:

```bash
python serve.py        # http://localhost:8000/docs
```

`POST /predict` (multipart/form-data) with fields `flair`, `t1`, `t1ce`, `t2` (all `.nii.gz`). Returns `tumor_mask.nii.gz` with the same shape and affine as the input. Set `MODEL_PATH` to use a different weights file (default `./model_final.pth`).

| Label | Region |
|---|---|
| 0 | Background |
| 1 | NCR (necrotic core) |
| 2 | ED (edema) |
| 4 | ET (enhancing tumor) |

## Limitations

- **Outside-brain masking:** the raw network can still output tumor labels in empty background. Inference sets every voxel where the FLAIR scan is zero to background, and the evaluation does the same.
- **Small holdout:** 9 patients, so Dice differences of a few hundredths are not statistically meaningful.
- **Single FedAvg round:** the global model comes from one aggregation after long local training. Multi-round FedAvg with fewer local epochs is the natural next step.
- **Simulated federation:** the three nodes are random splits of one dataset, so the data is close to IID.
- **Research prototype, not for clinical use.** The inference API has no authentication.

## Roadmap

- [x] Phase 1: local ML node, weight-delta export, FedAvg, evaluation, inference API
- [ ] Phase 2: zk proofs (input commitment and bounded update norm)
- [ ] Phase 3: on-chain registry and IPFS anchoring
- [ ] Phase 4: federated coordinator with proof checks before aggregation
- [ ] Phase 5: dashboard with in-browser 3D MRI viewer
