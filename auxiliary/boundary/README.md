# Shared auxiliary boundary predictor

This optional component runs in the **separate TL/boundary environment**, not the core
Python3.11 scikit-learn environment. Recorded versions: Python3.12.9, PyTorch2.5.1
(CUDA11.8), NumPy2.2.4, SciPy1.15.2, laspy2.6.1. Install the matching spconv and the
SoftGroup `voxelization_idx` CUDA extension from the original upstream environment.
Do not substitute the old per-point voxelization fallback when retraining.
The main A/B package only needs precomputed boundary probabilities.

Prepare each of the 13 train and 8 validation LAS files with its recorded domain:

```bash
python auxiliary/boundary/prepare_labels.py --las YOUR_TRAIN.las --domain NIBIO \
  --output data/boundary/train/plot_merged_inst_nostuff.pth
python auxiliary/boundary/train_boundary_only.py --config auxiliary/boundary/config.yaml --epochs 100
```

Repeat using the split list in `docs/protocols/forinstance_manifest.json`; validation
files go under `data/boundary/val`. Configuration key `data.test` denotes the validation
loader, not the held-out experiment test set. The candidate uses fixed domain-specific K values.
Checkpoint selection uses sampled validation AP; see [the exact procedure](../../docs/BOUNDARY_VALIDATION.md).

Use the distributed frozen predictor to generate probabilities on common evaluation points:

```bash
python auxiliary/boundary/infer.py --points points.npz \
  --checkpoint checkpoints/boundary/best_boundary_model.pth --output probabilities.npy
```

`points.npz` contains `points` (N,3). Inference uses geometry only, zero RGB features,
0.1m voxels, at most 800,000-point cores with a 2m halo, and exactly one output per point.
Map these probabilities to TL source points using the same frozen coordinate correspondence;
SAT consumes them on the aligned evaluation points. Full neural training was not rerun
as part of code packaging. The archived checkpoint remains the reproduction reference.

See [ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md) for the complete component-specific version table.
