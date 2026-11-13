# IBPR + BPGS: Quick Start

Version: **0.1.0rc1, 2026-11-09**. This release contains the current candidate
experiments on FOR-instance, Ofental, and LAUTx using TreeLearn (TL) and
SegmentAnyTree (SAT), with **39-dimensional IBPR and 38-dimensional BPGS**.
See [README.md](README.md) for the full documentation.

## Package Contents

- `src/ibpr_bpgs/`: candidate generation, 13 base features and 39-dimensional relations,
  scoring, region construction, marginal-gain supervision, training, inference, and evaluation.
- `configs/`: four FOR-instance/Ofental configurations and twelve LAUTx held-out-plot
  configurations, together with checkpoint SHA256 hashes.
- `configs/manifests/`: ordered training and test input manifests.
- `auxiliary/boundary/`: boundary-label preparation, predictor training, and inference;
  this component requires a separate CUDA environment.
- `results/`: 14 aggregate result rows: baseline and +A+B for FOR-instance and Ofental; baseline, +A and +A+B for LAUTx.
- `docs/`: input contracts, training splits, metric
  definitions, environment records, and release validation.
- `tests/`: checks of thresholds, matching, 5NN propagation, and marginal supervision.

Code and checkpoints are distributed separately. Extract the checkpoint archive into
the repository root to create `checkpoints/`. It contains four final A models,
sixteen final B models, and one shared boundary predictor. LAUTx reuses the
corresponding FOR-instance A model.

## Installation and Inference

The core environment uses Python 3.11.7 and scikit-learn 1.5.1.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
ibpr-bpgs verify-weights
```

Inputs are aligned baseline predictions and point features. Coordinates alone are
insufficient. Prepare NPZ files according to the [input contract](docs/DATA_FORMAT.md).
TL uses standardized and scaled 32-dimensional features and a frozen 5NN mapping.
SAT uses native 5-dimensional embeddings and outputs labels directly on the common
evaluation points.

```bash
ibpr-bpgs infer --scene data/LAUTx/SAT/p1.npz \
  --config configs/LAUTx_SAT_p1.json --weights checkpoints \
  --output outputs/LAUTx_SAT/p1 --evaluate
```

Inference does not use ground-truth labels to choose edits. The `--evaluate` option
computes reference metrics after producing predictions. Benchmark aggregation and
training commands are documented in [README.md](README.md) and
[TRAINING.md](docs/TRAINING.md).

## Version and Publication Notes

The release uses 39-dimensional IBPR scorers and the selected candidate configurations.
The FOR-instance SAT full-method reference has G = 31,930.

The code archive excludes point clouds, upstream baseline checkpoints and repositories,
historical exploratory experiments, and large prediction caches. Reproduction from raw
point clouds additionally requires the upstream models and datasets, followed by export
to the documented interface.

Commit the code directory to GitHub and distribute checkpoints as a separate release
asset. Author names, a DOI, a repository URL, and a software licence have not been
invented. The owner should add the approved licence and citation metadata before
advertising an openly licensed release. See [VALIDATION.md](docs/VALIDATION.md) for
the exact scope of the executed verification.
