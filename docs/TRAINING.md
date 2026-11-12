# Reproduce the selected 39D training

Run from the repository root in the pinned core environment. Prepare training scenes
using the same frozen base-model exports, source/evaluation point order and boundary
probabilities. TL `source_gt` must be geometrically aligned reference labels; it cannot
be obtained by blindly indexing evaluation GT with source-point indices.

```bash
ibpr-bpgs-train reproduce-fixed \
  --manifest configs/manifests/FOR-instance_TL_train.json \
  --config configs/FOR-instance_TL.json --output outputs/train_FOR_TL
ibpr-bpgs-train reproduce-fixed \
  --manifest configs/manifests/Ofental_SAT_train.json \
  --config configs/Ofental_SAT.json --output outputs/train_Ofental_SAT
```

FOR: 13 ordered training plots, seed 20261029+plot-index, at most 250,000 probes and
30,000 competitive points per plot. Five OOF groups are manifest-index modulo 5.
Each region target comes from the scorer fitted without that plot's group.
Ofental: seed 20261106, at most 800,000 probes / 120,000 competitive points; five x-axis
quantile folds. Each OOF scorer is fitted without the held spatial fold; its suggestions
are restricted to that fold before building training regions.

A uses maximum-intersection Hungarian labels. B uses a frozen baseline maximum-IoU
matching and all-proposals-minus-one-region exact replay, including TL 5NN. Selected
parameters and size-weight exponents are in each config. `reproduce-fixed` does not
inspect validation/test GT or reselect thresholds. Historical validation selections,
including Ofental's IoU/mAP constraints, are recorded in `docs/selection/`.

# LAUTx frozen A, nested B

Generate six plot region/target caches using the fixed FOR-instance A:

```bash
ibpr-bpgs-train regions --scene data/LAUTx/SAT/p1.npz \
  --a-model checkpoints/FOR-instance/SAT/A.pkl --config configs/LAUTx_SAT_p1.json \
  --output data/LAUTx/SAT/p1
```

Repeat for the other five plots, then:

```bash
ibpr-bpgs-train nested-b --manifest configs/manifests/LAUTx_SAT_nested.json \
  --config configs/LAUTx_SAT_p1.json --output outputs/LAUTx_SAT_nested
```

Each outer plot is excluded from both inner fitting and selection. Inner folds train
on four of the remaining plots and validate on the fifth. Select aggregate G, break
ties by fewer modified points, then refit B on all five. The grid is leaf {3,8}, size
power {0,.5}, threshold {-1e12,-1,0,1,2,5,1e12}. Outer plot GT is used only when later
reporting its held-out evaluation. Generating caches for all plots does not authorize
using an outer plot's cache in that fold's fitting/selection.

To reproduce the existing 39D **fixed-choice** LAUTx refit instead of rerunning selection,
use `fit-b` with the five other plots' `regions.npz` and the corresponding held-out
plot's supplied config. Inference can use the resulting B by editing a copy of the
config's relative checkpoint paths. Never overwrite the archived reference configs.

Model training and fitting are included, but full neural-network retraining and the
entire six-benchmark training job were not repeated while preparing this release.
