# Experiment Protocol

This package contains the released method with 39-dimensional IBPR and
38-dimensional BPGS. TL and SAT use separate scorers. Four final A models and sixteen
final B models are distributed in the unchanged checkpoint archive.

## IBPR and BPGS training and evaluation

For IBPR and BPGS, training data are used for fitting, validation data for parameter
selection, and the corresponding held-out test data only for evaluation. This statement
is scoped to these two modules. Base segmentations and the auxiliary boundary predictor
are fixed inputs to the A/B experiments.

| Dataset | IBPR fitting / selection | BPGS fitting / selection |
|---|---|---|
| FOR-instance | 13 training plots / 8 validation plots | Same 13 with five plot-OOF A fits / same 8 validation plots |
| Ofental | Training region / validation region | Five spatial A-OOF folds in the training region / validation region |
| LAUTx | Fixed corresponding FOR-instance A; no LAUTx A fitting or anchor | Six outer folds; other five plots use inner 4-train/1-validation; refit on five |

For LAUTx, exclusion is defined per outer fold: the held-out plot does not participate
in that fold's B fitting or parameter selection. Final selected configurations are
provided under `configs/`; the current fitted models use these locked choices.

## Implementation

A suggestions use score >= 0.4 and advantage > 0.02. Seeds use score >= 0.5 and
advantage > 0.1. Regions are 0.25m 26-neighbour voxel components within source/destination
pairs, with no additional seed-count gate. B accepts predicted marginal gain at or
above its configured threshold. TL includes the same frozen 5NN propagation in B target
construction and final output; SAT outputs labels directly on aligned evaluation points.

TL's 32-dimensional backbone features undergo per-channel standardization and fixed
scaling before relation distances are calculated. SAT retains native 5-dimensional
embeddings. The transformation and point contract are specified in [DATA_FORMAT.md](DATA_FORMAT.md).

## Evaluation and reporting

Test scales: FOR-instance 9,236,696 points / 323 reference trees; Ofental 2,474,717 /
119; LAUTx 23,016,952 / 515. The result files contain baseline and +A+B for FOR-instance and Ofental,
and baseline, +A and +A+B for LAUTx. All retained metric values are unchanged.

A training uses maximum-intersection Hungarian matching, discarding zero-intersection
matches. B targets and C/H/G freeze baseline maximum-IoU matching without a 0.5 cutoff.
Detection PRF uses maximum-total-IoU Hungarian matching with strict IoU > 0.5 and counts
an unmatched prediction as FP only when at least 50% of its points lie in the reference
labelled region. AP uses fixed baseline instance confidence, greedy matching at each
IoU threshold, pooled ranking and 101 recall samples, averaged over 0.50:0.05:0.95.
AP does not apply the PRF FP-overlap filter.

Auxiliary checkpoint selection uses sampled-point boundary AP as described in
[BOUNDARY_VALIDATION.md](BOUNDARY_VALIDATION.md). Software versions are separated by
component in [ENVIRONMENTS.md](ENVIRONMENTS.md). Baseline checkpoint sources and our
FOR-instance/LAUTx reproduced metrics are listed in [BASELINES.md](BASELINES.md).
