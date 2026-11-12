# Candidate protocol and development provenance

This release freezes the 2026-11-07 39D candidate replay. IBPR removes prediction-count
features from all three blocks (old 42D columns 3,17,31), preserving all other ordering.
BPGS remains 38D, including prediction-count statistics. The two models have independent
scorers. Four final A models and sixteen final B models are distributed.

| Dataset | A fitting / selection | B fitting / selection |
|---|---|---|
| FOR-instance | 13 training plots / 8 validation plots | Same 13 via five plot-OOF A fits / same 8 validation |
| Ofental | Training region / validation region | Five spatial A-OOF folds in training region / validation region |
| LAUTx | Fixed corresponding FOR-instance A; no LAUTx A fitting or anchor | Six outer folds; other five plots use inner 4-train/1-validation; refit on five |

Test scales: FOR-instance 9,236,696 points / 323 trees; Ofental 2,474,717 points / 119
trees; LAUTx 23,016,952 points / 515 trees. See `protocols/` for ordered split records.

All final A thresholds are score>=0.4 and advantage>0.02. Seeds use score>=0.5 and
advantage>0.1. Regions use 0.25m 26-neighbour connected voxels within source/destination
pairs; no additional seed-count gate. B accepts predicted marginal gain >= its locked
threshold. Ofental's selected -1e12 accepts all regions: its A and AB outputs coincide.
Do not alter that threshold to manufacture an improvement.

Matching rules differ deliberately: A training maximizes intersection and discards
zero-intersection matches; B targets and C/H/G freeze baseline maximum-IoU matching
without a 0.5 cutoff. PRF uses Hungarian maximum-total-IoU matching, strict IoU>0.5,
and counts an unmatched prediction as FP only when >=50% of its points overlap the
reference-labelled region. mAP uses frozen source-baseline instance confidence,
per-scene greedy matching at IoU>=threshold, pooled ranking, and 101 recall samples,
for thresholds 0.50:0.05:0.95. AP does not apply the PRF FP-overlap filter.

# Auxiliary boundary provenance

One boundary predictor is fitted on FOR-instance 13 training plots and selected on
8 validation plots. Labels are binary, using cross-instance KNN without a radius cap.
Fixed K: CULS128, NIBIO192, RMIT128, SCION256, TUWIEN160. Historical K selection for the
last three sources used FOR-instance test reference labels to reach a coverage target.
Thus direct A/B fold exclusion must not be described as proof that the entire development
pipeline never used test information. Baselines also have historical test tuning.
This release preserves that provenance rather than silently claiming a train-only K search.

Boundary checkpoint selection pools up to 5,000 equally spaced valid points per
validation batch for AP; it is not full-point validation AP. Selected epoch is 97 of
100. A/B use probabilities from this frozen predictor, not test GT boundary labels.

# Version identity

The FOR SAT 39D AB reference is G=31,930, NIR=32.5949367%, IoU=76.9762146%,
mAP=49.8351459%. Do not substitute the older 42D G=31,988 result. LAUTx SAT uses
baseline 65.8426977% IoU / 32.1320345% mAP, and AB G=151,165. All reported arms are
recorded in `results/`; exploratory high-IoU plot variants are not part of this release.
