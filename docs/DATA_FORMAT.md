# Prepared scene contract

All geometry is in metres. Arrays must be finite and retain their original point order
and numerical precision; do not downcast coordinates during packaging.
NPZ keys (N source points, M evaluation points):

| Key | Shape | Meaning |
|---|---|---|
| pos | N×3 | Source coordinates |
| pred | N integer | Frozen instance IDs; positive tree IDs, 0 background |
| predicted_tree | N bool | TL: pred>0; SAT: tree semantic class AND pred>0 |
| offset | N×3 | Network-predicted offset |
| embedding | N×32 TL / N×5 SAT | Prepared feature representation |
| tree_probability | N | Frozen tree confidence |
| prediction_count | N | SAT overlap coverage; TL all ones; retained for BPGS only |
| boundary_score | N | Aligned probability from shared auxiliary predictor |
| nn | M×5 integer, TL only | Frozen evaluation-point to source-point neighbour indices |
| gt | M integer, optional | Reference IDs >0; non-tree/background <=0 |
| source_gt | N integer, training only | Reference labels aligned to source points |

For SAT, N=M and `nn` is absent. For TL, inference changes source labels then applies
5NN majority voting (smaller ID wins ties; label 0 participates). Do not recompute the
neighbours with a different point order. No GT field is used in `pipeline.predict`.

TL input to `scripts/prepare_scene.py` supplies `backbone` instead of `embedding`.
It performs exactly:

    sample = backbone[flatnonzero(pred > 0)[::100]]
    embedding = ((backbone - sample.mean(0)) / maximum(sample.std(0), .25)
                 * sqrt(5/32) * .8).astype(float32)

Do not apply this again to an already prepared scene. SAT supplies the native 5D
`embedding`, aligned `pred_semantic` (tree class 1), and `prediction_count` after
continuous window aggregation. Raw SAT instance IDs with background -1 must be
converted consistently to the positive-ID/zero-background contract before this adapter.
Do not relabel only one of the baseline, features or cached neighbour arrays.

Both models are evaluated on the same 0.1m point set. FOR-instance removes class=3
**before** voxelization. Native network representations may differ; the public scene
contract starts after correspondence alignment. In particular, LAUTx SAT features and
predictions are mapped from the native output to the common evaluation points.
`prepare_scene.py` does not infer this mapping from inconsistent inputs.

Training manifest `path` is relative to the manifest file. Test manifest `config` is
relative to the repository working directory. Use the provided ordered manifests.
Raw datasets and upstream segmentation checkpoints must be obtained separately under
their own terms; this package does not redistribute them.

## Dataset-specific export details

Keep the coordinate frame used by the frozen scene; region voxel boundaries are
frame-dependent. FOR TL uses its saved `coords`; LAUTx TL and Ofental TL use
`(coords.astype(float64) - coords.astype(float64).mean(0)).astype(float32)` for
A/B `pos`. Their neighbour/correspondence arrays remain those established before
this scene preparation. Do not translate an already prepared scene a second time.
SAT uses the corresponding aligned saved `pos`/common-point coordinates.

The Ofental SAT reference export uses labels <=1 as non-tree: convert them to 0
before the public `gt>0` metric convention. Positive tree IDs may remain nonconsecutive.
Raw GT sentinel conventions are not automatically inferred by the core loader.
Baseline weights and reproduced metrics are documented in [BASELINES.md](BASELINES.md).
