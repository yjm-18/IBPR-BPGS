# IBPR + BPGS

Candidate research release **0.1.0rc1 (2026-11-09)** for local tree-instance label correction.
The source release corresponds to the **39-dimensional IBPR** candidate experiments on
FOR-instance, Ofental and LAUTx with TreeLearn (TL) and SegmentAnyTree (SAT).

IBPR scores point-to-instance relations; BPGS predicts the marginal net correction of
spatial proposal regions. Baseline predictions remain frozen. TL and SAT use separate
scorers. This is the method release, starting from aligned base-model exports; it is
not a bundled distribution of the two upstream segmentation frameworks or their datasets.

## Install

Use Python 3.11.7 for the core package. The checkpoint-compatible environment is pinned.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

Auxiliary boundary training/inference requires a separate CUDA environment; see
[auxiliary boundary instructions](auxiliary/boundary/README.md).

## Checkpoints

Extract `ibpr-bpgs-0.1.0rc1-checkpoints.zip` **inside this repository**. It creates
`checkpoints/`. The archive contains four final A models, sixteen B models, and the
shared auxiliary boundary checkpoint. LAUTx reuses the FOR-instance A model and has
one B model per held-out plot. Checkpoints are excluded from normal Git commits.

```bash
ibpr-bpgs verify-weights --weights checkpoints
```

Load pickle/PyTorch files only from a trusted source. SHA256 and byte sizes are in
[configs/checkpoints.json](configs/checkpoints.json). No external download URL is
invented; the repository owner can attach the supplied weight archive when publishing.

## Prepare inputs and infer

Read [the point/feature contract](docs/DATA_FORMAT.md) first. Prepared NPZ files retain
the exact frozen baseline, feature preprocessing and TL 5NN mapping. Ground truth is
not required for inference. A point cloud alone is not sufficient: base-model outputs
and auxiliary boundary probabilities must first be exported.

```bash
python scripts/prepare_scene.py --model TL --input raw_tl_export.npz --output scene.npz
ibpr-bpgs infer --scene scene.npz --config configs/FOR-instance_TL.json \
  --weights checkpoints --output outputs/example --evaluate
```

For SAT use `--model SAT` and its corresponding config. Omit `--evaluate` for unlabelled
inputs. Outputs include baseline / A / AB source and evaluation labels, proposal scores,
region scores/acceptance, and optional evaluation metrics.

Full benchmark manifests preserve scene order, which also determines stable AP tie order:

```bash
python scripts/benchmark.py --manifest configs/manifests/LAUTx_SAT_test.json \
  --weights checkpoints --output outputs/LAUTx_SAT
```

Place independently obtained prepared scenes at the manifest paths. The archives do not
contain point clouds, upstream weights or multi-gigabyte prediction caches.

## Training and protocol

See [training commands](docs/TRAINING.md), [experiment protocol](docs/PROTOCOL.md),
and the selected A/B configurations under `configs/`. `reproduce-fixed` trains with
the supplied fixed parameters. `nested-b` implements LAUTx inner/outer
leave-one-plot-out training and selection.

## Results and validation

[results/comparison.csv](results/comparison.csv) contains 14 dataset/model/arm summaries: baseline and +A+B for FOR-instance and Ofental; baseline, +A and +A+B for LAUTx.
These are reference outputs, not a claim that new upstream inference is bitwise identical.
IoU, mAP, PRF and NIR in aggregate reports are percentages. G is a point count.

[Release validation](docs/VALIDATION.md) states which paths were actually executed.
The repository includes tests of 5NN ties, strict thresholds, region connectivity,
39D features, marginal target replay and empty predictions.

## Publication metadata

The owner has not yet supplied a public repository URL, author list, DOI or licence choice.
These are not fabricated. No broad software licence is granted by this draft;
see [rights and dependencies](NOTICE.md). Add the owner-approved licence and citation
metadata before advertising this as an openly licensed release.
