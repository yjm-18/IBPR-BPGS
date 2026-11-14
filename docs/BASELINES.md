# Baseline Models and Reproduced Metrics

## FOR-instance and LAUTx

The two datasets use the corresponding official pretrained checkpoint below.
TL and SAT are evaluated independently. The numbers are our reproduced baseline
measurements under the reported experiment settings, not values copied from the
original model papers. Obtaining a checkpoint alone does not reproduce the complete
input preparation, point alignment and evaluation pipeline.

| Model | Checkpoint | Download | Official reference |
|---|---|---|---|
| TreeLearn (TL) | `model_weights_with_small_20241213.pth` | [Official weight file](https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/TYZJ4E) | [Pipeline documentation](https://github.com/ecker-lab/TreeLearn/blob/main/docs/segmentation_pipeline.md) |
| SegmentAnyTree (SAT) | `PointGroup-PAPER.pt`, alias `latest` | [Official Git LFS weight file](https://media.githubusercontent.com/media/SmartForest-no/SegmentAnyTree/a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9/model_file/PointGroup-PAPER.pt) | [Pinned checkpoint page](https://github.com/SmartForest-no/SegmentAnyTree/blob/a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9/model_file/PointGroup-PAPER.pt) |

The SAT link points to the binary Git LFS object, rather than its small text pointer.
The FOR-instance SAT local copy changes stored data-root metadata only; its model
tensors match the official checkpoint. File SHA256 values are provided in
[baseline_weights.json](../configs/baseline_weights.json) to distinguish metadata
serialization from model-weight changes.

### Reproduced baseline metrics

| Dataset | Model | P (%) | R (%) | F1 (%) | IoU (%) | mAP (%) |
|---|---|---:|---:|---:|---:|---:|
| FOR-instance | TL | 74.919 | 71.207 | 73.016 | 67.466 | 40.005 |
| FOR-instance | SAT | 84.858 | 83.282 | 84.063 | 76.628 | 48.695 |
| LAUTx | TL | 95.633 | 85.049 | 90.031 | 77.156 | 62.273 |
| LAUTx | SAT | 81.458 | 75.922 | 78.593 | 65.843 | 32.132 |

P/R/F1 use pooled scene TP/FP/FN counts. IoU is the mean best IoU over reference
trees; mAP uses pooled prediction scores under the fixed evaluation protocol.
Both models use the same evaluation point set within each dataset. FOR-instance
removes class=3 before 0.1m voxelization. Full-precision baseline measurements are
available in [baseline_reproduction.csv](../results/baseline_reproduction.csv).

## Ofental

The Ofental baselines are reproduced through standard model training and evaluation
on the designated data split. Their baseline and full-method results appear in the
main result table. No separate Ofental pretrained-weight download or additional
baseline-reproduction table is provided here.
