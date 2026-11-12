# Release verification (2026-11-09)

Executed in the recorded core environment:

- Eight unit/smoke tests passed: removed coverage dimension, 5NN tie/background handling,
  source/destination region isolation, strict suggestion margin, PRF/AP threshold distinction,
  exact marginal targets after 5NN, empty predictions, classifier/regressor fitting and inference.
- Six real dataset/model checks passed: FOR-instance CULS plot2 for TL/SAT,
  Ofental test for TL/SAT, LAUTx p1 for TL/SAT. For all six, baseline/A/AB source and
  evaluation labels matched the current 39D candidate arrays exactly, as did C/H/G/D,
  TP/FP/FN, IoU and mAP. Up to 4,096 candidate scores per scene were compared exactly.
  Real-label replay used the archived score arrays; it was not full neural inference.
- LAUTx p1 TL/SAT 38D region features and exact marginal target arrays matched the
  original experiment arrays exactly.
- Additional **full CLI inference** on FOR-instance CULS plot2 for TL/SAT independently
  regenerated every candidate score, regions, labels and metrics. All candidate score
  arrays and all three output label arms matched the archived 39D outputs exactly.
- A Python wheel was built locally without fetching dependencies; isolated source-archive
  extraction was tested for imports, command help and the eight tests.
- Archive hashes, checkpoint entries, syntax, and local-path/secret-pattern scans were checked.

Machine-readable evidence: `real_validation.json`, `cli_validation.json`,
`release_checks.json`. Null marginal-target flags mean that target comparison was not
run for that scene, not a failed check.

Not executed during release preparation: full A/B retraining over every plot, a fresh
nested hyperparameter search, upstream TL/SAT neural inference/training, or auxiliary
boundary retraining. Selected weights and complete benchmark summary/per-scene reference
metrics are included. The published source is designed to consume the prepared scene
contract; no claim is made that arbitrary upstream re-exports are bitwise identical.
