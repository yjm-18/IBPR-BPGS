# Release Consistency

The release contains the candidate implementation and selected A/B configurations.
The published results contain 14 aggregate rows. Per-scene result files are not distributed.
FOR-instance and Ofental include baseline and +A+B; LAUTx includes all three arms.

All retained metric values are copied exactly from the selected candidate measurements.
The original implementation checks covered proposal scores, region targets, source and
evaluation labels, and metric computation. Source tests remain available under `tests/`.
No new scientific experiment, checkpoint fitting or parameter search is introduced by
this documentation revision.
