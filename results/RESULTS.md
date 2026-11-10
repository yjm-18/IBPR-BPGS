# Candidate Aggregate Results

Percentages are on the 0-100 scale. A dash denotes undefined NIR.

| Dataset | Model | Arm | G | NIR (%) | IoU (%) | mAP (%) |
|---|---|---|---:|---:|---:|---:|
| FOR-instance | TL | baseline | 0 | — | 67.466 | 40.005 |
| FOR-instance | TL | A | 67,078 | 47.143 | 68.631 | 42.089 |
| FOR-instance | TL | AB | 66,959 | 51.835 | 68.601 | 41.876 |
| FOR-instance | SAT | baseline | 0 | — | 76.628 | 48.695 |
| FOR-instance | SAT | A | 31,894 | 31.642 | 76.976 | 49.918 |
| FOR-instance | SAT | AB | 31,930 | 32.595 | 76.976 | 49.835 |
| Ofental | TL | baseline | 0 | — | 48.160 | 15.983 |
| Ofental | TL | A | 21,430 | 30.846 | 49.287 | 17.350 |
| Ofental | TL | AB | 21,430 | 30.846 | 49.287 | 17.350 |
| Ofental | SAT | baseline | 0 | — | 51.861 | 16.769 |
| Ofental | SAT | A | 33,783 | 36.979 | 53.451 | 19.696 |
| Ofental | SAT | AB | 33,783 | 36.979 | 53.451 | 19.696 |
| LAUTx | TL | baseline | 0 | — | 77.156 | 62.273 |
| LAUTx | TL | A | 21,673 | 23.092 | 77.536 | 63.042 |
| LAUTx | TL | AB | 26,381 | 39.016 | 77.586 | 63.070 |
| LAUTx | SAT | baseline | 0 | — | 65.843 | 32.132 |
| LAUTx | SAT | A | 148,186 | 40.557 | 66.465 | 33.105 |
| LAUTx | SAT | AB | 151,165 | 44.109 | 66.489 | 33.226 |

## Baseline Detection Metrics

| Dataset | Model | P (%) | R (%) | F1 (%) |
|---|---|---:|---:|---:|
| FOR-instance | TL | 74.919 | 71.207 | 73.016 |
| FOR-instance | SAT | 84.858 | 83.282 | 84.062 |
| Ofental | TL | 73.913 | 57.143 | 64.455 |
| Ofental | SAT | 73.958 | 59.664 | 66.047 |
| LAUTx | TL | 95.633 | 85.049 | 90.031 |
| LAUTx | SAT | 81.458 | 75.922 | 78.593 |

See [metric notes](../docs/RESULTS.md) for definitions and limitations.
