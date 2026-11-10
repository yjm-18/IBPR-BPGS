# IBPR-BPGS

Public preview of the IBPR-BPGS research project. The manuscript is in preparation.
This repository provides selected aggregate candidate results and utilities for viewing
them. The implementation and pretrained models are not released at this stage.

## Available Materials

| Path | Contents |
|---|---|
| `results/aggregate_metrics.csv` | Three datasets, two baseline models, and three experiment arms |
| `results/aggregate_metrics.json` | The same metrics with explicit units |
| `results/baseline_prf.csv` | Aggregate baseline precision, recall, and F1 |
| `results/evaluation_scale.csv` | Evaluation point and reference-tree counts |
| `configs/` | Dataset/model selection and formatting for public result reports |
| `tools/render_results.py` | Standalone viewer for the released aggregate results |
| `docs/` | Result interpretation, configuration scope, and availability information |

These are candidate measurements, not a final publication claim. Values are preserved
from the current experiment snapshot, including ties and decreases between arms.

## View Results

The reporting utility uses the Python standard library and does not load model weights.
Run it from the repository root:

```bash
python tools/render_results.py --config configs/LAUTx_SAT.json
python tools/render_results.py --config configs/FOR-instance_TL.json --output report.md
```

See [the result tables](results/RESULTS.md) and [metric notes](docs/RESULTS.md).
The supplied configurations are **report configurations**, not runnable method
configurations. They do not contain model parameters or acceptance thresholds.

## Availability

Core implementation, training procedures, model features, selected method parameters,
per-scene records, and checkpoints remain unavailable. This preview is insufficient
to reproduce the method. Further release information will be posted here when available.

See [release scope](docs/AVAILABILITY.md). No public software licence or paper citation
has been assigned in this preview.
