"""Convert explicitly aligned base-model exports into the public scene contract.
No implicit nearest-neighbour search or label reindexing is performed here.
"""

import argparse
import numpy as np
from ibpr_bpgs.io import normalize_tl_features, load_scene

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", choices=["TL", "SAT"], required=True)
p.add_argument(
    "--input",
    required=True,
    help="NPZ with common fields plus TL raw backbone or SAT embedding",
)
p.add_argument("--output", required=True)
a = p.parse_args()
with np.load(a.input, allow_pickle=False) as z:
    d = dict(z)
if a.model == "TL":
    if "backbone" not in d or "nn" not in d:
        raise ValueError("TL needs raw backbone (N,32) and frozen 5NN indices")
    d["embedding"] = normalize_tl_features(d.pop("backbone"), d["pred"])
    d["predicted_tree"] = d["pred"] > 0
    d["prediction_count"] = np.ones(len(d["pred"]), np.float32)
else:
    if "pred_semantic" not in d:
        raise ValueError("SAT requires aligned semantic labels (tree=1)")
    d["predicted_tree"] = (d.pop("pred_semantic") == 1) & (d["pred"] > 0)
    if "nn" in d:
        raise ValueError(
            "SAT correction uses direct aligned output, not TL propagation"
        )
np.savez_compressed(a.output, **d)
load_scene(a.output)
print(a.output)
