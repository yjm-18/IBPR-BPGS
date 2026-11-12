"""Frozen-baseline IBPR/BPGS inference; no GT used to choose edits."""

import numpy as np
from .scorer import score_scene
from .regions import region_data, apply
from .metrics import propagate, evaluate, pooled_ap, THRESHOLDS


def predict(scene, nn, a_model, b_model, config, chunk_size=50000):
    if a_model.n_features_in_ != 39 or b_model.n_features_in_ != 38:
        raise ValueError("Release requires 39D IBPR and 38D BPGS checkpoints")
    scores = score_scene(scene, a_model, chunk_size=chunk_size)
    regions = region_data(scene, scores, config["A"])
    u = (
        b_model.predict(regions["features"])
        if len(regions["components"])
        else np.empty(0)
    )
    accepted = u >= config["B"]["threshold"]
    source = {
        "baseline": scene.pred.copy(),
        "A": apply(scene, regions, np.ones(len(u), bool)),
        "AB": apply(scene, regions, accepted),
    }
    evaluation = {
        k: propagate(v, nn) if nn is not None else v.copy() for k, v in source.items()
    }
    return source, evaluation, scores, regions, u, accepted


def aggregate(rows, entries):
    out = {
        k: sum(r[k] for r in rows)
        for k in ["GT", "TP", "FP", "FN", "C", "H", "G", "modified_points"]
    }
    n = out["GT"]
    tp = out["TP"]
    fp = out["FP"]
    fn = out["FN"]
    d = out["modified_points"]
    out.update(
        P=100 * tp / max(tp + fp, 1),
        R=100 * tp / max(n, 1),
        F1=200 * tp / max(2 * tp + fp + fn, 1),
        IoU=100 * sum(r["IoU"] * r["GT"] for r in rows) / max(n, 1),
        mAP=100 * float(np.mean([pooled_ap(entries, t) for t in THRESHOLDS])),
        NIR=100 * out["G"] / d if d else None,
    )
    return out
