"""Portable source-point contract. Reference labels are optional for inference."""

from pathlib import Path
from types import SimpleNamespace
import json
import pickle
import numpy as np

FIELDS = (
    "pos",
    "pred",
    "predicted_tree",
    "offset",
    "embedding",
    "tree_probability",
    "prediction_count",
    "boundary_score",
)


def load_scene(path):
    with np.load(path, allow_pickle=False) as z:
        scene = SimpleNamespace(**{k: z[k] for k in FIELDS})
        gt = z["gt"] if "gt" in z else None
        nn = z["nn"] if "nn" in z else None
        source_gt = z["source_gt"] if "source_gt" in z else (gt if nn is None else None)
    n = len(scene.pred)
    for k in FIELDS:
        v = getattr(scene, k)
        if len(v) != n or not np.isfinite(v).all():
            raise ValueError(f"{k}: invalid length or nonfinite values")
    for k in ("pos", "offset"):
        if getattr(scene, k).shape != (n, 3):
            raise ValueError(f"{k} must have shape (N,3)")
    for k in (
        "pred",
        "predicted_tree",
        "tree_probability",
        "prediction_count",
        "boundary_score",
    ):
        if getattr(scene, k).shape != (n,):
            raise ValueError(f"{k} must have shape (N,)")
    if scene.pred.dtype.kind not in "iu" or np.any(scene.pred < 0):
        raise ValueError("pred must be nonnegative integer labels; 0 is background")
    if scene.predicted_tree.dtype != np.bool_:
        raise ValueError("predicted_tree must be boolean")
    if scene.embedding.ndim != 2:
        raise ValueError("embedding must have shape (N,D)")
    if nn is not None:
        if nn.ndim != 2 or nn.shape[1] != 5 or nn.dtype.kind not in "iu":
            raise ValueError("TL nn must be integer evaluation-to-source indices (M,5)")
        if np.any(nn < 0) or np.any(nn >= n):
            raise ValueError("nn indices outside source points")
    m = n if nn is None else len(nn)
    if gt is not None and (gt.shape != (m,) or gt.dtype.kind not in "iu"):
        raise ValueError(
            "gt must contain one integer reference label per evaluation point"
        )
    if source_gt is not None and (
        source_gt.shape != (n,) or source_gt.dtype.kind not in "iu"
    ):
        raise ValueError("source_gt must match source points")
    return scene, gt, nn, source_gt


def load_model(path):
    # Pickle is a code-executing format. Load only trusted release/training outputs.
    with Path(path).open("rb") as f:
        return pickle.load(f)


def dump_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if hasattr(x, "item") else x.tolist(),
        )
        + "\n"
    )


def normalize_tl_features(features, pred):
    """Apply once, before writing the scene. Do not renormalize prepared scenes."""
    if features.ndim != 2 or features.shape[1] != 32:
        raise ValueError("Expected 32-dimensional TL backbone features")
    sample = features[np.flatnonzero(pred > 0)[::100]]
    if not len(sample):
        raise ValueError("No valid instance points for TL normalization statistics")
    return (
        (features - sample.mean(0))
        / np.maximum(sample.std(0), 0.25)
        * np.sqrt(5 / 32)
        * 0.8
    ).astype(np.float32)
