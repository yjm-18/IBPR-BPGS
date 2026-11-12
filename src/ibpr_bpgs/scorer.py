from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
from .features import AFeatureContext, RELATION_DIM, relationship_39d


def score_scene(
    scene: Any, model: HistGradientBoostingClassifier, chunk_size: int = 100000
) -> dict[str, np.ndarray]:
    """输出点ID、4个候选实例ID及其归属正确概率。"""
    if not np.any(scene.predicted_tree & (scene.pred > 0)):
        return {
            "point_ids": np.empty(0, np.int64),
            "candidates": np.empty((0, 4), np.int32),
            "scores": np.empty((0, 4), np.float32),
        }
    context = AFeatureContext(scene)
    records = []
    for start in range(0, len(context.pool), chunk_size):
        point_ids = context.pool[start : start + chunk_size]
        candidates, evidence = context.proposals(point_ids)
        keep = candidates[:, 1] > 0
        point_ids = point_ids[keep]
        candidates = candidates[keep]
        evidence = evidence[keep]
        if not len(point_ids):
            continue
        features = context.features_13d(point_ids, candidates, evidence)
        relation = relationship_39d(features)
        flat = relation.reshape(-1, RELATION_DIM)
        with threadpool_limits(limits=8):
            probability = model.predict_proba(flat)[:, 1]
        probability = probability.reshape(len(point_ids), 4).astype(np.float32)
        probability[candidates == 0] = -1.0
        records.append((point_ids, candidates, probability))
    if not records:
        return {
            "point_ids": np.empty(0, dtype=np.int64),
            "candidates": np.empty((0, 4), dtype=np.int32),
            "scores": np.empty((0, 4), dtype=np.float32),
        }
    return {
        "point_ids": np.concatenate([row[0] for row in records]),
        "candidates": np.concatenate([row[1] for row in records]),
        "scores": np.concatenate([row[2] for row in records]),
    }
