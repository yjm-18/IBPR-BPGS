"""正式A模块：固定13维候选描述和39维关系向量。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from scipy.spatial import cKDTree


FEATURE_NAMES = (
    "is_current",
    "boundary",
    "tree_probability",
    "center_xy",
    "center_dz",
    "center_distance",
    "radial_normalized",
    "embedding_distance",
    "shifted_center_distance",
    "log_instance_size",
    "local_fraction",
    "local_weight_fraction",
    "local_min_distance",
)
FEATURE_DIM = 13
RELATION_DIM = 39
CANDIDATE_SLOTS = 4


def _means(values: np.ndarray, labels: np.ndarray, count: np.ndarray) -> np.ndarray:
    output = np.zeros((len(count), values.shape[1]), dtype=np.float32)
    safe = np.maximum(count, 1)
    for axis in range(values.shape[1]):
        output[:, axis] = (
            np.bincount(labels, weights=values[:, axis], minlength=len(count)) / safe
        )
    return output


def instance_statistics(scene: Any) -> dict[str, np.ndarray]:
    """只用当前预测的有效树点计算候选实例统计。"""
    mask = scene.predicted_tree & (scene.pred > 0)
    labels = scene.pred[mask].astype(np.int32, copy=False)
    size = int(scene.pred.max(initial=0)) + 1
    count = np.bincount(labels, minlength=size).astype(np.float32)
    pos = scene.pos[mask]
    center = _means(pos, labels, count)
    shifted_center = _means(pos + scene.offset[mask], labels, count)
    embedding_center = _means(scene.embedding[mask], labels, count)
    horizontal_sq = np.sum((pos[:, :2] - center[labels, :2]) ** 2, axis=1)
    radial = np.sqrt(
        np.bincount(labels, weights=horizontal_sq, minlength=size)
        / np.maximum(count, 1)
    ).astype(np.float32)
    radial = np.maximum(radial, 0.25)
    return {
        "count": count,
        "center": center,
        "shifted_center": shifted_center,
        "embedding_center": embedding_center,
        "radial": radial,
    }


@numba.njit(cache=True)
def _local_candidates(current, labels, distances):
    """槽位0为当前实例；其余槽位按逆距离加权支持选择竞争实例。"""
    n = len(current)
    candidates = np.zeros((n, CANDIDATE_SLOTS), np.int32)
    evidence = np.zeros((n, CANDIDATE_SLOTS, 3), np.float32)
    for i in range(n):
        candidates[i, 0] = current[i]
        unique = np.zeros(33, np.int32)
        weighted = np.zeros(33)
        counts = np.zeros(33)
        nearest = np.full(33, 5.0)
        used = 1
        unique[0] = current[i]
        total_weight = 0.0
        total_count = 0
        for j in range(labels.shape[1]):
            distance = distances[i, j]
            if distance > 0.5 or distance < 1e-7:
                continue
            label = labels[i, j]
            k = 0
            while k < used and unique[k] != label:
                k += 1
            if k == used:
                unique[k] = label
                used += 1
            weight = 1.0 / (0.02 + distance)
            weighted[k] += weight
            counts[k] += 1
            nearest[k] = min(nearest[k], distance)
            total_weight += weight
            total_count += 1
        selected = np.zeros(33, np.bool_)
        selected[0] = True
        for slot in range(CANDIDATE_SLOTS):
            best = 0
            if slot > 0:
                best = -1
                best_weight = 0.0
                for k in range(1, used):
                    if not selected[k] and weighted[k] > best_weight:
                        best = k
                        best_weight = weighted[k]
                if best < 0:
                    continue
                selected[best] = True
                candidates[i, slot] = unique[best]
            evidence[i, slot, 0] = counts[best] / max(total_count, 1)
            evidence[i, slot, 1] = weighted[best] / max(total_weight, 1e-8)
            evidence[i, slot, 2] = nearest[best]
    return candidates, evidence


@dataclass
class AFeatureContext:
    scene: Any

    def __post_init__(self) -> None:
        self.pool = np.flatnonzero(self.scene.predicted_tree & (self.scene.pred > 0))
        if len(self.pool) == 0:
            raise ValueError("场景中没有有效预测树点")
        self.tree = cKDTree(self.scene.pos[self.pool])
        self.stats = instance_statistics(self.scene)

    def proposals(self, point_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = min(32, len(self.pool))
        distances, neighbors = self.tree.query(
            self.scene.pos[point_ids], k=k, workers=-1
        )
        if k == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        return _local_candidates(
            self.scene.pred[point_ids],
            self.scene.pred[self.pool[neighbors]],
            distances,
        )

    def features_13d(
        self,
        point_ids: np.ndarray,
        candidates: np.ndarray,
        evidence: np.ndarray,
    ) -> np.ndarray:
        """返回形状[N,4,13]，维度顺序由FEATURE_NAMES固定。"""
        scene = self.scene
        stats = self.stats
        position = scene.pos[point_ids]
        embedding = scene.embedding[point_ids]
        shifted = position + scene.offset[point_ids]
        output = np.zeros(
            (len(point_ids), CANDIDATE_SLOTS, FEATURE_DIM), dtype=np.float32
        )
        for slot in range(CANDIDATE_SLOTS):
            labels = candidates[:, slot]
            center = stats["center"][labels]
            delta = position - center
            center_xy = np.linalg.norm(delta[:, :2], axis=1)
            output[:, slot] = np.column_stack(
                (
                    labels == scene.pred[point_ids],
                    scene.boundary_score[point_ids],
                    scene.tree_probability[point_ids],
                    center_xy,
                    np.abs(delta[:, 2]),
                    np.linalg.norm(delta, axis=1),
                    center_xy / stats["radial"][labels],
                    np.linalg.norm(
                        embedding - stats["embedding_center"][labels], axis=1
                    ),
                    np.linalg.norm(shifted - stats["shifted_center"][labels], axis=1),
                    np.log1p(stats["count"][labels]),
                    evidence[:, slot, 0],
                    evidence[:, slot, 1],
                    evidence[:, slot, 2],
                )
            )
        if output.shape[-1] != FEATURE_DIM or not np.isfinite(output).all():
            raise ValueError("13维特征形状错误或包含非有限值")
        return output


def relationship_39d(features_13d: np.ndarray) -> np.ndarray:
    """拼接[候选13维, 当前13维, 候选-当前13维]，返回[N,4,39]。"""
    if features_13d.ndim != 3 or features_13d.shape[1:] != (
        CANDIDATE_SLOTS,
        FEATURE_DIM,
    ):
        raise ValueError(f"期望[N,4,13]，实际为{features_13d.shape}")
    current = np.broadcast_to(features_13d[:, 0:1, :], features_13d.shape)
    output = np.concatenate(
        (features_13d, current, features_13d - current), axis=2
    ).astype(np.float32, copy=False)
    if output.shape[-1] != RELATION_DIM:
        raise AssertionError("关系向量必须为39维")
    return output
