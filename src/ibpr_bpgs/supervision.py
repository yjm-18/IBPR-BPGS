"""A评分器的正式训练监督：预测实例与参考树最大交集匈牙利匹配。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_prediction_to_reference(
    gt_instance: np.ndarray,
    predicted_instance: np.ndarray,
    minimum_gt_id: int = 1,
) -> tuple[np.ndarray, dict[int, int]]:
    """返回逐点匹配参考ID和预测实例到参考实例的一对一映射。

    匹配目标为最大化预测实例与参考实例的交集点数。没有正交集的
    匹配不保留；未匹配预测实例映射为-1。
    """
    valid_gt = gt_instance >= minimum_gt_id
    gt_ids = np.unique(gt_instance[valid_gt])
    predicted_ids = np.unique(predicted_instance[predicted_instance > 0])
    intersections = np.zeros((len(predicted_ids), len(gt_ids)), dtype=np.int64)
    joint = valid_gt & (predicted_instance > 0)
    if np.any(joint):
        pred_index = np.searchsorted(predicted_ids, predicted_instance[joint])
        gt_index = np.searchsorted(gt_ids, gt_instance[joint])
        np.add.at(intersections, (pred_index, gt_index), 1)
    mapping: dict[int, int] = {}
    if intersections.size:
        rows, columns = linear_sum_assignment(intersections, maximize=True)
        mapping = {
            int(predicted_ids[row]): int(gt_ids[column])
            for row, column in zip(rows, columns)
            if intersections[row, column] > 0
        }
    mapped_reference = np.full(len(predicted_instance), -1, dtype=np.int64)
    for predicted_id, reference_id in mapping.items():
        mapped_reference[predicted_instance == predicted_id] = reference_id
    return mapped_reference, mapping


def candidate_supervision(
    gt_instance: np.ndarray,
    predicted_instance: np.ndarray,
    point_ids: np.ndarray,
    candidates: np.ndarray,
    minimum_gt_id: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """生成每个“点—候选实例”的二元监督标签。

    候选预测实例所匹配的参考树等于该点的参考树时为正样本；其他
    候选为负样本。非参考树点和无效候选通过valid_pair排除训练。
    """
    _, mapping = hungarian_prediction_to_reference(
        gt_instance, predicted_instance, minimum_gt_id
    )
    maximum = int(max(predicted_instance.max(initial=0), candidates.max(initial=0)))
    lookup = np.full(maximum + 1, -1, dtype=np.int64)
    for predicted_id, reference_id in mapping.items():
        lookup[predicted_id] = reference_id
    labels = (
        (candidates > 0)
        & (gt_instance[point_ids, None] >= minimum_gt_id)
        & (lookup[candidates] == gt_instance[point_ids, None])
    )
    valid_pair = (candidates > 0) & (gt_instance[point_ids, None] >= minimum_gt_id)
    return labels, valid_pair, mapping
