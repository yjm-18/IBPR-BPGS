"""A评分后的最佳竞争实例选择、建议筛选和同步标签写回。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SuggestionParameters:
    target_score_minimum: float = 0.40
    score_advantage_minimum: float = 0.02
    seed_target_score_minimum: float = 0.50
    seed_score_advantage_minimum: float = 0.10


def select_best_competitor(
    point_ids: np.ndarray,
    current_instance: np.ndarray,
    candidates: np.ndarray,
    scores: np.ndarray,
) -> dict[str, np.ndarray]:
    """在槽位1—3选择得分最高的竞争实例。

    np.argmax在并列时返回最靠前槽位；候选槽位本身已按局部逆距离
    加权支持排序，因此并列处理是确定性的。
    """
    if candidates.shape != scores.shape or candidates.shape[1] != 4:
        raise ValueError("candidates和scores必须具有相同的[N,4]形状")
    row = np.arange(len(point_ids))
    competitor_slot = 1 + np.argmax(scores[:, 1:], axis=1)
    target_instance = candidates[row, competitor_slot]
    target_score = scores[row, competitor_slot]
    current_score = scores[:, 0]
    advantage = target_score - current_score
    valid = (target_instance > 0) & (target_instance != current_instance[point_ids])
    return {
        "point_ids": point_ids,
        "target_instance": target_instance,
        "target_score": target_score,
        "current_score": current_score,
        "score_advantage": advantage,
        "valid": valid,
    }


def filter_suggestions(
    proposal: dict[str, np.ndarray],
    parameters: SuggestionParameters = SuggestionParameters(),
) -> dict[str, np.ndarray]:
    """生成区域构造候选池与高置信种子，不读取GT。"""
    pool = (
        proposal["valid"]
        & (proposal["target_score"] >= parameters.target_score_minimum)
        & (proposal["score_advantage"] > parameters.score_advantage_minimum)
    )
    seed = (
        proposal["valid"]
        & (proposal["target_score"] >= parameters.seed_target_score_minimum)
        & (proposal["score_advantage"] > parameters.seed_score_advantage_minimum)
    )
    return {**proposal, "pool": pool, "seed": seed}


def apply_synchronously(
    baseline_instances: np.ndarray,
    point_ids: np.ndarray,
    target_instances: np.ndarray,
    accepted: np.ndarray,
) -> np.ndarray:
    """所有建议基于冻结基线一次生成，随后一次性同步写回。"""
    output = baseline_instances.copy()
    output[point_ids[accepted]] = target_instances[accepted]
    return output
