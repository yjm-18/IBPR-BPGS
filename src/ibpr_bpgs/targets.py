import numpy as np
from scipy.sparse import csr_matrix
from .metrics import lookup_for, gain


def mode(values):
    if values.shape[1] == 1:
        return values[:, 0].copy()
    values = np.sort(values, axis=1)
    best = values[:, 0].copy()
    count = np.zeros(len(values), np.int8)
    for j in range(values.shape[1]):
        c = np.sum(values == values[:, j, None], axis=1)
        take = c > count
        best[take] = values[take, j]
        count[take] = c[take]
    return best


def compact_cache(scene, gt, nn, base, d):
    member = np.full(len(scene.pred), -1, np.int32)
    target = scene.pred.copy()
    chosen = d["chosen"]
    member[d["ids"][chosen]] = d["labels"]
    target[d["ids"][chosen]] = d["dest"][chosen]
    assert np.array_equal(d["components"], np.arange(len(d["components"])))
    if nn is None:
        affected = np.flatnonzero(member >= 0)
        near = affected[:, None]
    else:
        parts = []
        for i in range(0, len(nn), 100000):
            parts.append(
                i + np.flatnonzero(np.any(member[nn[i : i + 100000]] >= 0, axis=1))
            )
        affected = np.concatenate(parts)
        near = nn[affected]
    return dict(
        affected=affected,
        gt=gt[affected],
        baseline=base[affected],
        lookup=lookup_for(gt, base),
        regions=member[near],
        base_neighbors=scene.pred[near],
        target_neighbors=target[near],
    )


def outcome(cache, accept):
    r = cache["regions"]
    valid = r >= 0
    take = valid & accept[np.maximum(r, 0)] if len(accept) else np.zeros(r.shape, bool)
    return mode(np.where(take, cache["target_neighbors"], cache["base_neighbors"]))


def fast_gain(cache, accept):
    return gain(cache["gt"], cache["baseline"], outcome(cache, accept), cache["lookup"])


def marginal_targets(cache, nregions):
    r = cache["regions"]
    allpred = outcome(cache, np.ones(nregions, bool))
    gt = cache["gt"]
    lookup = cache["lookup"]
    correct = (gt > 0) & (lookup[allpred] == gt)
    result = np.zeros(nregions, np.float64)
    if not nregions:
        return result
    point = np.broadcast_to(np.arange(len(r))[:, None], r.shape)
    mask = r >= 0
    inverse = csr_matrix(
        (np.ones(mask.sum(), np.int8), (r[mask], point[mask])), shape=(nregions, len(r))
    )
    values = np.where(r >= 0, cache["target_neighbors"], cache["base_neighbors"])
    for j in range(nregions):
        ids = inverse[j].indices
        v = values[ids].copy()
        revert = r[ids] == j
        v[revert] = cache["base_neighbors"][ids][revert]
        before = mode(v)
        bc = (gt[ids] > 0) & (lookup[before] == gt[ids])
        result[j] = int(correct[ids].sum() - bc.sum())
    return result
