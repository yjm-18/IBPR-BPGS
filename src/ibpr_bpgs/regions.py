import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .features import instance_statistics as _statistics
from .suggestions import select_best_competitor


def voxel_components(
    xyz: np.ndarray, source: np.ndarray, destination: np.ndarray, voxel_size: float
) -> np.ndarray:
    """Connected components within source->destination pairs (26-neighbour voxels)."""
    labels = np.empty(len(xyz), dtype=np.int64)
    if not len(xyz):
        return labels
    order = np.lexsort((destination, source))
    boundaries = np.r_[
        0,
        1
        + np.flatnonzero(
            (source[order][1:] != source[order][:-1])
            | (destination[order][1:] != destination[order][:-1])
        ),
        len(order),
    ]
    next_label = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        point_ids = order[start:stop]
        voxels, inverse = np.unique(
            np.floor(xyz[point_ids] / voxel_size).astype(np.int64),
            axis=0,
            return_inverse=True,
        )
        if len(voxels) == 1:
            component = np.zeros(1, dtype=np.int32)
            count = 1
        else:
            edges = cKDTree(voxels).query_pairs(r=1.01, p=np.inf, output_type="ndarray")
            if len(edges):
                rows = np.r_[edges[:, 0], edges[:, 1]]
                cols = np.r_[edges[:, 1], edges[:, 0]]
                graph = coo_matrix(
                    (np.ones(len(rows), np.uint8), (rows, cols)),
                    shape=(len(voxels), len(voxels)),
                ).tocsr()
                count, component = connected_components(
                    graph, directed=False, return_labels=True
                )
            else:
                count = len(voxels)
                component = np.arange(count, dtype=np.int32)
        labels[point_ids] = next_label + component[inverse]
        next_label += int(count)
    return labels


FEATURE_NAMES = (
    "log_size",
    "log_seed_count",
    "seed_fraction",
    "log_added_count",
    "extent_3d",
    "extent_max_axis",
    "extent_xy",
    "extent_z",
    "score_mean",
    "score_std",
    "score_min",
    "score_q10",
    "score_q25",
    "gain_mean",
    "gain_std",
    "gain_min",
    "gain_q10",
    "gain_q25",
    "boundary_mean",
    "boundary_std",
    "tree_probability_mean",
    "tree_probability_min",
    "log_prediction_count_mean",
    "log_source_size",
    "log_destination_size",
    "log_size_ratio",
    "centroid_source_distance",
    "centroid_destination_distance",
    "centroid_distance_difference",
    "point_source_distance",
    "point_destination_distance",
    "point_distance_difference",
    "embedding_source_distance",
    "embedding_destination_distance",
    "embedding_distance_difference",
    "shifted_source_distance",
    "shifted_destination_distance",
    "shifted_distance_difference",
)


def instance_statistics(scene):
    d = _statistics(scene)
    d["embed_center"] = d["embedding_center"]
    return d


def candidate_arrays(scene, scores):
    p = select_best_competitor(
        scores["point_ids"], scene.pred, scores["candidates"], scores["scores"]
    )
    seed = p["valid"] & (p["target_score"] >= 0.5) & (p["score_advantage"] > 0.1)
    return (
        p["point_ids"],
        p["target_instance"],
        p["target_score"],
        p["score_advantage"],
        p["valid"],
        seed,
    )


def region_data(scene, score_path, config):
    ids, dest, score, gain, valid, seed = candidate_arrays(scene, score_path)
    pool = valid & (score >= config["score"]) & (gain > config["margin"])
    chosen = np.flatnonzero(pool)
    labels = voxel_components(
        scene.pos[ids[chosen]], scene.pred[ids[chosen]], dest[chosen], 0.25
    )
    ncomp = labels.max(initial=-1) + 1
    size = np.bincount(labels, minlength=ncomp)
    seeds = np.bincount(labels, weights=seed[chosen], minlength=ncomp)
    fraction = seeds / np.maximum(size, 1)
    accepted = size > 0
    components = np.flatnonzero(accepted)
    row_of = np.full(ncomp, -1, np.int64)
    row_of[components] = np.arange(len(components))
    stats = instance_statistics(scene)
    xyz = scene.pos[ids[chosen]]
    features = np.empty((len(components), len(FEATURE_NAMES)), np.float32)
    corrected = np.zeros(len(components), np.int64)
    harmed = np.zeros(len(components), np.int64)
    mapping = None
    lookup = None
    for j, component in enumerate(components):
        loc = np.flatnonzero(labels == component)
        points = chosen[loc]
        point_ids = ids[points]
        source = int(scene.pred[point_ids[0]])
        target = int(dest[points[0]])
        p = scene.pos[point_ids]
        lo = p.min(0)
        hi = p.max(0)
        delta = hi - lo
        center = p.mean(0)
        ss = score[points]
        gg = gain[points]
        bb = scene.boundary_score[point_ids]
        tp = scene.tree_probability[point_ids]
        pc = np.log1p(scene.prediction_count[point_ids])
        source_center = stats["center"][source]
        target_center = stats["center"][target]
        pd_source = np.linalg.norm(p - source_center, axis=1)
        pd_target = np.linalg.norm(p - target_center, axis=1)
        emb = scene.embedding[point_ids]
        shifted = p + scene.offset[point_ids]
        ed_source = np.linalg.norm(emb - stats["embed_center"][source], axis=1)
        ed_target = np.linalg.norm(emb - stats["embed_center"][target], axis=1)
        sd_source = np.linalg.norm(shifted - stats["shifted_center"][source], axis=1)
        sd_target = np.linalg.norm(shifted - stats["shifted_center"][target], axis=1)
        cs = np.linalg.norm(center - source_center)
        ct = np.linalg.norm(center - target_center)
        features[j] = (
            np.log1p(len(loc)),
            np.log1p(seeds[component]),
            fraction[component],
            np.log1p(len(loc) - seeds[component]),
            np.linalg.norm(delta),
            delta.max(),
            np.linalg.norm(delta[:2]),
            delta[2],
            ss.mean(),
            ss.std(),
            ss.min(),
            np.quantile(ss, 0.1),
            np.quantile(ss, 0.25),
            gg.mean(),
            gg.std(),
            gg.min(),
            np.quantile(gg, 0.1),
            np.quantile(gg, 0.25),
            bb.mean(),
            bb.std(),
            tp.mean(),
            tp.min(),
            pc.mean(),
            np.log1p(stats["count"][source]),
            np.log1p(stats["count"][target]),
            np.log((stats["count"][target] + 1) / (stats["count"][source] + 1)),
            cs,
            ct,
            ct - cs,
            pd_source.mean(),
            pd_target.mean(),
            pd_target.mean() - pd_source.mean(),
            ed_source.mean(),
            ed_target.mean(),
            ed_target.mean() - ed_source.mean(),
            sd_source.mean(),
            sd_target.mean(),
            sd_target.mean() - sd_source.mean(),
        )
    assert np.isfinite(features).all()
    return {
        "ids": ids,
        "dest": dest,
        "seed": seed,
        "chosen": chosen,
        "labels": labels,
        "components": components,
        "row_of": row_of,
        "features": features,
        "corrected": corrected,
        "harmed": harmed,
        "size": size,
        "seed_count": seeds,
    }


def apply(scene, d, accept):
    output = scene.pred.copy()
    keep = np.zeros(len(d["size"]), bool)
    keep[d["components"][accept]] = True
    chosen = d["chosen"][keep[d["labels"]]]
    output[d["ids"][chosen]] = d["dest"][chosen]
    return output
