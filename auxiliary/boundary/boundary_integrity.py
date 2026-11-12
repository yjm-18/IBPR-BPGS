"""Fail-fast checks for point/voxel alignment; no training dependencies.

No point-to-voxel mapping is inferred from point counts.  A two-dimensional
table retains the original code's point-index/-1-padding interpretation and
must agree with an independently supplied point-to-voxel vector.
"""

import numpy as np


class BoundaryIntegrityError(ValueError):
    """A batch cannot be interpreted without corrupting point identities."""


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def require_point_shapes(expected_points, **specifications):
    """Each specification is (array, expected number of dimensions)."""
    for name, (value, dimensions) in specifications.items():
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != dimensions or shape[0] != expected_points:
            raise BoundaryIntegrityError(
                f"{name} must have {expected_points} point rows and "
                f"{dimensions} dimensions; got shape {shape}. "
                "Truncation is forbidden: align arrays by point identity upstream."
            )


def validate_index_vector(value, num_points, num_voxels, name):
    array = as_numpy(value)
    require_point_shapes(num_points, **{name: (array, 1)})
    if array.dtype.kind not in "iu":
        raise BoundaryIntegrityError(f"{name} must contain integer voxel indices.")
    if num_voxels <= 0 or np.any(array < 0) or np.any(array >= num_voxels):
        raise BoundaryIntegrityError(
            f"{name} contains an index outside [0, {num_voxels}); "
            "clamping/modulo would silently change point identities."
        )
    return array


def point_to_voxel_map(input_dict, num_voxels):
    coords = input_dict.get("coords_float")
    shape = getattr(coords, "shape", None)
    if shape is None or len(shape) != 2 or shape[1] != 3 or shape[0] == 0:
        raise BoundaryIntegrityError("coords_float must be a nonempty (N, 3) array.")
    num_points = shape[0]
    candidates = []
    for name in ("v2p_map", "p2v_map"):
        value = input_dict.get(name)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is None:
            raise BoundaryIntegrityError(f"{name} must be an array with an explicit shape.")
        ndim = len(shape)
        if name == "p2v_map" and ndim == 2:
            continue
        checked = validate_index_vector(value, num_points, num_voxels, name)
        candidates.append((name, value, checked))
    if not candidates:
        raise BoundaryIntegrityError(
            "A verified one-dimensional point-to-voxel index vector is required "
            "as v2p_map or p2v_map. Equal point/voxel counts do not establish "
            "identity order; no identity or modulo mapping is synthesized."
        )
    if len(candidates) > 1 and not np.array_equal(candidates[0][2], candidates[1][2]):
        raise BoundaryIntegrityError("One-dimensional v2p_map and p2v_map disagree.")
    return candidates[0][1]


def validate_voxel_point_table(table, point_map, num_points, num_voxels):
    array = as_numpy(table)
    require_point_shapes(num_voxels, p2v_map=(array, 2))
    if array.dtype.kind not in "iu" or np.any(array < -1) or np.any(array >= num_points):
        raise BoundaryIntegrityError("p2v_map must contain point indices or -1 padding.")
    valid = array >= 0
    if np.any(valid.sum(axis=1) == 0):
        raise BoundaryIntegrityError("p2v_map contains a voxel with no input points.")
    used = array[valid]
    if len(used) != num_points or not np.array_equal(np.sort(used), np.arange(num_points)):
        raise BoundaryIntegrityError("p2v_map must assign every input point exactly once.")
    rows = np.broadcast_to(np.arange(num_voxels)[:, None], array.shape)[valid]
    inverse = as_numpy(point_map)
    if not np.array_equal(inverse[used], rows):
        raise BoundaryIntegrityError("p2v_map table disagrees with the point-to-voxel vector.")


def validate_voxel_feature_domain(feats, point_map, num_points, num_voxels, layout=None):
    """Reject the equal-sized point/voxel feature-domain ambiguity.

    With no voxel-to-point table, the original branch consumes voxel-ordered
    features directly. Equal row counts cannot establish that order when the
    supplied point-to-voxel map is not the identity.
    """
    require_point_shapes(num_voxels, voxel_feats=(feats, 2))
    ambiguous = num_points == num_voxels and not np.array_equal(
        as_numpy(point_map), np.arange(num_points)
    )
    if ambiguous and layout != "voxel":
        raise BoundaryIntegrityError(
            "Without a two-dimensional voxel-to-point table, N=M with a "
            "nonidentity point-to-voxel map leaves feats ordering ambiguous. "
            "Only after confirming feats rows already follow voxel_coords, "
            "set batch['feats_layout'] = 'voxel'; otherwise provide a verified "
            "voxel-to-point table for the point features. No ordering is guessed."
        )


def validate_metric_arrays(pred_probs, gt_labels, valid_mask=None):
    probs = as_numpy(pred_probs)
    labels = as_numpy(gt_labels)
    if probs.ndim != 1:
        raise BoundaryIntegrityError("Boundary probabilities must be a one-dimensional point array.")
    require_point_shapes(len(probs), gt_labels=(labels, 1))
    if not np.isfinite(probs).all() or not np.isfinite(labels).all():
        raise BoundaryIntegrityError("Boundary probabilities/labels contain nonfinite values.")
    mask = None
    if valid_mask is not None:
        mask = as_numpy(valid_mask)
        require_point_shapes(len(probs), valid_mask=(mask, 1))
        if mask.dtype.kind != "b":
            raise BoundaryIntegrityError("valid_mask must be Boolean, not a list of point indices.")
    return probs, labels, mask
