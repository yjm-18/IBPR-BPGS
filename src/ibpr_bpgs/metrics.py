import numpy as np
from scipy.optimize import linear_sum_assignment


def matrices(gt, pred, min_gt):
    gs = np.unique(gt[gt > min_gt])
    ps = np.unique(pred[pred > 0])
    valid = (gt > min_gt) & (pred > 0)
    gi = np.searchsorted(gs, gt[valid])
    pi = np.searchsorted(ps, pred[valid])
    inter = np.bincount(pi * len(gs) + gi, minlength=len(ps) * len(gs)).reshape(
        len(ps), len(gs)
    )
    psz = np.bincount(np.searchsorted(ps, pred[pred > 0]), minlength=len(ps))
    gsz = np.bincount(np.searchsorted(gs, gt[gt > min_gt]), minlength=len(gs))
    union = psz[:, None] + gsz[None, :] - inter
    iou = np.divide(
        inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0
    )
    return (gs, ps, inter, psz, gsz, iou)


def ap_flags(iou, scores, threshold):
    order = np.argsort(-scores, kind="mergesort")
    used = set()
    flags = np.zeros(len(order), dtype=np.uint8)
    for j, pred_index in enumerate(order):
        available = np.flatnonzero(iou[pred_index] >= threshold)
        available = [int(x) for x in available if int(x) not in used]
        if available:
            best = max(available, key=lambda x: iou[pred_index, x])
            used.add(best)
            flags[j] = 1
    return (scores[order], flags)


def pooled_ap(entries, threshold):
    scores = []
    flags = []
    total_gt = 0
    for entry in entries:
        s, f = ap_flags(entry["iou"], entry["scores"], threshold)
        scores.append(s)
        flags.append(f)
        total_gt += entry["iou"].shape[1]
    scores = np.concatenate(scores)
    flags = np.concatenate(flags)
    order = np.argsort(-scores, kind="mergesort")
    tp = flags[order].astype(float)
    cum_tp = np.cumsum(tp)
    recall = cum_tp / max(total_gt, 1)
    precision = cum_tp / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    sampled = []
    for target in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, target, side="left")
        sampled.append(envelope[idx] if idx < len(envelope) else 0.0)
    return float(np.mean(sampled))


def propagate(pred, idx):
    out = np.empty(len(idx), np.int32)
    for start in range(0, len(idx), 250000):
        v = np.sort(pred[idx[start : start + 250000]], axis=1)
        counts = (v[:, :, None] == v[:, None, :]).sum(2)
        out[start : start + len(v)] = v[np.arange(len(v)), counts.argmax(1)]
    return out


def lookup_for(gt, base):
    gs, ps, _, _, _, iou = matrices(gt, base, 0)
    a, b = linear_sum_assignment(iou, maximize=True)
    lookup = np.full(int(base.max(initial=0)) + 1, -1, np.int64)
    lookup[ps[a]] = gs[b]
    return lookup


def gain(gt, base, after, lookup):
    valid = gt > 0
    change = base != after
    bc = lookup[base] == gt
    ac = lookup[after] == gt
    C = int(np.sum(valid & change & ~bc & ac))
    H = int(np.sum(valid & change & bc & ~ac))
    D = int(change.sum())
    assert C - H == int(np.sum(valid & ac) - np.sum(valid & bc))
    return dict(
        C=C, H=H, G=C - H, modified_points=D, NIR=100 * (C - H) / D if D else None
    )


THRESHOLDS = np.arange(0.5, 0.951, 0.05)


def _measure(pred, gt, prob, source_pred):
    gs, ps, inter, psz, gsz, iou = matrices(gt, pred, 0)
    a, b = linear_sum_assignment(iou, maximize=True)
    matched = a[iou[a, b] > 0.5]
    other = np.setdiff1d(np.arange(len(ps)), matched)
    tp = len(matched)
    fp = int((inter[other].sum(1) / np.maximum(psz[other], 1) >= 0.5).sum())
    fn = len(gs) - tp
    mask = source_pred > 0
    count = np.bincount(source_pred[mask])
    sums = np.bincount(source_pred[mask], weights=prob[mask], minlength=len(count))
    scores = (sums / np.maximum(count, 1))[ps]
    entry = dict(iou=iou, scores=scores)
    m = dict(
        TP=tp,
        FP=fp,
        FN=fn,
        GT=len(gs),
        P=tp / max(tp + fp, 1),
        R=tp / max(tp + fn, 1),
        F1=2 * tp / max(2 * tp + fp + fn, 1),
        IoU=float(iou.max(0).mean()) if iou.size else 0.0,
        mAP=float(np.mean([pooled_ap([entry], t) for t in THRESHOLDS])),
    )
    return (m, entry)


def evaluate(scene, gt, nn, prediction):
    baseline = propagate(scene.pred, nn) if nn is not None else scene.pred
    after = propagate(prediction, nn) if nn is not None else prediction
    # Confidence always uses frozen baseline source labels.
    metrics, entry = _measure(after, gt, scene.tree_probability, scene.pred)
    metrics.update(gain(gt, baseline, after, lookup_for(gt, baseline)))
    return metrics, entry, after
