"""Training from prepared predictions; no dependency on workstation experiment trees.

The release reproduces the 39D refit using previously locked configurations.
`nested-b` additionally supports a fresh LAUTx inner-LOPO selection.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from .features import AFeatureContext, relationship_39d
from .supervision import candidate_supervision
from .scorer import score_scene
from .regions import region_data
from .metrics import propagate
from .targets import compact_cache, marginal_targets, fast_gain
from .io import load_scene, load_model, dump_model, write_json


def pairs(scene, gt, seed, probes, active):
    ctx = AFeatureContext(scene)
    rng = np.random.default_rng(seed)
    ids = rng.choice(ctx.pool, min(probes, len(ctx.pool)), replace=False)
    cand, ev = ctx.proposals(ids)
    chosen = np.flatnonzero(cand[:, 1] > 0)
    if len(chosen) > active:
        chosen = rng.choice(chosen, active, replace=False)
    ids = ids[chosen]
    cand = cand[chosen]
    ev = ev[chosen]
    labels, valid, _ = candidate_supervision(gt, scene.pred, ids, cand, minimum_gt_id=1)
    X = relationship_39d(ctx.features_13d(ids, cand, ev))[valid]
    return X, labels[valid], ids, valid


def fit_a(X, y, parameters):
    if X.ndim != 2 or X.shape[1] != 39 or len(np.unique(y)) != 2:
        raise ValueError("IBPR training requires (N,39) and both binary classes")
    model = HistGradientBoostingClassifier(**parameters)
    with threadpool_limits(limits=4):
        model.fit(X, y)
    return model


def fit_b(X, y, size, parameters, power):
    if X.ndim != 2 or X.shape[1] != 38 or not len(y):
        raise ValueError("BPGS training requires nonempty (N,38)")
    model = ExtraTreesRegressor(**parameters)
    model.fit(X, y, sample_weight=np.maximum(size, 1) ** power)
    return model


def make_regions(scene, gt, nn, scorer, config, point_mask=None):
    scores = score_scene(scene, scorer, chunk_size=50000)
    if point_mask is not None:
        keep = point_mask[scores["point_ids"]]
        scores = {k: v[keep] for k, v in scores.items()}
    d = region_data(scene, scores, config["A"])
    baseline = propagate(scene.pred, nn) if nn is not None else scene.pred
    cache = compact_cache(scene, gt, nn, baseline, d)
    return d, cache, marginal_targets(cache, len(d["components"]))


def load_cache(paths):
    rows = []
    for path in paths:
        with np.load(path) as z:
            rows.append((z["X"], z["y"], z["size"]))
    return tuple(np.concatenate([r[i] for r in rows]) for i in range(3))


def reproduce_fixed(manifest, config, out):
    """FOR scene OOF / Ofental spatial OOF; frozen selected params, no new search."""
    data = json.loads(Path(manifest).read_text())
    base = Path(manifest).resolve().parent
    rows = data["train"]
    dataset = config["dataset"]
    if dataset not in ["FOR-instance", "Ofental"]:
        raise ValueError("Use nested-b or regions/fit-b for LAUTx")
    if dataset == "FOR-instance" and len(rows) != 13:
        raise ValueError("Expected ordered 13 training scenes")
    if dataset == "Ofental" and len(rows) != 1:
        raise ValueError("Expected one Ofental training region")
    arrays = []
    folds = []
    cuts = None
    for i, row in enumerate(rows):
        if row.get("split") != "train":
            raise ValueError("Only explicitly labelled training scenes may be fitted")
        scene, gt, nn, sgt = load_scene(base / row["path"])
        if sgt is None:
            raise ValueError(
                "TL training needs source_gt (not evaluation GT indexed arbitrarily)"
            )
        X, y, ids, valid = pairs(
            scene,
            sgt,
            20261029 + i if dataset == "FOR-instance" else 20261106,
            250000 if dataset == "FOR-instance" else 800000,
            30000 if dataset == "FOR-instance" else 120000,
        )
        arrays.append((X, y))
        if dataset == "FOR-instance":
            folds.append(np.full(len(y), i % 5, np.int8))
        else:
            cuts = np.quantile(scene.pos[:, 0], [0.2, 0.4, 0.6, 0.8])
            pf = np.digitize(scene.pos[:, 0], cuts)
            folds.append(np.broadcast_to(pf[ids, None], valid.shape)[valid])
        del scene
    X = np.concatenate([x[0] for x in arrays])
    y = np.concatenate([x[1] for x in arrays])
    fold = np.concatenate(folds)
    model = fit_a(X, y, config["A"]["parameters"])
    dump_model(out / "A.pkl", model)
    np.savez_compressed(out / "A_pairs.npz", X=X, y=y, fold=fold)
    caches = []
    for f in range(5):
        scorer = fit_a(X[fold != f], y[fold != f], config["A"]["parameters"])
        dump_model(out / f"fold{f}/A.pkl", scorer)
        for i, row in enumerate(rows):
            if dataset == "FOR-instance" and i % 5 != f:
                continue
            scene, gt, nn, _ = load_scene(base / row["path"])
            if gt is None:
                raise ValueError("B training requires evaluation GT")
            mask = (
                None
                if dataset == "FOR-instance"
                else np.digitize(scene.pos[:, 0], cuts) == f
            )
            d, cache, target = make_regions(scene, gt, nn, scorer, config, mask)
            path = out / f"fold{f}/regions_{i}.npz"
            np.savez_compressed(
                path, X=d["features"], y=target, size=d["size"][d["components"]]
            )
            caches.append(path)
    bx, by, bs = load_cache(caches)
    b = fit_b(bx, by, bs, config["B"]["parameters"], config["B"]["power"])
    dump_model(out / "B.pkl", b)
    write_json(
        out / "training.json",
        {
            "dataset": dataset,
            "train_ids": [r["id"] for r in rows],
            "A_features": 39,
            "B_features": 38,
            "config": config,
            "selection": "locked configuration; no new validation search",
            "folds": 5,
        },
    )


def nested_b(manifest, config, out):
    """True outer leave-one-out, inner leave-one-of-remaining-out B selection.

    Inputs: per-scene training region X/y/size and compact validation replay arrays.
    All inner fitting/selection excludes the outer scene. A is frozen.
    """
    data = json.loads(Path(manifest).read_text())
    root = Path(manifest).resolve().parent
    rows = data["scenes"]
    ids = [r["id"] for r in rows]
    if len(rows) != 6 or len(set(ids)) != 6:
        raise ValueError("Expected six distinct LAUTx plots")
    threshold = [-1e12, -1.0, 0.0, 1.0, 2.0, 5.0, 1e12]
    for outer in rows:
        train = [r for r in rows if r["id"] != outer["id"]]
        trials = []
        for leaf in [3, 8]:
            for power in [0.0, 0.5]:
                counts = {t: dict(G=0, modified_points=0) for t in threshold}
                for val in train:
                    x, y, size = load_cache(
                        [root / r["regions"] for r in train if r["id"] != val["id"]]
                    )
                    params = {**config["B"]["parameters"], "min_samples_leaf": leaf}
                    model = fit_b(x, y, size, params, power)
                    with np.load(root / val["regions"]) as z:
                        u = model.predict(z["X"]) if len(z["X"]) else np.empty(0)
                    with np.load(root / val["cache"]) as z:
                        cache = dict(z)
                    for t in threshold:
                        g = fast_gain(cache, u >= t)
                        for k in counts[t]:
                            counts[t][k] += g[k]
                trials.extend(
                    {"leaf": leaf, "power": power, "threshold": t, **counts[t]}
                    for t in threshold
                )
        best = max(trials, key=lambda x: (x["G"], -x["modified_points"]))
        x, y, size = load_cache([root / r["regions"] for r in train])
        params = {**config["B"]["parameters"], "min_samples_leaf": best["leaf"]}
        model = fit_b(x, y, size, params, best["power"])
        dump_model(out / outer["id"] / "B.pkl", model)
        write_json(
            out / outer["id"] / "selection.json",
            {
                "selected": best,
                "trials": trials,
                "training_ids": [r["id"] for r in train],
                "outer": outer["id"],
            },
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["reproduce-fixed", "nested-b"]:
        q = sub.add_parser(name)
        q.add_argument("--manifest", required=True)
        q.add_argument("--config", required=True)
        q.add_argument("--output", required=True)
    q = sub.add_parser("regions")
    q.add_argument("--scene", required=True)
    q.add_argument("--a-model", required=True)
    q.add_argument("--config", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("fit-b")
    q.add_argument("--caches", nargs="+", required=True)
    q.add_argument("--config", required=True)
    q.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.output)
    if a.command == "fit-b":
        if out.exists():
            raise ValueError("Output already exists")
        x, y, s = load_cache(a.caches)
        dump_model(out, fit_b(x, y, s, cfg["B"]["parameters"], cfg["B"]["power"]))
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new/empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    if a.command == "reproduce-fixed":
        reproduce_fixed(a.manifest, cfg, out)
    elif a.command == "nested-b":
        nested_b(a.manifest, cfg, out)
    else:
        scene, gt, nn, _ = load_scene(a.scene)
        if gt is None:
            raise ValueError(
                "Region targets require GT; use only the designated training/inner-validation scenes"
            )
        d, c, y = make_regions(scene, gt, nn, load_model(a.a_model), cfg)
        np.savez_compressed(
            out / "regions.npz", X=d["features"], y=y, size=d["size"][d["components"]]
        )
        np.savez_compressed(out / "cache.npz", **c)


if __name__ == "__main__":
    main()
