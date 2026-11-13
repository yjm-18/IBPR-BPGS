import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .io import load_scene, load_model, write_json
from .pipeline import predict, aggregate
from .metrics import evaluate


def main():
    p = argparse.ArgumentParser(description="IBPR39+BPGS38")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("infer", help="Run baseline, A and AB from a prepared scene")
    q.add_argument("--scene", required=True)
    q.add_argument("--config", required=True)
    q.add_argument("--weights", default="checkpoints")
    q.add_argument("--output", required=True)
    q.add_argument("--chunk-size", type=int, default=50000)
    q.add_argument(
        "--evaluate",
        action="store_true",
        help="Read optional GT after inference for reporting",
    )
    q = sub.add_parser("aggregate", help="Micro PRF, tree-weighted IoU and pooled AP")
    q.add_argument(
        "--runs", nargs="+", required=True, help="Ordered inference output directories"
    )
    q.add_argument("--output", required=True)
    q = sub.add_parser("verify-weights")
    q.add_argument("--manifest", default="configs/checkpoints.json")
    q.add_argument("--weights", default="checkpoints")
    args = p.parse_args()
    if args.command == "verify-weights":
        for item in json.loads(Path(args.manifest).read_text()):
            f = Path(args.weights) / item["path"]
            with f.open("rb") as stream:
                h = hashlib.file_digest(stream, "sha256").hexdigest()
            if h != item["sha256"]:
                raise ValueError(f"Hash mismatch: {f}")
        print("All checkpoint hashes verified")
        return
    if args.command == "aggregate":
        result = {}
        for arm in ["baseline", "A", "AB"]:
            rows = []
            entries = []
            for run in args.runs:
                folder = Path(run)
                rows.append(json.loads((folder / f"{arm}_result.json").read_text()))
                with np.load(folder / f"{arm}_metrics.npz") as z:
                    entries.append(dict(z))
            result[arm] = aggregate(rows, entries)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    cfg = json.loads(Path(args.config).read_text())
    scene, gt, nn, _ = load_scene(args.scene)
    if (cfg["model"] == "TL") != (nn is not None):
        raise ValueError("TL requires 5NN; SAT must use direct aligned points")
    if scene.embedding.shape[1] != (32 if cfg["model"] == "TL" else 5):
        raise ValueError("Unexpected backbone feature dimension")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new/empty output directory to preserve results")
    out.mkdir(parents=True, exist_ok=True)
    a = load_model(Path(args.weights) / cfg["A"]["path"])
    b = load_model(Path(args.weights) / cfg["B"]["path"])
    source, ep, scores, regions, u, accept = predict(
        scene, nn, a, b, cfg, args.chunk_size
    )
    np.savez_compressed(out / "scores.npz", **scores)
    np.savez_compressed(
        out / "regions.npz",
        X=regions["features"],
        size=regions["size"][regions["components"]],
        scores=u,
        accepted=accept,
    )
    for arm in source:
        np.savez_compressed(
            out / f"{arm}_prediction.npz",
            source_prediction=source[arm],
            evaluation_prediction=ep[arm],
        )
        if args.evaluate:
            if gt is None:
                raise ValueError("--evaluate requires gt in the prepared scene")
            m, e, _ = evaluate(scene, gt, nn, source[arm])
            write_json(out / f"{arm}_result.json", m)
            np.savez_compressed(out / f"{arm}_metrics.npz", **e)
    write_json(
        out / "run.json",
        {
            "config": cfg,
            "scene": str(args.scene),
            "evaluated": args.evaluate,
            "regions": len(u),
            "accepted_regions": int(accept.sum()),
            "metric_units": "per-scene IoU/mAP/PRF fractions; NIR percent",
        },
    )
    print(f"Finished: {out}")


if __name__ == "__main__":
    main()
