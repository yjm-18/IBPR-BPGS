"""Evaluate ordered scene manifests and pool AP/counts without scene-macro averaging."""

import argparse, json, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--weights", default="checkpoints")
p.add_argument("--output", required=True)
a = p.parse_args()
path = Path(a.manifest).resolve()
data = json.loads(path.read_text())
out = Path(a.output)
runs = []
for r in data["scenes"]:
    run = out / r["id"]
    runs.append(str(run))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ibpr_bpgs.cli",
            "infer",
            "--scene",
            str(path.parent / r["path"]),
            "--config",
            r["config"],
            "--weights",
            a.weights,
            "--output",
            str(run),
            "--evaluate",
        ],
        check=True,
    )
subprocess.run(
    [
        sys.executable,
        "-m",
        "ibpr_bpgs.cli",
        "aggregate",
        "--runs",
        *runs,
        "--output",
        str(out / "summary.json"),
    ],
    check=True,
)
