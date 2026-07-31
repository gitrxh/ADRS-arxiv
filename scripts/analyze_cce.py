#!/usr/bin/env python3
"""
Offline analysis for CCE-bootstrap dumps (Stage 0).

Reads the JSONL produced by the DUMP_CCE hook (one record per decision/turn) and
produces the two thesis artifacts:

  1. delta-vs-CCE scatter + binned-mean curve. The discriminative claim is visible
     as a mass of high-|delta| points whose CCE ~= 0 ("skill changes the action but
     it does not improve value").
  2. Summary stats: corr(delta, CCE), the high-delta-low-CCE fraction, ESS health,
     and (if a wrong_skill tag is present) a real-vs-wrong overlay.

No GPU / model / torch needed — just numpy + matplotlib.

Usage:
    python scripts/analyze_cce.py --dump cce_dump/cce_records.jsonl --out cce_dump/
    python scripts/analyze_cce.py --dump cce_dump/cce_records.jsonl --min-step 5
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def load_records(path, min_step=0):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("global_step", 0) < min_step:
                continue
            recs.append(r)
    return recs


def to_arrays(recs, tag=None):
    if tag is not None:
        recs = [r for r in recs if r.get("tag") == tag]
    if not recs:
        return None
    d = {
        "delta": np.array([r["delta_mean"] for r in recs], dtype=float),
        "cce": np.array([r["cce"] for r in recs], dtype=float),
        "ret": np.array([r.get("return", 0.0) for r in recs], dtype=float),
        "ess": np.array([r.get("ess", 0.0) for r in recs], dtype=float),
        "won": np.array([r.get("won", np.nan) for r in recs], dtype=float),
        "step": np.array([r.get("global_step", 0) for r in recs], dtype=int),
    }
    return d


def binned_curve(x, y, nbins=12):
    """Mean of y within equal-count bins of x. Returns (centers, means, sems)."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    if n < nbins:
        nbins = max(1, n)
    edges = np.linspace(0, n, nbins + 1).astype(int)
    centers, means, sems = [], [], []
    for i in range(nbins):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            continue
        xb, yb = x[s:e], y[s:e]
        centers.append(xb.mean())
        means.append(yb.mean())
        sems.append(yb.std() / max(1, np.sqrt(len(yb))))
    return np.array(centers), np.array(means), np.array(sems)


def pearson(x, y):
    if len(x) < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 0 else 0.0


def summarize(name, d):
    delta, cce, ess = d["delta"], d["cce"], d["ess"]
    valid = cce != 0.0  # groups with <min_group_size were zeroed
    nz_cce = cce[valid] if valid.any() else cce
    abs_scale = np.mean(np.abs(nz_cce)) if len(nz_cce) else 1e-8
    abs_scale = max(abs_scale, 1e-8)

    # high-|delta| quartile decisions with near-zero causal value
    thr = np.quantile(np.abs(delta), 0.75) if len(delta) >= 4 else np.inf
    high_d = np.abs(delta) >= thr
    low_c = np.abs(cce) < 0.25 * abs_scale
    frac = float((high_d & low_c).sum() / max(1, high_d.sum()))

    print(f"\n===== {name} =====")
    print(f"  decisions (rows)         : {len(delta)}")
    print(f"  groups with CCE!=0       : {int(valid.sum())}")
    print(f"  corr(delta_mean, CCE)    : {pearson(delta, cce):+.4f}   <- low |corr| = delta alone is a poor causal signal")
    print(f"  high-|delta| & low-CCE   : {frac:.1%}        <- the discriminative mass (skill changes action, value unchanged)")
    print(f"  CCE  mean / std          : {cce.mean():+.4f} / {cce.std():.4f}")
    print(f"  CCE  positive ratio      : {float((cce > 0).mean()):.3f}")
    print(f"  delta mean / std         : {delta.mean():+.4f} / {delta.std():.4f}")
    print(f"  ESS  mean / min          : {ess.mean():.2f} / {ess.min():.2f}   (low ESS => IS estimate degenerate)")
    return {"corr": pearson(delta, cce), "highd_lowc": frac}


def plot(real, wrong, out_dir):
    if not HAVE_MPL:
        print("\n[plot] matplotlib unavailable — skipping figures (stats printed above).")
        return
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- (a) scatter: delta vs CCE, colored by trajectory success ----
    ax = axes[0]
    d = real
    has_won = not np.all(np.isnan(d["won"]))
    if has_won:
        for val, col, lab in [(1.0, "#2a9d8f", "won"), (0.0, "#e76f51", "lost")]:
            m = d["won"] == val
            ax.scatter(d["delta"][m], d["cce"][m], s=10, alpha=0.4, c=col, label=lab)
        ax.legend(loc="upper left", fontsize=9)
    else:
        ax.scatter(d["delta"], d["cce"], s=10, alpha=0.4, c="#264653")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("delta_mean  (skill intervention magnitude)")
    ax.set_ylabel("CCE_boot  (counterfactual value gain)")
    ax.set_title("(a) skill intervention vs causal value\nhigh-delta band near CCE=0 = the thesis")

    # ---- (b) binned-mean curve: does delta predict CCE? ----
    ax = axes[1]
    cx, cy, ce = binned_curve(d["delta"], d["cce"])
    ax.errorbar(cx, cy, yerr=ce, fmt="-o", c="#264653", label="real skill", capsize=3)
    if wrong is not None:
        wx, wy, we = binned_curve(wrong["delta"], wrong["cce"])
        ax.errorbar(wx, wy, yerr=we, fmt="--s", c="#9b2226", label="wrong skill", capsize=3)
        ax.legend(fontsize=9)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("delta_mean (binned, equal count)")
    ax.set_ylabel("mean CCE_boot")
    ax.set_title("(b) binned causal value vs intervention\nflat/zero => correlational credit would be wrong")

    fig.tight_layout()
    p = os.path.join(out_dir, "cce_delta_scatter.png")
    fig.savefig(p, dpi=140)
    print(f"\n[plot] saved {p}")

    # ---- ESS health ----
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    ax2.hist(real["ess"], bins=30, color="#457b9d")
    ax2.set_xlabel("Kish ESS per group")
    ax2.set_ylabel("count")
    ax2.set_title("IS effective sample size\n(low => need rollout-CCE instead)")
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "cce_ess_hist.png")
    fig2.savefig(p2, dpi=140)
    print(f"[plot] saved {p2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="path to cce_records.jsonl")
    ap.add_argument("--out", default=None, help="output dir for figures (default: dir of --dump)")
    ap.add_argument("--min-step", type=int, default=0, help="ignore records before this global_step")
    args = ap.parse_args()

    out_dir = args.out or (os.path.dirname(os.path.abspath(args.dump)) or ".")

    recs = load_records(args.dump, min_step=args.min_step)
    if not recs:
        print(f"No records in {args.dump} (min_step={args.min_step}).")
        return
    print(f"Loaded {len(recs)} records from {args.dump} (steps "
          f"{min(r['global_step'] for r in recs)}..{max(r['global_step'] for r in recs)}).")

    real = to_arrays(recs, tag="real_skill")
    wrong = to_arrays(recs, tag="wrong_skill")

    if real is None:
        real = to_arrays(recs)  # no tags — treat all as real
    summarize("real skill", real)
    if wrong is not None:
        summarize("wrong skill (ablation)", wrong)

    plot(real, wrong, out_dir)


if __name__ == "__main__":
    main()
