#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot prior vs posterior for a_pow (x^a trend) in the xpow Stan model.")
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--spec", type=str, default="time_irt__theta_xpow")
    p.add_argument("--run-id", type=str, default=None, help="Defaults to --runs-root/<spec>/LATEST.")
    p.add_argument("--fitdir", type=Path, default=None, help="Fit directory (overrides --spec/--run-id).")
    p.add_argument("--outdir", type=Path, default=None, help="Output dir (default: run figures dir).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed (default: 123).")
    p.add_argument("--prior-samples", type=int, default=200_000, help="Number of prior samples (default: 200000).")
    return p.parse_args()


def chain_csv_files(fitdir: Path) -> list[Path]:
    files = sorted(fitdir.glob("*.csv"))
    keep: list[Path] = []
    for f in files:
        if f.name in {"summary.csv", "theta.csv", "b.csv", "a.csv"}:
            continue
        keep.append(f)
    return keep


def truncated_normal_prior(
    *, mu: float, sigma: float, lo: float, hi: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    out = np.empty(size, dtype=float)
    filled = 0
    # Rejection sampling (fast enough in this range).
    while filled < size:
        draw = rng.normal(mu, sigma, size=size - filled)
        draw = draw[(draw >= lo) & (draw <= hi)]
        k = len(draw)
        if k == 0:
            continue
        out[filled : filled + k] = draw
        filled += k
    return out


def summarize(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q025, q50, q975 = np.quantile(x, [0.025, 0.5, 0.975])
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "q025": float(q025),
        "q50": float(q50),
        "q975": float(q975),
        "n": float(len(x)),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.fitdir is None:
        run_id = args.run_id or (args.runs_root / args.spec / "LATEST").read_text(encoding="utf-8").strip()
        fitdir = args.runs_root / args.spec / run_id / "fit"
    else:
        fitdir = args.fitdir
        run_id = fitdir.parent.name if fitdir.name == "fit" else fitdir.name

    if not fitdir.exists():
        raise SystemExit(f"Missing fitdir: {fitdir}")

    meta = json.loads((fitdir / "meta.json").read_text(encoding="utf-8"))
    trend = str(meta.get("theta_trend", ""))
    if trend != "xpow":
        raise SystemExit(f"Expected theta_trend=xpow, got {trend!r} in {fitdir/'meta.json'}")

    if args.outdir is None:
        args.outdir = fitdir.parent / "figures"
    args.outdir.mkdir(parents=True, exist_ok=True)

    chain_files = chain_csv_files(fitdir)
    if not chain_files:
        raise SystemExit(f"No chain CSVs found in {fitdir}")
    post = pd.concat([pd.read_csv(f, comment="#", usecols=["a_pow"]) for f in chain_files], ignore_index=True)[
        "a_pow"
    ].to_numpy(dtype=float)
    post = post[np.isfinite(post)]

    # Prior from the Stan model:
    #   a_pow ~ normal(1.0, 0.5) truncated to [0.1, 2.0]
    prior = truncated_normal_prior(mu=1.0, sigma=0.5, lo=0.1, hi=2.0, size=int(args.prior_samples), rng=rng)

    post_s = summarize(post)
    prior_s = summarize(prior)
    summary = pd.DataFrame(
        [
            {"dist": "prior", **prior_s},
            {"dist": "posterior", **post_s},
        ]
    )
    summary.to_csv(args.outdir / "a_pow_prior_posterior_summary.csv", index=False)

    # Plot
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.4))
    bins = np.linspace(0.1, 2.0, 55)
    ax.hist(prior, bins=bins, density=True, alpha=0.35, color="C0", label="prior (trunc Normal(1,0.5), [0.1,2])")
    ax.hist(post, bins=bins, density=True, alpha=0.45, color="C1", label="posterior")

    ax.axvline(prior_s["q50"], color="C0", linestyle="--", linewidth=1)
    ax.axvline(post_s["q50"], color="C1", linestyle="--", linewidth=1)

    ax.set_xlabel(r"$a_{\mathrm{pow}}$")
    ax.set_ylabel("Density")
    ax.set_title(f"Prior vs posterior for a_pow ({args.spec} / {run_id})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "a_pow_prior_posterior.png", dpi=200)
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"wrote: {args.outdir / 'a_pow_prior_posterior.png'}")
    print(f"wrote: {args.outdir / 'a_pow_prior_posterior_summary.csv'}")


if __name__ == "__main__":
    main()

