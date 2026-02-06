#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cmdstanpy
from cmdstanpy import CmdStanModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a 2PL (binomial) IRT model in Stan via cmdstanpy.")
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("data/irt_counts_task_id.csv"),
        help="Aggregated counts CSV from scripts/build_data.py --write-csv (default: data/irt_counts_task_id.csv).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model names (and pass rates) in --counts, then exit.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/out_stan"),
        help="Output directory (default: analysis/out_stan).",
    )
    parser.add_argument(
        "--anchor-low",
        type=str,
        default=None,
        help="Model name to anchor at --theta-low (default: lowest pass-rate model).",
    )
    parser.add_argument(
        "--anchor-high",
        type=str,
        default=None,
        help="Model name to anchor at --theta-high (default: highest pass-rate model).",
    )
    parser.add_argument(
        "--theta-low",
        type=float,
        default=-1.0,
        help="Ability value for anchor-low (default: -1).",
    )
    parser.add_argument(
        "--theta-high",
        type=float,
        default=1.0,
        help="Ability value for anchor-high (default: +1).",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234).",
    )
    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=1000,
        help="Warmup iterations per chain (default: 1000).",
    )
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=1000,
        help="Sampling iterations per chain (default: 1000).",
    )
    parser.add_argument(
        "--adapt-delta",
        type=float,
        default=0.9,
        help="Stan adapt_delta (default: 0.9).",
    )
    parser.add_argument(
        "--max-treedepth",
        type=int,
        default=12,
        help="Stan max_treedepth (default: 12).",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=100,
        help="CmdStan refresh rate (default: 100).",
    )
    return parser.parse_args()


def load_counts(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    required = {"model", "n", "s"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"--counts missing required columns: {sorted(missing)}")

    item_cols = [c for c in df.columns if c not in {"model", "n", "s", "f"}]
    if len(item_cols) != 1:
        raise SystemExit(f"--counts must have exactly one item column besides model/n/s (got {item_cols})")
    item_col = item_cols[0]
    return df, item_col


def build_stan_data(df: pd.DataFrame, item_col: str) -> tuple[dict[str, object], list[str], list[str]]:
    df = df.copy()
    df["n"] = df["n"].astype(int)
    df["s"] = df["s"].astype(int)

    model_names = sorted(df["model"].unique().tolist())
    item_names = sorted(df[item_col].unique().tolist())
    model_index = {m: i + 1 for i, m in enumerate(model_names)}  # 1-based for Stan
    item_index = {t: j + 1 for j, t in enumerate(item_names)}

    ii = df["model"].map(model_index).to_numpy(dtype=int)
    jj = df[item_col].map(item_index).to_numpy(dtype=int)

    stan_data: dict[str, object] = {
        "N": int(len(df)),
        "I": int(len(model_names)),
        "J": int(len(item_names)),
        "ii": ii.tolist(),
        "jj": jj.tolist(),
        "n": df["n"].to_numpy(dtype=int).tolist(),
        "s": df["s"].to_numpy(dtype=int).tolist(),
    }
    return stan_data, model_names, item_names


def pick_anchors(counts_df: pd.DataFrame, model_names: list[str], anchor_low: str | None, anchor_high: str | None) -> tuple[str, str]:
    totals = counts_df.groupby("model", as_index=True).agg(n=("n", "sum"), s=("s", "sum"))
    rates = (totals["s"] / totals["n"]).sort_values()
    if anchor_low is None and anchor_high is None:
        # Prefer recognizable anchors when available.
        preferred_low = "gpt_4"
        preferred_high = "claude_3_7_sonnet_inspect"
        if preferred_low in set(model_names) and preferred_high in set(model_names):
            low, high = preferred_low, preferred_high
        else:
            low, high = str(rates.index[0]), str(rates.index[-1])
    else:
        low = anchor_low or str(rates.index[0])
        high = anchor_high or str(rates.index[-1])
    if low not in set(model_names):
        raise SystemExit(f"--anchor-low {low!r} not found in counts")
    if high not in set(model_names):
        raise SystemExit(f"--anchor-high {high!r} not found in counts")
    if low == high:
        raise SystemExit("anchor-low and anchor-high must be different models")
    return low, high


def summarize_param(
    draws: np.ndarray, names: list[str], param_name: str, out_path: Path
) -> pd.DataFrame:
    if draws.ndim != 2 or draws.shape[1] != len(names):
        raise ValueError(f"{param_name}: expected (S, {len(names)}) draws, got {draws.shape}")
    mean = draws.mean(axis=0)
    sd = draws.std(axis=0, ddof=1)
    q05, q50, q95 = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    df = pd.DataFrame(
        {
            "name": names,
            "mean": mean,
            "sd": sd,
            "q05": q05,
            "q50": q50,
            "q95": q95,
        }
    )
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    args = parse_args()

    counts_df, item_col = load_counts(args.counts)
    if args.list_models:
        totals = counts_df.groupby("model", as_index=False).agg(n=("n", "sum"), s=("s", "sum"))
        totals["pass_rate"] = totals["s"] / totals["n"]
        totals = totals.sort_values("pass_rate", ascending=True)
        print(totals.to_string(index=False))
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    stan_data, model_names, item_names = build_stan_data(counts_df, item_col)

    anchor_low, anchor_high = pick_anchors(counts_df, model_names, args.anchor_low, args.anchor_high)
    stan_data["anchor_low"] = int(model_names.index(anchor_low) + 1)
    stan_data["anchor_high"] = int(model_names.index(anchor_high) + 1)
    stan_data["theta_low"] = float(args.theta_low)
    stan_data["theta_high"] = float(args.theta_high)

    stan_file = Path("analysis/stan/2pl_binomial.stan")
    if not stan_file.exists():
        raise SystemExit(f"Missing Stan model file: {stan_file}")

    model = CmdStanModel(stan_file=str(stan_file))
    fit = model.sample(
        data=stan_data,
        seed=args.seed,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        refresh=args.refresh,
    )

    fit_summary = fit.summary()
    fit_summary.to_csv(args.outdir / "summary.csv")

    theta_draws = fit.stan_variable("theta")
    b_draws = fit.stan_variable("b")
    a_draws = fit.stan_variable("a")

    summarize_param(theta_draws, model_names, "theta", args.outdir / "theta.csv")
    item_df = summarize_param(b_draws, item_names, "b", args.outdir / "b.csv").rename(columns={"name": item_col})
    a_df = summarize_param(a_draws, item_names, "a", args.outdir / "a.csv").rename(columns={"name": item_col})
    items = item_df.merge(a_df[[item_col, "mean", "sd", "q05", "q50", "q95"]], on=item_col, suffixes=("_b", "_a"))
    items.to_csv(args.outdir / "items.csv", index=False)

    meta = {
        "counts": str(args.counts),
        "item_col": item_col,
        "stan_file": str(stan_file),
        "cmdstanpy_version": cmdstanpy.__version__,
        "cmdstan_path": cmdstanpy.cmdstan_path(),
        "chains": args.chains,
        "iter_warmup": args.iter_warmup,
        "iter_sampling": args.iter_sampling,
        "seed": args.seed,
        "adapt_delta": args.adapt_delta,
        "max_treedepth": args.max_treedepth,
        "anchor_low": anchor_low,
        "anchor_high": anchor_high,
        "theta_low": args.theta_low,
        "theta_high": args.theta_high,
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote: {args.outdir / 'theta.csv'}")
    print(f"wrote: {args.outdir / 'items.csv'}")
    print(f"wrote: {args.outdir / 'summary.csv'}")
    print(f"wrote: {args.outdir / 'meta.json'}")


if __name__ == "__main__":
    main()
