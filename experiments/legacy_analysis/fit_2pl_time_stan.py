#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a 2PL model with log-linear difficulty vs log task length and constant-in-time discrimination, "
            "using Stan via cmdstanpy."
        )
    )
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("data/irt_counts_task_id.csv"),
        help="Counts CSV from scripts/build_data.py --write-csv (default: data/irt_counts_task_id.csv).",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/runs.pkl"),
        help="Runs DataFrame pickle (for task lengths) (default: data/runs.pkl).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/out_stan_time"),
        help="Output directory (default: analysis/out_stan_time).",
    )
    parser.add_argument(
        "--variant",
        choices=["b_logt__loga_const", "b_logt__loga_logt"],
        default="b_logt__loga_const",
        help="Model variant to fit (default: b_logt__loga_const).",
    )
    parser.add_argument(
        "--x-transform",
        choices=["log_hours", "log1p_minutes"],
        default="log_hours",
        help="Length transform used in b(t) and (optionally) loga(t) (default: log_hours).",
    )
    parser.add_argument(
        "--t0-minutes",
        type=float,
        default=1.0,
        help="Scale for log1p transform: x=log1p(t_minutes/t0) (default: 1 minute).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models (and pass rates) in --counts, then exit.",
    )
    parser.add_argument(
        "--anchor-low",
        type=str,
        default=None,
        help="Model name to anchor at --theta-low (default: gpt_4 if present else lowest pass-rate).",
    )
    parser.add_argument(
        "--anchor-high",
        type=str,
        default=None,
        help="Model name to anchor at --theta-high (default: claude_3_7_sonnet_inspect if present else highest pass-rate).",
    )
    parser.add_argument("--theta-low", type=float, default=-1.0)
    parser.add_argument("--theta-high", type=float, default=1.0)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--iter-warmup", type=int, default=1000)
    parser.add_argument("--iter-sampling", type=int, default=1000)
    parser.add_argument("--adapt-delta", type=float, default=0.9)
    parser.add_argument("--max-treedepth", type=int, default=12)
    parser.add_argument("--refresh", type=int, default=100)
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
    if item_col != "task_id":
        raise SystemExit("This model expects task-level counts with item column task_id.")
    return df, item_col


def pick_anchors(counts_df: pd.DataFrame, model_names: list[str], anchor_low: str | None, anchor_high: str | None) -> tuple[str, str]:
    totals = counts_df.groupby("model", as_index=True).agg(n=("n", "sum"), s=("s", "sum"))
    rates = (totals["s"] / totals["n"]).sort_values()
    preferred_low = "gpt_4"
    preferred_high = "claude_3_7_sonnet_inspect"

    if anchor_low is None and preferred_low in set(model_names):
        low = preferred_low
    else:
        low = anchor_low or str(rates.index[0])

    if anchor_high is None and preferred_high in set(model_names):
        high = preferred_high
    else:
        high = anchor_high or str(rates.index[-1])

    if low not in set(model_names):
        raise SystemExit(f"--anchor-low {low!r} not found in counts")
    if high not in set(model_names):
        raise SystemExit(f"--anchor-high {high!r} not found in counts")
    if low == high:
        raise SystemExit("anchor-low and anchor-high must be different models")
    return low, high


def build_task_x(runs_df: pd.DataFrame, task_ids: list[str]) -> tuple[np.ndarray, float]:
    task_time = runs_df.groupby("task_id", as_index=True)["human_minutes"].first()
    missing = [t for t in task_ids if t not in task_time.index]
    if missing:
        raise SystemExit(f"Missing human_minutes for {len(missing)} task_ids (example: {missing[:5]})")
    t_minutes = task_time.reindex(task_ids).to_numpy(dtype=float)
    if np.any(t_minutes <= 0) or np.any(~np.isfinite(t_minutes)):
        raise SystemExit("Non-positive or invalid task lengths encountered.")
    # Default transform: log(hours)
    t_hours = t_minutes / 60.0
    log_t_hours = np.log(t_hours)
    mean_log_t_hours = float(np.mean(log_t_hours))
    x_log_hours = log_t_hours - mean_log_t_hours
    return x_log_hours, mean_log_t_hours


def build_task_x_transform(
    runs_df: pd.DataFrame, task_ids: list[str], x_transform: str, t0_minutes: float
) -> tuple[np.ndarray, dict[str, float]]:
    task_time = runs_df.groupby("task_id", as_index=True)["human_minutes"].first()
    t_minutes = task_time.reindex(task_ids).to_numpy(dtype=float)
    if np.any(t_minutes <= 0) or np.any(~np.isfinite(t_minutes)):
        raise SystemExit("Non-positive or invalid task lengths encountered.")
    if x_transform == "log_hours":
        t_hours = t_minutes / 60.0
        x_raw = np.log(t_hours)
        mean_x = float(np.mean(x_raw))
        x = x_raw - mean_x
        meta = {"mean_x": mean_x}
        return x, meta
    if x_transform == "log1p_minutes":
        if not np.isfinite(t0_minutes) or t0_minutes <= 0:
            raise SystemExit("--t0-minutes must be positive")
        x_raw = np.log1p(t_minutes / t0_minutes)
        mean_x = float(np.mean(x_raw))
        x = x_raw - mean_x
        meta = {"mean_x": mean_x, "t0_minutes": float(t0_minutes)}
        return x, meta
    raise SystemExit(f"Unknown --x-transform {x_transform!r}")


def main() -> None:
    args = parse_args()
    counts_df, item_col = load_counts(args.counts)

    if args.list_models:
        totals = counts_df.groupby("model", as_index=False).agg(n=("n", "sum"), s=("s", "sum"))
        totals["pass_rate"] = totals["s"] / totals["n"]
        totals = totals.sort_values("pass_rate", ascending=True)
        print(totals.to_string(index=False))
        return

    runs_df = pd.read_pickle(args.runs)

    model_names = sorted(counts_df["model"].unique().tolist())
    task_ids = sorted(counts_df[item_col].unique().tolist())
    model_index = {m: i + 1 for i, m in enumerate(model_names)}  # Stan 1-based
    task_index = {t: j + 1 for j, t in enumerate(task_ids)}

    low, high = pick_anchors(counts_df, model_names, args.anchor_low, args.anchor_high)

    x, x_meta = build_task_x_transform(runs_df, task_ids, args.x_transform, args.t0_minutes)

    ii = counts_df["model"].map(model_index).to_numpy(dtype=int)
    jj = counts_df[item_col].map(task_index).to_numpy(dtype=int)

    stan_data: dict[str, object] = {
        "N": int(len(counts_df)),
        "I": int(len(model_names)),
        "J": int(len(task_ids)),
        "ii": ii.tolist(),
        "jj": jj.tolist(),
        "n": counts_df["n"].to_numpy(dtype=int).tolist(),
        "s": counts_df["s"].to_numpy(dtype=int).tolist(),
        "x": x.tolist(),
        "anchor_low": int(model_names.index(low) + 1),
        "anchor_high": int(model_names.index(high) + 1),
        "theta_low": float(args.theta_low),
        "theta_high": float(args.theta_high),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)

    stan_file = (
        Path("analysis/stan/2pl_time_loglinear.stan")
        if args.variant == "b_logt__loga_const"
        else Path("analysis/stan/2pl_time_loglinear_a_logt.stan")
    )
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
        output_dir=str(args.outdir),
    )

    # Save summaries in the same style as fit_2pl_stan.py
    fit.summary().to_csv(args.outdir / "summary.csv")

    def summarize(draws: np.ndarray, names: list[str], out_path: Path, colname: str) -> None:
        mean = draws.mean(axis=0)
        sd = draws.std(axis=0, ddof=1)
        q05, q50, q95 = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
        pd.DataFrame(
            {
                colname: names,
                "mean": mean,
                "sd": sd,
                "q05": q05,
                "q50": q50,
                "q95": q95,
            }
        ).to_csv(out_path, index=False)

    summarize(fit.stan_variable("theta"), model_names, args.outdir / "theta.csv", "model")
    summarize(fit.stan_variable("b"), task_ids, args.outdir / "b.csv", "task_id")
    summarize(fit.stan_variable("a"), task_ids, args.outdir / "a.csv", "task_id")

    scalar_params = {}
    scalar_names = ["alpha", "kappa", "sigma_b", "mu_loga", "sigma_loga"]
    if args.variant == "b_logt__loga_logt":
        scalar_names.insert(4, "eta")
    for name in scalar_names:
        draws = fit.stan_variable(name).reshape(-1)
        scalar_params[name] = {
            "mean": float(draws.mean()),
            "sd": float(draws.std(ddof=1)),
            "q05": float(np.quantile(draws, 0.05)),
            "q50": float(np.quantile(draws, 0.50)),
            "q95": float(np.quantile(draws, 0.95)),
        }

    meta = {
        "counts": str(args.counts),
        "runs": str(args.runs),
        "stan_file": str(stan_file),
        "variant": args.variant,
        "cmdstanpy_version": cmdstanpy.__version__,
        "cmdstan_path": cmdstanpy.cmdstan_path(),
        "chains": args.chains,
        "iter_warmup": args.iter_warmup,
        "iter_sampling": args.iter_sampling,
        "seed": args.seed,
        "adapt_delta": args.adapt_delta,
        "max_treedepth": args.max_treedepth,
        "anchor_low": low,
        "anchor_high": high,
        "theta_low": args.theta_low,
        "theta_high": args.theta_high,
        "x_transform": args.x_transform,
        **x_meta,
        "scalar_params": scalar_params,
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote: {args.outdir / 'theta.csv'}")
    print(f"wrote: {args.outdir / 'summary.csv'}")
    print(f"wrote: {args.outdir / 'meta.json'}")


if __name__ == "__main__":
    main()
