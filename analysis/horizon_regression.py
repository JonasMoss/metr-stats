#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess a fitted 2PL: regress item difficulty on log task length, "
            "compute 50%% horizons per model, and plot horizon vs release date."
        )
    )
    parser.add_argument(
        "--theta-csv",
        type=Path,
        default=Path("analysis/out_stan/theta.csv"),
        help="Theta posterior summary CSV from fit_2pl_stan.py (default: analysis/out_stan/theta.csv).",
    )
    parser.add_argument(
        "--b-csv",
        type=Path,
        default=Path("analysis/out_stan/b.csv"),
        help="Item difficulty posterior summary CSV from fit_2pl_stan.py (default: analysis/out_stan/b.csv).",
    )
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("data/irt_counts_task_id.csv"),
        help="Counts CSV used for IRT fit (for weights) (default: data/irt_counts_task_id.csv).",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/runs.pkl"),
        help="Runs DataFrame pickle produced by scripts/build_data.py (default: data/runs.pkl).",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("data_raw/benchmark_results_1_1.yaml"),
        help="YAML containing model release dates (default: data_raw/benchmark_results_1_1.yaml).",
    )
    parser.add_argument(
        "--item-col",
        choices=["task_id", "task_family"],
        default="task_id",
        help="Which item identifier to use (default: task_id).",
    )
    parser.add_argument(
        "--time-units",
        choices=["hours", "minutes"],
        default="hours",
        help="Units for horizon and plots (default: hours).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/out_horizon"),
        help="Output directory (default: analysis/out_horizon).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=1,
        help="Minimum total attempts for an item to include in b~log(t) regression (default: 1).",
    )
    parser.add_argument(
        "--label-top",
        type=int,
        default=12,
        help="Label this many highest-horizon models in the plot (default: 12).",
    )
    return parser.parse_args()


def load_release_dates(yaml_path: Path) -> dict[str, pd.Timestamp]:
    obj = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    results = obj.get("results", {})
    out: dict[str, pd.Timestamp] = {}
    for model_name, payload in results.items():
        rd = payload.get("release_date")
        if rd is None:
            continue
        ts = pd.to_datetime(rd)
        if pd.isna(ts):
            continue
        out[str(model_name)] = ts
    return out


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Some installations emit noisy joblib warnings via statsmodels; keep output clean.
    warnings.filterwarnings("ignore", message=".*joblib.*operate in serial mode.*")
    warnings.filterwarnings("ignore", message=".*joblib will operate in serial mode.*")

    import statsmodels.api as sm
    from matplotlib import pyplot as plt

    theta_df = pd.read_csv(args.theta_csv)
    b_df = pd.read_csv(args.b_csv)
    counts_df = pd.read_csv(args.counts)
    runs_df = pd.read_pickle(args.runs)

    # Stan summaries name the index column "name".
    theta_df = theta_df.rename(columns={"name": "model"})
    b_df = b_df.rename(columns={"name": args.item_col})

    if args.item_col == "task_id":
        item_time = runs_df.groupby("task_id", as_index=False)["human_minutes"].first()
        item_time = item_time.rename(columns={"human_minutes": "t_minutes"})
    else:
        # For families: use the mean of constituent task times (deterministic in current data).
        item_time = (
            runs_df.groupby(["task_family", "task_id"], as_index=False)["human_minutes"].first()
            .groupby("task_family", as_index=False)["human_minutes"]
            .mean()
            .rename(columns={"human_minutes": "t_minutes"})
        )
        item_time = item_time.rename(columns={"task_family": args.item_col})

    if args.time_units == "hours":
        item_time["t"] = item_time["t_minutes"] / 60.0
        y_label_units = "hours"
    else:
        item_time["t"] = item_time["t_minutes"]
        y_label_units = "minutes"

    item_time["log_t"] = np.log(item_time["t"])

    item_totals = (
        counts_df.groupby(args.item_col, as_index=False)
        .agg(n=("n", "sum"))
        .astype({"n": int})
    )

    b_merge = (
        b_df.merge(item_time[[args.item_col, "t", "log_t"]], on=args.item_col, how="inner")
        .merge(item_totals, on=args.item_col, how="left")
    )
    b_merge["n"] = b_merge["n"].fillna(0).astype(int)
    b_merge = b_merge[b_merge["n"] >= args.min_n].copy()

    if b_merge.empty:
        raise SystemExit("No items left after merging b estimates with task times; check --item-col and input files.")

    # Weighted least squares: weight items by number of attempts (simple, transparent).
    X = sm.add_constant(b_merge["log_t"].to_numpy())
    y = b_merge["mean"].to_numpy()
    w = np.maximum(1.0, b_merge["n"].to_numpy(dtype=float))
    wls = sm.WLS(y, X, weights=w).fit()

    alpha = float(wls.params[0])
    kappa = float(wls.params[1])

    (args.outdir / "b_on_logt_summary.txt").write_text(str(wls.summary()), encoding="utf-8")

    # Compute each model's 50% horizon: solve theta = alpha + kappa*log(t).
    theta = theta_df[["model", "mean", "sd", "q05", "q50", "q95"]].copy()
    theta = theta.rename(columns={"mean": "theta_mean"})
    theta["log_t50"] = (theta["theta_mean"] - alpha) / kappa
    theta["t50"] = np.exp(theta["log_t50"])
    theta["t50_units"] = y_label_units

    release_dates = load_release_dates(args.yaml)
    theta["release_date"] = theta["model"].map(release_dates)

    horizons = theta.dropna(subset=["release_date"]).copy()
    horizons = horizons.sort_values("t50", ascending=False)
    horizons.to_csv(args.outdir / "horizons.csv", index=False)

    # Trend over time (simple): log(t50) ~ c + gamma * days.
    horizons["release_days"] = horizons["release_date"].map(lambda ts: ts.toordinal())
    X2 = sm.add_constant(horizons["release_days"].to_numpy())
    y2 = np.log(horizons["t50"].to_numpy())
    ols = sm.OLS(y2, X2).fit()
    gamma = float(ols.params[1])
    doubling_time_days = float(np.log(2.0) / gamma) if gamma != 0 else float("inf")

    meta = {
        "item_col": args.item_col,
        "time_units": args.time_units,
        "b_on_logt": {"alpha": alpha, "kappa": kappa, "r2": float(wls.rsquared)},
        "logt50_on_date": {"gamma_per_day": gamma, "doubling_time_days": doubling_time_days, "r2": float(ols.rsquared)},
        "counts": str(args.counts),
        "theta_csv": str(args.theta_csv),
        "b_csv": str(args.b_csv),
        "yaml": str(args.yaml),
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Plot 1: b_j vs log t_j with fitted line.
    plt.figure(figsize=(7.5, 5.0))
    plt.scatter(b_merge["t"], b_merge["mean"], s=np.clip(b_merge["n"], 5, 80), alpha=0.5)
    t_grid = np.exp(np.linspace(b_merge["log_t"].min(), b_merge["log_t"].max(), 200))
    b_hat = alpha + kappa * np.log(t_grid)
    plt.plot(t_grid, b_hat, color="black", linewidth=2, label="WLS fit")
    plt.xscale("log")
    plt.xlabel(f"Task length t ({y_label_units}, log scale)")
    plt.ylabel("Estimated item difficulty b")
    plt.title("Difficulty vs task length (post-fit regression)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.outdir / "b_vs_t.png", dpi=200)

    # Plot 2: horizon vs release date (METR-style).
    plt.figure(figsize=(9.5, 5.5))
    plt.scatter(horizons["release_date"], horizons["t50"], s=40, alpha=0.8)
    plt.yscale("log")
    plt.xlabel("Model release date")
    plt.ylabel(f"50% horizon t50 ({y_label_units}, log scale)")
    plt.title("Estimated 50% task-completion horizon vs release date")

    # Trend line.
    x_days = np.linspace(horizons["release_days"].min(), horizons["release_days"].max(), 200)
    y_line = np.exp(ols.params[0] + ols.params[1] * x_days)
    x_dates = [pd.Timestamp.fromordinal(int(d)) for d in x_days]
    plt.plot(x_dates, y_line, color="black", linewidth=2, label=f"Trend (doubling ~ {doubling_time_days:.0f} days)")

    # Label some points (top horizons).
    to_label = horizons.head(args.label_top)
    for _, row in to_label.iterrows():
        plt.annotate(
            row["model"],
            (row["release_date"], row["t50"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.outdir / "horizon_vs_date.png", dpi=200)

    print(f"wrote: {args.outdir / 'horizons.csv'}")
    print(f"wrote: {args.outdir / 'horizon_vs_date.png'}")
    print(f"wrote: {args.outdir / 'b_vs_t.png'}")
    print(f"wrote: {args.outdir / 'meta.json'}")


if __name__ == "__main__":
    main()
