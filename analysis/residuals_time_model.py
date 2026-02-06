#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot residual task difficulty u_j = b_j - (alpha + kappa x_j) for a fitted time model."
    )
    parser.add_argument(
        "--fitdir",
        type=Path,
        default=Path("analysis/out_stan_time_long"),
        help="Fit directory produced by analysis/fit_2pl_time_stan.py (default: analysis/out_stan_time_long).",
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
        default=Path("analysis/out_residuals"),
        help="Output directory (default: analysis/out_residuals).",
    )
    parser.add_argument(
        "--lowess-frac",
        type=float,
        default=0.25,
        help="LOWESS smoothing fraction (default: 0.25).",
    )
    parser.add_argument(
        "--top-families",
        type=int,
        default=12,
        help="Color the top N task_families by #tasks; others are gray (default: 12).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", message=".*joblib.*operate in serial mode.*")
    warnings.filterwarnings("ignore", message=".*joblib will operate in serial mode.*")

    from matplotlib import pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    meta = json.loads((args.fitdir / "meta.json").read_text(encoding="utf-8"))
    scalar = meta["scalar_params"]
    alpha = float(scalar["alpha"]["mean"])
    kappa = float(scalar["kappa"]["mean"])
    x_transform = str(meta.get("x_transform", "log_hours"))
    mean_x = float(meta.get("mean_x", meta.get("mean_log_t_hours", float("nan"))))
    t0_minutes = float(meta.get("t0_minutes", 1.0))
    variant = str(meta.get("variant", ""))

    b_df = pd.read_csv(args.fitdir / "b.csv")
    if "task_id" not in b_df.columns or "mean" not in b_df.columns:
        raise SystemExit(f"Unexpected b.csv schema in {args.fitdir}")
    b_df = b_df[["task_id", "mean"]].rename(columns={"mean": "b_mean"})

    runs_df = pd.read_pickle(args.runs)
    task_time = (
        runs_df.groupby("task_id", as_index=False)
        .agg(t_minutes=("human_minutes", "first"), task_family=("task_family", "first"))
        .rename(columns={"t_minutes": "t_minutes"})
    )
    task_time["t_hours"] = task_time["t_minutes"] / 60.0
    task_time["log_t_hours"] = np.log(task_time["t_hours"])
    if x_transform == "log_hours":
        task_time["x"] = task_time["log_t_hours"] - mean_x
    elif x_transform == "log1p_minutes":
        task_time["x"] = np.log1p(task_time["t_minutes"] / t0_minutes) - mean_x
    else:
        raise SystemExit(f"Unknown x_transform in meta.json: {x_transform!r}")

    df = b_df.merge(task_time[["task_id", "t_hours", "log_t_hours", "x"]], on="task_id", how="inner")
    if df.empty:
        raise SystemExit("No tasks after merging b.csv with runs task lengths")

    df["b_hat"] = alpha + kappa * df["x"]
    df["u_hat"] = df["b_mean"] - df["b_hat"]
    df.to_csv(args.outdir / f"residuals_{args.fitdir.name}.csv", index=False)

    # Also store task_family for diagnostics
    df = b_df.merge(task_time[["task_id", "t_hours", "x", "task_family"]], on="task_id", how="inner")
    df["b_hat"] = alpha + kappa * df["x"]
    df["u_hat"] = df["b_mean"] - df["b_hat"]

    # Scatter u_hat vs x with LOWESS
    df_sorted = df.sort_values("x")
    sm = lowess(df_sorted["u_hat"].to_numpy(), df_sorted["x"].to_numpy(), frac=args.lowess_frac, return_sorted=True)

    plt.figure(figsize=(9.5, 5.5))
    plt.scatter(df["x"], df["u_hat"], s=14, alpha=0.25)
    plt.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2, label="LOWESS")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    if x_transform == "log_hours":
        xlabel = r"Centered log task length  $x=\log(t_\mathrm{hours})-\overline{\log t}$"
    else:
        xlabel = r"Centered transformed length  $x=\log(1+t_\mathrm{min}/t_0)-\overline{x}$"
    plt.xlabel(xlabel)
    plt.ylabel(r"Residual difficulty  $\hat u_j = \hat b_j - (\hat{\alpha} + \hat{\kappa}\,x_j)$")
    plt.title(f"Difficulty residuals vs task length ({args.fitdir.name}{' / ' + variant if variant else ''})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.outdir / f"u_vs_x_{args.fitdir.name}.png", dpi=200)
    plt.close()

    # Color by task_family (top N by task count)
    fam_counts = df["task_family"].value_counts()
    top = fam_counts.head(args.top_families).index.tolist()
    df["family_plot"] = df["task_family"].where(df["task_family"].isin(top), other="other")
    palette = plt.get_cmap("tab20")
    families = [f for f in top if f in set(df["family_plot"])]
    colors = {fam: palette(i % 20) for i, fam in enumerate(families)}

    plt.figure(figsize=(9.5, 5.5))
    # Plot other first
    other = df[df["family_plot"] == "other"]
    plt.scatter(other["x"], other["u_hat"], s=12, alpha=0.12, color="gray", label="other")
    for fam in families:
        sub = df[df["family_plot"] == fam]
        plt.scatter(sub["x"], sub["u_hat"], s=16, alpha=0.40, color=colors[fam], label=f"{fam} (n={len(sub)})")
    plt.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2, label="LOWESS (all)")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(r"Residual difficulty  $\hat u_j$")
    plt.title(f"Residuals by task_family ({args.fitdir.name})")
    plt.legend(loc="best", fontsize=7, frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(args.outdir / f"u_vs_x_family_{args.fitdir.name}.png", dpi=200)
    plt.close()

    # Residuals vs t in hours (log x-axis)
    df_sorted2 = df.sort_values("t_hours")
    sm2 = lowess(df_sorted2["u_hat"].to_numpy(), np.log(df_sorted2["t_hours"].to_numpy()), frac=args.lowess_frac, return_sorted=True)
    plt.figure(figsize=(9.5, 5.5))
    plt.scatter(df["t_hours"], df["u_hat"], s=14, alpha=0.25)
    plt.plot(np.exp(sm2[:, 0]), sm2[:, 1], color="black", linewidth=2, label="LOWESS")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Task length t (hours, log scale)")
    plt.ylabel(r"Residual difficulty  $\hat u_j$")
    plt.title(f"Difficulty residuals vs t (hours) ({args.fitdir.name})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.outdir / f"u_vs_t_{args.fitdir.name}.png", dpi=200)
    plt.close()

    print(f"wrote: {args.outdir / f'u_vs_x_{args.fitdir.name}.png'}")
    print(f"wrote: {args.outdir / f'u_vs_x_family_{args.fitdir.name}.png'}")
    print(f"wrote: {args.outdir / f'u_vs_t_{args.fitdir.name}.png'}")


if __name__ == "__main__":
    main()
