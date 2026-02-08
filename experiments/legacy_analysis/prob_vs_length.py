#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "gpt_4",
    "gpt_4o_inspect",
    "claude_3_7_sonnet_inspect",
    "o3_inspect",
    "claude_opus_4_5_inspect",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot plug-in 2PL success probabilities p_ij = logistic(a_j*(theta_i - b_j)) "
            "against task length t_j (log scale)."
        )
    )
    parser.add_argument(
        "--theta-csv",
        type=Path,
        default=Path("analysis/out_stan/theta.csv"),
        help="Theta posterior summary CSV from fit_2pl_stan.py (default: analysis/out_stan/theta.csv).",
    )
    parser.add_argument(
        "--items-csv",
        type=Path,
        default=Path("analysis/out_stan/items.csv"),
        help="Items summary CSV from fit_2pl_stan.py (default: analysis/out_stan/items.csv).",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/runs.pkl"),
        help="Runs DataFrame pickle produced by scripts/build_data.py (default: data/runs.pkl).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model names to plot (default: {','.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=12,
        help="Number of log-length bins for binned curves (default: 12).",
    )
    parser.add_argument(
        "--lowess-frac",
        type=float,
        default=0.25,
        help="LOWESS smoothing fraction for scatter plot (default: 0.25).",
    )
    parser.add_argument(
        "--time-units",
        choices=["hours", "minutes"],
        default="hours",
        help="Units for x-axis task length (default: hours).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/out_probs"),
        help="Output directory for CSV and PNGs (default: analysis/out_probs).",
    )
    return parser.parse_args()


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Keep output clean; statsmodels can emit joblib warnings on some setups.
    warnings.filterwarnings("ignore", message=".*joblib.*operate in serial mode.*")
    warnings.filterwarnings("ignore", message=".*joblib will operate in serial mode.*")

    from matplotlib import pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    theta_df = pd.read_csv(args.theta_csv).rename(columns={"name": "model"})
    items_df = pd.read_csv(args.items_csv)
    runs_df = pd.read_pickle(args.runs)

    # Infer item column name from items.csv. Expect: item_col, mean_b, ..., mean_a, ...
    item_cols = [c for c in items_df.columns if c in {"task_id", "task_family"}]
    if len(item_cols) != 1:
        raise SystemExit(f"Could not infer item column from {args.items_csv} (found {item_cols})")
    item_col = item_cols[0]
    if item_col != "task_id":
        raise SystemExit(
            f"This script currently expects task-level items (task_id). Got item_col={item_col!r}. "
            f"Re-fit Stan on task_id counts so items.csv has a task_id column."
        )

    needed_item_cols = {item_col, "mean_b", "mean_a"}
    missing_item = needed_item_cols - set(items_df.columns)
    if missing_item:
        raise SystemExit(f"{args.items_csv} missing columns: {sorted(missing_item)}")

    needed_theta_cols = {"model", "mean"}
    missing_theta = needed_theta_cols - set(theta_df.columns)
    if missing_theta:
        raise SystemExit(f"{args.theta_csv} missing columns: {sorted(missing_theta)}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("--models must contain at least one model name")

    available_models = set(theta_df["model"].astype(str).tolist())
    missing_models = [m for m in models if m not in available_models]
    if missing_models:
        raise SystemExit(f"Models not found in theta.csv: {missing_models}")

    # Task length (deterministic per task_id in current data).
    task_time = runs_df.groupby("task_id", as_index=False)["human_minutes"].first().rename(columns={"human_minutes": "t_minutes"})
    if args.time_units == "hours":
        task_time["t"] = task_time["t_minutes"] / 60.0
        x_units = "hours"
    else:
        task_time["t"] = task_time["t_minutes"]
        x_units = "minutes"
    task_time["log_t"] = np.log(task_time["t"])

    items = items_df[[item_col, "mean_a", "mean_b"]].merge(task_time[["task_id", "t", "log_t"]], on="task_id", how="inner")
    items = items.rename(columns={"mean_a": "a", "mean_b": "b"})
    if items.empty:
        raise SystemExit("No items after merging items.csv with task times from runs.pkl")

    theta = theta_df[theta_df["model"].isin(models)][["model", "mean"]].rename(columns={"mean": "theta"}).copy()

    # Compute p_ij for selected models and all tasks.
    out_rows: list[pd.DataFrame] = []
    items_np = items[["task_id", "t", "log_t", "a", "b"]].copy()
    for _, row in theta.iterrows():
        model = str(row["model"])
        th = float(row["theta"])
        eta = items_np["a"].to_numpy() * (th - items_np["b"].to_numpy())
        p = logistic(eta)
        out_rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "task_id": items_np["task_id"].to_numpy(),
                    "t": items_np["t"].to_numpy(dtype=float),
                    "log_t": items_np["log_t"].to_numpy(dtype=float),
                    "p": p,
                }
            )
        )

    probs = pd.concat(out_rows, ignore_index=True)
    probs.to_csv(args.outdir / "probs_long.csv", index=False)

    # Plot 1: scatter + LOWESS, one panel per model.
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(9.5, 2.3 * n_models), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = probs[probs["model"] == model].sort_values("t")
        ax.scatter(sub["t"], sub["p"], s=12, alpha=0.25)
        smoothed = lowess(sub["p"].to_numpy(), sub["log_t"].to_numpy(), frac=args.lowess_frac, return_sorted=True)
        t_smooth = np.exp(smoothed[:, 0])
        ax.plot(t_smooth, smoothed[:, 1], color="black", linewidth=2)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("p(success)")
        ax.set_title(model, loc="left", fontsize=10)
    axes[-1].set_xlabel(f"Task length t ({x_units}, log scale)")
    fig.suptitle("Plug-in 2PL predicted success vs task length (scatter + LOWESS)", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.outdir / "scatter_lowess.png", dpi=200)
    plt.close(fig)

    # Plot 2: binned curves overlay.
    # Use global bins so models are comparable.
    bins = np.quantile(items["log_t"].to_numpy(), np.linspace(0, 1, args.bins + 1))
    # Make bins strictly increasing (guard against duplicate edges).
    bins = np.unique(bins)
    if len(bins) < 4:
        raise SystemExit("Not enough distinct task lengths to form bins; try --bins smaller")
    probs["bin"] = pd.cut(probs["log_t"], bins=bins, include_lowest=True)
    binned = (
        probs.groupby(["model", "bin"], as_index=False, observed=False)
        .agg(p_mean=("p", "mean"), t_median=("t", "median"), n_tasks=("task_id", "nunique"))
        .sort_values(["model", "t_median"])
    )

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.5))
    for model in models:
        sub = binned[binned["model"] == model]
        ax.plot(sub["t_median"], sub["p_mean"], marker="o", linewidth=2, label=model)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"Task length t ({x_units}, log scale; bin median)")
    ax.set_ylabel("Mean p(success) in bin")
    ax.set_title("Plug-in 2PL predicted success vs task length (binned means)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "binned_curves.png", dpi=200)
    plt.close(fig)

    print(f"wrote: {args.outdir / 'probs_long.csv'}")
    print(f"wrote: {args.outdir / 'scatter_lowess.png'}")
    print(f"wrote: {args.outdir / 'binned_curves.png'}")


if __name__ == "__main__":
    main()
