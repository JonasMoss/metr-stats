#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "gpt_4",
    "gpt_4o_inspect",
    "claude_3_7_sonnet_inspect",
    "o3_inspect",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple calibration diagnostics for binomial outcomes s_ij ~ Binomial(n_ij, p_ij). "
            "Compares full model plug-in predictions p_ij (using task-level a_j,b_j) against "
            "time-only predictions p_i(t_j) (typical and marginal)."
        )
    )
    parser.add_argument(
        "--fitdir",
        type=Path,
        default=Path("analysis/out_stan_time_long"),
        help="Fit directory produced by analysis/fit_2pl_time_stan.py (default: analysis/out_stan_time_long).",
    )
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("data/irt_counts_task_id.csv"),
        help="Counts CSV with columns model,task_id,n,s (default: data/irt_counts_task_id.csv).",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/runs.pkl"),
        help="Runs DataFrame pickle (for task lengths) (default: data/runs.pkl).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated models to plot (default: {','.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of bins for reliability curves (default: 10).",
    )
    parser.add_argument(
        "--mc",
        type=int,
        default=400,
        help="Monte Carlo samples for time-marginal p_i(t) (default: 400).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for MC integration (default: 123).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/out_calibration"),
        help="Output directory (default: analysis/out_calibration).",
    )
    return parser.parse_args()


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clip_loga(loga: np.ndarray) -> np.ndarray:
    return np.clip(loga, -2.0, 2.0)


def load_fit_means(fitdir: Path) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = json.loads((fitdir / "meta.json").read_text(encoding="utf-8"))
    scalar = meta["scalar_params"]
    scalars = {k: float(v["mean"]) for k, v in scalar.items()}

    theta = pd.read_csv(fitdir / "theta.csv")
    if "model" not in theta.columns or "mean" not in theta.columns:
        raise SystemExit(f"Unexpected schema: {fitdir / 'theta.csv'}")
    theta = theta[["model", "mean"]].rename(columns={"mean": "theta"})

    a = pd.read_csv(fitdir / "a.csv")
    b = pd.read_csv(fitdir / "b.csv")
    if not ({"task_id", "mean"} <= set(a.columns) and {"task_id", "mean"} <= set(b.columns)):
        raise SystemExit("Unexpected schema for a.csv/b.csv")
    a = a[["task_id", "mean"]].rename(columns={"mean": "a"})
    b = b[["task_id", "mean"]].rename(columns={"mean": "b"})

    extra = {
        "variant": str(meta.get("variant", "")),
        "x_transform": str(meta.get("x_transform", "log_hours")),
        "mean_x": float(meta.get("mean_x", meta.get("mean_log_t_hours", float("nan")))),
        "t0_minutes": float(meta.get("t0_minutes", 1.0)),
    }
    scalars.update(extra)
    return scalars, theta, a, b


def task_lengths(runs_pkl: Path) -> pd.DataFrame:
    runs = pd.read_pickle(runs_pkl)
    tt = runs.groupby("task_id", as_index=False).agg(t_minutes=("human_minutes", "first"))
    tt["t_hours"] = tt["t_minutes"] / 60.0
    if (tt["t_minutes"] <= 0).any():
        raise SystemExit("Non-positive task lengths found")
    return tt


def compute_x(tt: pd.DataFrame, x_transform: str, mean_x: float, t0_minutes: float) -> pd.DataFrame:
    tt = tt.copy()
    if x_transform == "log_hours":
        tt["x"] = np.log(tt["t_hours"]) - mean_x
    elif x_transform == "log1p_minutes":
        tt["x"] = np.log1p(tt["t_minutes"] / t0_minutes) - mean_x
    else:
        raise SystemExit(f"Unknown x_transform: {x_transform!r}")
    return tt


def binomial_brier_per_attempt(n: np.ndarray, s: np.ndarray, p: np.ndarray) -> float:
    # Mean squared error per attempt, aggregated by cells:
    # (1/n) * sum_k (y_k - p)^2 = (s*(1-p)^2 + (n-s)*p^2)/n
    per_cell = (s * (1.0 - p) ** 2 + (n - s) * p**2) / n
    return float(np.average(per_cell, weights=n))


def binomial_log_score_per_attempt(n: np.ndarray, s: np.ndarray, p: np.ndarray) -> float:
    # Average log probability per attempt (up to constants); stable form:
    # (s*log p + (n-s)*log(1-p)) / n
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    per_cell = (s * np.log(p) + (n - s) * np.log(1.0 - p)) / n
    return float(np.average(per_cell, weights=n))


def reliability_curve(n: np.ndarray, s: np.ndarray, p: np.ndarray, bins: int) -> pd.DataFrame:
    # Bin by predicted p (quantiles), then compute weighted averages.
    df = pd.DataFrame({"n": n, "s": s, "p": p})
    # Guard against identical predictions
    q = np.quantile(df["p"], np.linspace(0, 1, bins + 1))
    q = np.unique(q)
    if len(q) < 3:
        # fallback to equal-width
        q = np.linspace(df["p"].min(), df["p"].max(), bins + 1)
    df["bin"] = pd.cut(df["p"], bins=q, include_lowest=True, duplicates="drop")
    g = df.groupby("bin", observed=False, as_index=False).agg(n=("n", "sum"), s=("s", "sum"), p_mean=("p", lambda x: float(np.average(x, weights=df.loc[x.index, "n"]))))
    g["obs_rate"] = g["s"] / g["n"]
    # Approx SE for a binomial proportion (for quick error bars)
    g["se"] = np.sqrt(g["obs_rate"] * (1.0 - g["obs_rate"]) / g["n"])
    return g


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    scalars, theta_df, a_df, b_df = load_fit_means(args.fitdir)
    counts = pd.read_csv(args.counts)
    if not {"model", "task_id", "n", "s"} <= set(counts.columns):
        raise SystemExit("--counts must have columns model,task_id,n,s")
    counts = counts[["model", "task_id", "n", "s"]].copy()

    tt = compute_x(task_lengths(args.runs), scalars["x_transform"], scalars["mean_x"], scalars["t0_minutes"])

    # Join everything at the cell level (model, task_id).
    cells = (
        counts.merge(theta_df, on="model", how="inner")
        .merge(a_df, on="task_id", how="inner")
        .merge(b_df, on="task_id", how="inner")
        .merge(tt[["task_id", "t_hours", "t_minutes", "x"]], on="task_id", how="inner")
    )
    if cells.empty:
        raise SystemExit("No rows after merging counts with fit outputs")

    n = cells["n"].to_numpy(dtype=float)
    s = cells["s"].to_numpy(dtype=float)
    theta = cells["theta"].to_numpy(dtype=float)
    a_task = cells["a"].to_numpy(dtype=float)
    b_task = cells["b"].to_numpy(dtype=float)

    # Full plug-in predictions (task-level a_j,b_j)
    p_full = logistic(a_task * (theta - b_task))
    cells["p_full"] = p_full

    # Time-only predictions
    alpha = float(scalars["alpha"])
    kappa = float(scalars["kappa"])
    sigma_b = float(scalars["sigma_b"])
    mu_loga = float(scalars["mu_loga"])
    sigma_loga = float(scalars["sigma_loga"])
    eta = float(scalars.get("eta", 0.0))

    b_time = alpha + kappa * cells["x"].to_numpy(dtype=float)
    loga_time = clip_loga(mu_loga + eta * cells["x"].to_numpy(dtype=float))
    a_time = np.exp(loga_time)
    p_time_typ = logistic(a_time * (theta - b_time))
    cells["p_time_typical"] = p_time_typ

    # Time-marginal: integrate over u ~ N(0,sigma_b) and loga ~ N(mu_loga+eta*x, sigma_loga) truncated [-2,2]
    rng = np.random.default_rng(args.seed)
    M = int(args.mc)
    # Precompute per-task marginal predictions for each model to avoid repeated work.
    model_names = theta_df["model"].astype(str).tolist()
    model_index = {m: i for i, m in enumerate(model_names)}
    theta_by_model = theta_df.set_index("model")["theta"].to_dict()

    # Task-specific x and b/loga means
    task_tbl = tt[["task_id", "t_hours", "t_minutes", "x"]].copy()
    task_tbl["b_mean"] = alpha + kappa * task_tbl["x"]
    task_tbl["loga_mean"] = mu_loga + eta * task_tbl["x"]

    # Build a prediction table keyed by (model, task_id)
    preds = []
    for _, row in task_tbl.iterrows():
        task_id = row["task_id"]
        b0 = float(row["b_mean"])
        loga0 = float(row["loga_mean"])

        u = rng.normal(0.0, sigma_b, size=M)
        loga = rng.normal(loga0, sigma_loga, size=M)
        loga = clip_loga(loga)
        a_mc = np.exp(loga)

        # For all models at once
        th = np.array([theta_by_model[m] for m in model_names], dtype=float)  # (I,)
        eta_mc = a_mc[:, None] * (th[None, :] - (b0 + u)[:, None])  # (M, I)
        p_m = logistic(eta_mc).mean(axis=0)  # (I,)

        preds.append(pd.DataFrame({"task_id": task_id, "model": model_names, "p_time_marginal": p_m}))
    pred_time = pd.concat(preds, ignore_index=True)
    cells = cells.merge(pred_time, on=["model", "task_id"], how="left")

    # Metrics (overall and per model)
    methods = ["p_full", "p_time_typical", "p_time_marginal"]
    metrics_rows = []
    for method in methods:
        p = cells[method].to_numpy(dtype=float)
        metrics_rows.append(
            {
                "scope": "overall",
                "method": method,
                "brier_per_attempt": binomial_brier_per_attempt(n, s, p),
                "log_score_per_attempt": binomial_log_score_per_attempt(n, s, p),
            }
        )
        for model, g in cells.groupby("model", as_index=False):
            n_m = g["n"].to_numpy(dtype=float)
            s_m = g["s"].to_numpy(dtype=float)
            p_m = g[method].to_numpy(dtype=float)
            metrics_rows.append(
                {
                    "scope": str(model),
                    "method": method,
                    "brier_per_attempt": binomial_brier_per_attempt(n_m, s_m, p_m),
                    "log_score_per_attempt": binomial_log_score_per_attempt(n_m, s_m, p_m),
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.outdir / "metrics.csv", index=False)

    cells_out = cells[["model", "task_id", "n", "s", "t_hours", "p_full", "p_time_typical", "p_time_marginal"]]
    cells_out.to_csv(args.outdir / "cell_predictions.csv", index=False)

    # Plots: reliability for selected models, overlaying methods.
    from matplotlib import pyplot as plt

    plot_models = [m.strip() for m in args.models.split(",") if m.strip()]
    plot_models = [m for m in plot_models if m in set(cells["model"].unique())]
    if not plot_models:
        plot_models = sorted(cells["model"].unique())[:4]

    fig, axes = plt.subplots(len(plot_models), 1, figsize=(8.5, 2.6 * len(plot_models)), sharex=True, sharey=True)
    if len(plot_models) == 1:
        axes = [axes]

    colors = {"p_full": "C0", "p_time_typical": "C1", "p_time_marginal": "C2"}
    labels = {"p_full": "full plug-in", "p_time_typical": "time-only typical", "p_time_marginal": "time-only marginal"}

    for ax, model in zip(axes, plot_models):
        sub = cells[cells["model"] == model]
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        for method in methods:
            g = reliability_curve(
                n=sub["n"].to_numpy(dtype=float),
                s=sub["s"].to_numpy(dtype=float),
                p=sub[method].to_numpy(dtype=float),
                bins=args.bins,
            )
            ax.errorbar(
                g["p_mean"],
                g["obs_rate"],
                yerr=1.96 * g["se"],
                fmt="o-",
                markersize=4,
                linewidth=1.5,
                color=colors[method],
                alpha=0.9,
                label=labels[method],
            )
        ax.set_title(model, loc="left", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Observed success rate")
        ax.legend(loc="lower right", fontsize=7)

    axes[-1].set_xlabel("Mean predicted probability (bin)")
    fig.suptitle("Calibration (reliability) curves by model", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.outdir / "calibration_by_model.png", dpi=200)
    plt.close(fig)

    # Overall calibration (all cells pooled)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    for method in methods:
        g = reliability_curve(n=n, s=s, p=cells[method].to_numpy(dtype=float), bins=args.bins)
        ax.errorbar(
            g["p_mean"],
            g["obs_rate"],
            yerr=1.96 * g["se"],
            fmt="o-",
            markersize=5,
            linewidth=2,
            color=colors[method],
            alpha=0.95,
            label=labels[method],
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed success rate")
    ax.set_title("Overall calibration (pooled)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "calibration_overall.png", dpi=200)
    plt.close(fig)

    out_meta = {
        "fitdir": str(args.fitdir),
        "counts": str(args.counts),
        "runs": str(args.runs),
        "variant": scalars.get("variant"),
        "x_transform": scalars.get("x_transform"),
        "bins": args.bins,
        "mc": args.mc,
        "seed": args.seed,
    }
    (args.outdir / "meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    print(f"wrote: {args.outdir / 'calibration_overall.png'}")
    print(f"wrote: {args.outdir / 'calibration_by_model.png'}")
    print(f"wrote: {args.outdir / 'metrics.csv'}")


if __name__ == "__main__":
    main()

