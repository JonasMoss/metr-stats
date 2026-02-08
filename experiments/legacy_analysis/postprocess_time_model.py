#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_MODELS = [
    "gpt_4",
    "gpt_4o_inspect",
    "claude_3_7_sonnet_inspect",
    "o3_inspect",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess the log-linear time model fit: compute posterior p_i(t) curves for selected models "
            "and a METR-style 50% horizon vs release date plot."
        )
    )
    parser.add_argument(
        "--fitdir",
        type=Path,
        default=Path("analysis/out_stan_time"),
        help="Output directory created by analysis/fit_2pl_time_stan.py (default: analysis/out_stan_time).",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("data_raw/benchmark_results_1_1.yaml"),
        help="YAML with release dates (default: data_raw/benchmark_results_1_1.yaml).",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/runs.pkl"),
        help="Runs DataFrame pickle (for time range) (default: data/runs.pkl).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model names for p_i(t) plots (default: {','.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=200,
        help="Number of time grid points (log-spaced) (default: 200).",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=400,
        help="Number of posterior draws to use (subsampled) (default: 400).",
    )
    parser.add_argument(
        "--mc",
        type=int,
        default=80,
        help="Monte Carlo samples per draw for marginal p(t) over task heterogeneity (default: 80).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for subsampling/MC (default: 123).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/out_time_model"),
        help="Directory for outputs (default: analysis/out_time_model).",
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


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def subsample_rows(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if arr.shape[0] <= n:
        return arr
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[idx]


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.fitdir / "meta.json").read_text(encoding="utf-8"))
    x_transform = str(meta.get("x_transform", "log_hours"))
    mean_x = float(meta.get("mean_x", meta.get("mean_log_t_hours", float("nan"))))
    t0_minutes = float(meta.get("t0_minutes", 1.0))

    theta = pd.read_csv(args.fitdir / "theta.csv")
    scalar = meta["scalar_params"]

    # Load raw draws from CmdStan CSV files in fitdir (output_dir set to fitdir).
    import glob

    csv_files = sorted(glob.glob(str(args.fitdir / "*.csv")))
    # Filter out summary/theta/b/a CSVs; keep chain draws files which look like *.csv with "chain" in name.
    chain_files = [p for p in csv_files if "summary.csv" not in p and not p.endswith(("theta.csv", "b.csv", "a.csv"))]
    if not chain_files:
        # cmdstanpy writes chain files as e.g. <model>-<timestamp>.csv; just use cmdstanpy read.
        from cmdstanpy import CmdStanMCMC

        raise SystemExit(f"No chain draw CSVs found in {args.fitdir}. Re-run fit with output_dir set to fitdir.")

    # Read draws via cmdstanpy to avoid manual parsing.
    from cmdstanpy import from_csv

    fit = from_csv(path=chain_files)

    rng = np.random.default_rng(args.seed)

    theta_draws = subsample_rows(fit.stan_variable("theta"), args.draws, rng)  # (S, I)
    alpha_draws = subsample_rows(fit.stan_variable("alpha").reshape(-1, 1), args.draws, rng).reshape(-1)
    kappa_draws = subsample_rows(fit.stan_variable("kappa").reshape(-1, 1), args.draws, rng).reshape(-1)
    sigma_b_draws = subsample_rows(fit.stan_variable("sigma_b").reshape(-1, 1), args.draws, rng).reshape(-1)
    mu_loga_draws = subsample_rows(fit.stan_variable("mu_loga").reshape(-1, 1), args.draws, rng).reshape(-1)
    sigma_loga_draws = subsample_rows(fit.stan_variable("sigma_loga").reshape(-1, 1), args.draws, rng).reshape(-1)
    try:
        eta_all = fit.stan_variable("eta").reshape(-1)
        eta_draws = subsample_rows(eta_all.reshape(-1, 1), args.draws, rng).reshape(-1)
    except Exception:
        eta_draws = np.zeros_like(mu_loga_draws)

    model_names = theta["model"].astype(str).tolist()
    model_index = {m: i for i, m in enumerate(model_names)}
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    missing = [m for m in models if m not in model_index]
    if missing:
        raise SystemExit(f"Models not found in fit: {missing}")

    # Time grid in hours based on task lengths in runs.pkl
    runs_df = pd.read_pickle(args.runs)
    t_hours = runs_df.groupby("task_id")["human_minutes"].first().to_numpy(dtype=float) / 60.0
    t_min = float(np.min(t_hours[t_hours > 0]))
    t_max = float(np.max(t_hours))
    t_grid = np.exp(np.linspace(np.log(t_min), np.log(t_max), args.grid_points))
    if x_transform == "log_hours":
        x_grid = np.log(t_grid) - mean_x
    elif x_transform == "log1p_minutes":
        t_minutes_grid = t_grid * 60.0
        x_grid = np.log1p(t_minutes_grid / t0_minutes) - mean_x
    else:
        raise SystemExit(f"Unknown x_transform in meta.json: {x_transform!r}")

    # Compute curves per model:
    # - typical: u=v=0
    # - marginal: integrate over b ~ N(alpha+kappa*x, sigma_b) and log_a ~ Normal(mu_loga + eta*x, sigma_loga)
    #   (approximate truncation to [-2,2] by clipping normal draws)
    curves = []
    for model in models:
        i = model_index[model]
        th = theta_draws[:, i]  # (S,)

        # Typical curve
        loga_typ = mu_loga_draws[:, None] + eta_draws[:, None] * x_grid[None, :]  # (S, G)
        loga_typ = np.clip(loga_typ, -2.0, 2.0)
        a_typ = np.exp(loga_typ)
        b_typ = alpha_draws[:, None] + kappa_draws[:, None] * x_grid[None, :]  # (S, G)
        eta_typ = a_typ * (th[:, None] - b_typ)
        p_typ = logistic(eta_typ)  # (S, G)

        # Marginal curve (simple MC, chunked over the grid for memory)
        S = len(alpha_draws)
        M = args.mc
        G = len(t_grid)
        a_lo, a_hi = -2.0, 2.0

        u = rng.normal(0.0, sigma_b_draws[:, None], size=(S, M))

        p_marg = np.empty((S, G), dtype=float)
        chunk = 50
        for start in range(0, G, chunk):
            end = min(G, start + chunk)
            x_c = x_grid[start:end]  # (B,)
            B = len(x_c)

            # b mean at each t
            b_mean = (alpha_draws[:, None] + kappa_draws[:, None] * x_c[None, :])  # (S, B)
            b_mc = b_mean[:, None, :] + u[:, :, None]  # (S, M, B)

            # log a mean at each t
            loga_mean = (mu_loga_draws[:, None] + eta_draws[:, None] * x_c[None, :])  # (S, B)
            # Draw loga with per-draw scale and per-t mean, then clip to Stan bounds
            loga = loga_mean[:, None, :] + (sigma_loga_draws[:, None, None] * rng.normal(size=(S, M, B)))
            loga = np.clip(loga, a_lo, a_hi)
            a_mc = np.exp(loga)

            eta = a_mc * (th[:, None, None] - b_mc)
            p_marg[:, start:end] = logistic(eta).mean(axis=1)

        for name, p in [("typical", p_typ), ("marginal", p_marg)]:
            mean = p.mean(axis=0)
            q10, q50, q90 = np.quantile(p, [0.10, 0.50, 0.90], axis=0)
            curves.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "curve": name,
                        "t_hours": t_grid,
                        "mean": mean,
                        "q10": q10,
                        "q50": q50,
                        "q90": q90,
                    }
                )
            )

    curves_df = pd.concat(curves, ignore_index=True)
    curves_df.to_csv(args.outdir / "p_curves.csv", index=False)

    # 50% horizons (typical definition has a closed form).
    # Solve theta = alpha + kappa * x(t) with x(t) defined by x_transform.
    horizons = []
    for idx, model in enumerate(model_names):
        th = theta_draws[:, idx]
        z = mean_x + (th - alpha_draws) / kappa_draws
        if x_transform == "log_hours":
            t50 = np.exp(z)  # hours
        else:
            # z = log1p(t_minutes/t0) => t_minutes = t0*(exp(z)-1)
            t50 = (t0_minutes * (np.exp(z) - 1.0)) / 60.0  # hours
        horizons.append(
            {
                "model": model,
                "t50_hours_mean": float(np.mean(t50)),
                "t50_hours_q10": float(np.quantile(t50, 0.10)),
                "t50_hours_q50": float(np.quantile(t50, 0.50)),
                "t50_hours_q90": float(np.quantile(t50, 0.90)),
            }
        )
    horizons_df = pd.DataFrame(horizons)
    release = load_release_dates(args.yaml)
    horizons_df["release_date"] = horizons_df["model"].map(release)
    horizons_df = horizons_df.dropna(subset=["release_date"]).sort_values("t50_hours_q50", ascending=False)
    horizons_df.to_csv(args.outdir / "horizons.csv", index=False)

    # Plots
    from matplotlib import pyplot as plt

    # p_i(t) curves for selected models (typical + marginal)
    nrows = len(models)
    fig, axes = plt.subplots(nrows, 1, figsize=(9.5, 2.6 * nrows), sharex=True, sharey=True)
    if nrows == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = curves_df[curves_df["model"] == model]
        for curve, color in [("typical", "C0"), ("marginal", "C1")]:
            s = sub[sub["curve"] == curve]
            ax.plot(s["t_hours"], s["q50"], color=color, linewidth=2, label=curve)
            ax.fill_between(s["t_hours"], s["q10"], s["q90"], color=color, alpha=0.15)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(model, loc="left", fontsize=10)
        ax.set_ylabel("p(success)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Task length t (hours, log scale)")
    fig.suptitle("Posterior p_i(t) under log-linear difficulty model", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.outdir / "p_curves.png", dpi=200)
    plt.close(fig)

    # METR-style horizon vs date using median t50 and 10/90 bands
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.5))
    ax.scatter(horizons_df["release_date"], horizons_df["t50_hours_q50"], s=45, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Model release date")
    ax.set_ylabel("50% horizon t50 (hours, log scale)")
    ax.set_title("50% horizon vs release date (posterior median; 10–90% intervals)")

    # Add vertical interval bars
    for _, row in horizons_df.iterrows():
        ax.plot([row["release_date"], row["release_date"]], [row["t50_hours_q10"], row["t50_hours_q90"]], color="black", alpha=0.25)

    to_label = horizons_df.head(10)
    for _, row in to_label.iterrows():
        ax.annotate(
            row["model"],
            (row["release_date"], row["t50_hours_q50"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )
    fig.tight_layout()
    fig.savefig(args.outdir / "horizon_vs_date.png", dpi=200)
    plt.close(fig)

    out_meta = {
        "fitdir": str(args.fitdir),
        "yaml": str(args.yaml),
        "runs": str(args.runs),
        "models": models,
        "grid_points": args.grid_points,
        "draws_used": int(theta_draws.shape[0]),
        "mc_per_draw": args.mc,
        "x_transform": x_transform,
        "mean_x": mean_x,
        "t0_minutes": t0_minutes if x_transform == "log1p_minutes" else None,
        "scalar_posterior_means": {k: v["mean"] for k, v in scalar.items()},
    }
    (args.outdir / "meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    print(f"wrote: {args.outdir / 'p_curves.png'}")
    print(f"wrote: {args.outdir / 'horizon_vs_date.png'}")
    print(f"wrote: {args.outdir / 'horizons.csv'}")


if __name__ == "__main__":
    main()
