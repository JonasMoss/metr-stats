#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Map release date -> ability theta (via least squares), then theta -> horizon t_p under the time-IRT model. "
            "Reports implied doubling time (from slope of log horizon vs date) and writes simple plots."
        )
    )
    p.add_argument(
        "--fitdir",
        type=Path,
        default=Path("analysis/out_stan_time_long"),
        help="Fit directory from analysis/fit_2pl_time_stan.py (default: analysis/out_stan_time_long).",
    )
    p.add_argument(
        "--benchmark-json",
        type=Path,
        default=Path("data/benchmark_results_1_1.json"),
        help="Benchmark results JSON with release dates (default: data/benchmark_results_1_1.json).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/out_trend"),
        help="Output directory for plots/CSVs (default: analysis/out_trend).",
    )
    p.add_argument(
        "--p",
        type=str,
        default="0.5,0.8",
        help="Comma-separated horizon probability levels (default: 0.5,0.8).",
    )
    p.add_argument(
        "--draws",
        type=int,
        default=800,
        help="Posterior draws to use (subsampled) (default: 800).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for subsampling (default: 123).",
    )
    p.add_argument(
        "--a-summary",
        choices=["median", "mean"],
        default="median",
        help="How to plug in discrimination when computing horizons (default: median => exp(mu_loga)).",
    )
    p.add_argument(
        "--include-human",
        action="store_true",
        help="Include the 'human' point if it has a release date in the benchmark file (default: False).",
    )
    return p.parse_args()


def logit(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class FitMeta:
    x_transform: str
    mean_x: float
    t0_minutes: float


def load_fit_meta(fitdir: Path) -> FitMeta:
    meta = json.loads((fitdir / "meta.json").read_text(encoding="utf-8"))
    x_transform = str(meta.get("x_transform", "log_hours"))
    mean_x = float(meta.get("mean_x", meta.get("mean_log_t_hours", float("nan"))))
    if not np.isfinite(mean_x):
        raise SystemExit("fit meta.json missing mean_x / mean_log_t_hours")
    t0_minutes = float(meta.get("t0_minutes", 1.0))
    return FitMeta(x_transform=x_transform, mean_x=mean_x, t0_minutes=t0_minutes)


def load_release_dates(benchmark_json: Path) -> dict[str, pd.Timestamp]:
    obj = json.loads(benchmark_json.read_text(encoding="utf-8"))
    results = obj.get("results", {})
    out: dict[str, pd.Timestamp] = {}
    for name, payload in results.items():
        rd = payload.get("release_date")
        if rd is None:
            continue
        ts = pd.to_datetime(rd, errors="coerce")
        if pd.isna(ts):
            continue
        out[str(name)] = ts
    return out


def chain_csv_files(fitdir: Path) -> list[Path]:
    files = sorted(fitdir.glob("*.csv"))
    keep: list[Path] = []
    for f in files:
        if f.name in {"summary.csv", "theta.csv", "b.csv", "a.csv"}:
            continue
        keep.append(f)
    return keep


def read_draws(fitdir: Path, draws: int, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    theta_df = pd.read_csv(fitdir / "theta.csv")
    model_names = theta_df["model"].astype(str).tolist()
    I = len(model_names)

    theta_cols = [f"theta.{i}" for i in range(1, I + 1)]
    scalar_cols = ["alpha", "kappa", "sigma_b", "mu_loga", "sigma_loga"]

    usecols = scalar_cols + theta_cols
    chain_files = chain_csv_files(fitdir)
    if not chain_files:
        raise SystemExit(f"No chain CSVs found in {fitdir}")

    frames = []
    for f in chain_files:
        frames.append(pd.read_csv(f, comment="#", usecols=usecols))
    all_draws = pd.concat(frames, ignore_index=True)

    rng = np.random.default_rng(seed)
    if len(all_draws) > draws:
        idx = rng.choice(len(all_draws), size=draws, replace=False)
        all_draws = all_draws.iloc[np.sort(idx)].reset_index(drop=True)

    theta = all_draws[theta_cols].to_numpy(dtype=float)
    scalars = {c: all_draws[c].to_numpy(dtype=float) for c in scalar_cols}
    return theta, scalars


def fit_theta_trend(theta: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit theta_i = a + b*x_i per posterior draw (least squares).
    Returns (a, b) arrays of shape (S,).
    """
    x0 = float(np.mean(x))
    xc = x - x0
    denom = float(np.sum(xc * xc))
    if denom <= 0:
        raise SystemExit("Release dates have zero variance; can't fit trend.")

    y = theta  # (S, N)
    b = (y @ xc) / denom  # (S,)
    a = y.mean(axis=1) - b * x0
    return a, b


def horizon_hours_approx(
    theta: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
    sigma_b: np.ndarray,
    mu_loga: np.ndarray,
    sigma_loga: np.ndarray,
    *,
    mean_x: float,
    x_transform: str,
    t0_minutes: float,
    p: float,
    a_summary: str,
) -> np.ndarray:
    """
    Approximate marginal horizon t_p(theta) under:
      b | t ~ Normal(alpha + kappa*x(t), sigma_b)
    by using the logistic-normal approximation (m/s scaling) with a plug-in a.

    Returns array of shape theta.shape (broadcasted), in hours.
    """
    lp = logit(p)
    if a_summary == "median":
        a_eff = np.exp(mu_loga)
    else:
        # mean of lognormal with parameters (mu, sigma)
        a_eff = np.exp(mu_loga + 0.5 * (sigma_loga**2))

    c = (math.pi**2) / 3.0
    # Adjust required margin for random item difficulty u ~ N(0, sigma_b).
    adj = (lp / a_eff) * np.sqrt(1.0 + c * ((a_eff * sigma_b) ** 2))

    b_star = theta - adj
    z = mean_x + (b_star - alpha) / kappa  # z is the *uncentered* x(t)

    if x_transform == "log_hours":
        return np.exp(z)
    if x_transform == "log1p_minutes":
        t_minutes = t0_minutes * (np.exp(z) - 1.0)
        return t_minutes / 60.0
    raise SystemExit(f"Unknown x_transform: {x_transform!r}")


def summarize_over_draws(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90], axis=0)
    return q10, q50, q90


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fit_meta = load_fit_meta(args.fitdir)
    release = load_release_dates(args.benchmark_json)

    theta_draws_all, scalars = read_draws(args.fitdir, args.draws, args.seed)
    model_names = pd.read_csv(args.fitdir / "theta.csv")["model"].astype(str).tolist()

    # Models with release dates
    rows = []
    for i, m in enumerate(model_names):
        if (not args.include_human) and m == "human":
            continue
        if m in release:
            rows.append((i, m, release[m]))
    if not rows:
        raise SystemExit("No models in fit have release dates in benchmark JSON.")

    idx = np.array([r[0] for r in rows], dtype=int)
    names = [r[1] for r in rows]
    dates = pd.to_datetime([r[2] for r in rows])

    # Work in years relative to the mean release date for numerical stability.
    date0 = pd.to_datetime(dates.view("i8").mean()).normalize()
    x = ((dates - date0).days.to_numpy(dtype=float)) / 365.25  # (N,)

    # Fit theta trend per draw
    theta_sub = theta_draws_all[:, idx]  # (S, N)
    a_draws, b_draws = fit_theta_trend(theta_sub, x)

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 120)
    date_grid = (date0 + pd.to_timedelta(x_grid * 365.25, unit="D")).normalize()

    theta_grid = a_draws[:, None] + b_draws[:, None] * x_grid[None, :]  # (S, G)

    # Compute horizons for each p on the grid, and also at observed model dates
    ps = [float(s.strip()) for s in args.p.split(",") if s.strip()]
    horizon_grid_frames = []
    horizon_points_frames = []
    dt_rows = []

    # Doubling time distribution (per draw) for the log-hours transform.
    dt_draws = None
    slope_logt = None
    if fit_meta.x_transform == "log_hours":
        slope_logt = b_draws / scalars["kappa"]  # (S,) per year
        ok = slope_logt > 0
        dt_draws = np.full_like(slope_logt, np.nan)
        dt_draws[ok] = math.log(2.0) / slope_logt[ok]
        pd.DataFrame({"slope_log_horizon_per_year": slope_logt, "doubling_time_years": dt_draws}).to_csv(
            args.outdir / "doubling_time_draws.csv", index=False
        )

    for p in ps:
        t_grid = horizon_hours_approx(
            theta_grid,
            scalars["alpha"][:, None],
            scalars["kappa"][:, None],
            scalars["sigma_b"][:, None],
            scalars["mu_loga"][:, None],
            scalars["sigma_loga"][:, None],
            mean_x=fit_meta.mean_x,
            x_transform=fit_meta.x_transform,
            t0_minutes=fit_meta.t0_minutes,
            p=p,
            a_summary=args.a_summary,
        )
        q10, q50, q90 = summarize_over_draws(t_grid)
        horizon_grid_frames.append(
            pd.DataFrame(
                {
                    "p": p,
                    "x_years_since_mean_release": x_grid,
                    "date": date_grid,
                    "t_hours_q10": q10,
                    "t_hours_q50": q50,
                    "t_hours_q90": q90,
                }
            )
        )

        # Horizon points at observed release dates, using each model's theta (no trend smoothing)
        t_points = horizon_hours_approx(
            theta_sub,
            scalars["alpha"][:, None],
            scalars["kappa"][:, None],
            scalars["sigma_b"][:, None],
            scalars["mu_loga"][:, None],
            scalars["sigma_loga"][:, None],
            mean_x=fit_meta.mean_x,
            x_transform=fit_meta.x_transform,
            t0_minutes=fit_meta.t0_minutes,
            p=p,
            a_summary=args.a_summary,
        )
        q10p, q50p, q90p = summarize_over_draws(t_points)
        horizon_points_frames.append(
            pd.DataFrame(
                {
                    "model": names,
                    "release_date": dates,
                    "p": p,
                    "t_hours_q10": q10p,
                    "t_hours_q50": q50p,
                    "t_hours_q90": q90p,
                }
            )
        )

        # Doubling time from slope of log horizon vs date (per draw).
        # For log_hours, log t is affine in theta, and theta is affine in date => doubling time is constant.
        if fit_meta.x_transform == "log_hours":
            dt_years = dt_draws
            dt_rows.append(
                {
                    "p": p,
                    "doubling_time_years_q10": float(np.nanquantile(dt_years, 0.10)),
                    "doubling_time_years_q50": float(np.nanquantile(dt_years, 0.50)),
                    "doubling_time_years_q90": float(np.nanquantile(dt_years, 0.90)),
                    "slope_log_horizon_per_year_q10": float(np.nanquantile(slope_logt, 0.10)),
                    "slope_log_horizon_per_year_q50": float(np.nanquantile(slope_logt, 0.50)),
                    "slope_log_horizon_per_year_q90": float(np.nanquantile(slope_logt, 0.90)),
                }
            )
        else:
            dt_rows.append(
                {
                    "p": p,
                    "doubling_time_years_q10": float("nan"),
                    "doubling_time_years_q50": float("nan"),
                    "doubling_time_years_q90": float("nan"),
                    "slope_log_horizon_per_year_q10": float("nan"),
                    "slope_log_horizon_per_year_q50": float("nan"),
                    "slope_log_horizon_per_year_q90": float("nan"),
                }
            )

    grid_df = pd.concat(horizon_grid_frames, ignore_index=True)
    points_df = pd.concat(horizon_points_frames, ignore_index=True)
    dt_df = pd.DataFrame(dt_rows)

    grid_df.to_csv(args.outdir / "horizon_grid.csv", index=False)
    points_df.to_csv(args.outdir / "horizon_points.csv", index=False)
    dt_df.to_csv(args.outdir / "doubling_time.csv", index=False)

    # Theta trend summary at model dates (posterior bands).
    theta_obs_q10, theta_obs_q50, theta_obs_q90 = summarize_over_draws(theta_sub)
    theta_points = pd.DataFrame(
        {
            "model": names,
            "release_date": dates,
            "theta_q10": theta_obs_q10,
            "theta_q50": theta_obs_q50,
            "theta_q90": theta_obs_q90,
        }
    ).sort_values("release_date")
    theta_points.to_csv(args.outdir / "theta_points.csv", index=False)

    # Plotting
    from matplotlib import pyplot as plt

    # theta vs date with fitted trend band
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.0))
    theta_points_sorted = theta_points.sort_values("release_date")
    ax.errorbar(
        theta_points_sorted["release_date"],
        theta_points_sorted["theta_q50"],
        yerr=np.vstack(
            [
                theta_points_sorted["theta_q50"] - theta_points_sorted["theta_q10"],
                theta_points_sorted["theta_q90"] - theta_points_sorted["theta_q50"],
            ]
        ),
        fmt="o",
        color="black",
        ecolor="gray",
        elinewidth=1,
        capsize=2,
        alpha=0.85,
        label="model θ (posterior median; 10–90%)",
    )

    theta_grid_q10, theta_grid_q50, theta_grid_q90 = summarize_over_draws(theta_grid)
    ax.plot(date_grid, theta_grid_q50, color="C0", linewidth=2, label="θ trend (date → θ)")
    ax.fill_between(date_grid, theta_grid_q10, theta_grid_q90, color="C0", alpha=0.15)
    ax.set_xlabel("Release date")
    ax.set_ylabel("Ability θ (anchored)")
    ax.set_title("Ability vs release date (fit in θ-space)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "theta_vs_date.png", dpi=200)
    plt.close(fig)

    # Horizons vs date (per p)
    for p in ps:
        g = grid_df[grid_df["p"] == p].sort_values("date")
        pts = points_df[points_df["p"] == p].sort_values("release_date")
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2))
        ax.plot(g["date"], g["t_hours_q50"], color="C1", linewidth=2, label="trend-based horizon")
        ax.fill_between(g["date"], g["t_hours_q10"], g["t_hours_q90"], color="C1", alpha=0.15)

        ax.scatter(pts["release_date"], pts["t_hours_q50"], s=35, color="black", alpha=0.8, label="model horizon (median)")
        for _, r in pts.iterrows():
            ax.plot([r["release_date"], r["release_date"]], [r["t_hours_q10"], r["t_hours_q90"]], color="black", alpha=0.20)

        ax.set_yscale("log")
        ax.set_xlabel("Release date")
        ax.set_ylabel(f"Horizon t{int(round(100*p))} (hours, log scale)")
        ax.set_title(f"Horizon vs release date (p={p:g}; approx-marginal over task difficulty)")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.outdir / f"horizon_p{int(round(100*p)):02d}.png", dpi=200)
        plt.close(fig)

    # Combined horizons plot (all p values on one axis)
    if len(ps) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.4))
        colors = [f"C{i}" for i in range(10)]
        for k, p in enumerate(sorted(ps)):
            g = grid_df[grid_df["p"] == p].sort_values("date")
            pts = points_df[points_df["p"] == p].sort_values("release_date")
            color = colors[k % len(colors)]
            label = f"trend t{int(round(100*p))} (median; 10–90%)"
            ax.plot(g["date"], g["t_hours_q50"], color=color, linewidth=2, label=label)
            ax.fill_between(g["date"], g["t_hours_q10"], g["t_hours_q90"], color=color, alpha=0.12)
            ax.scatter(pts["release_date"], pts["t_hours_q50"], s=18, color=color, alpha=0.55)
        ax.set_yscale("log")
        ax.set_xlabel("Release date")
        ax.set_ylabel("Horizon t_p (hours, log scale)")
        ax.set_title("Horizon vs release date (multiple p levels; approx-marginal over task difficulty)")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.outdir / "horizons_multi.png", dpi=200)
        plt.close(fig)

    # Doubling time distribution plot (log-hours transform only)
    if dt_draws is not None:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))
        vals = dt_draws[np.isfinite(dt_draws)]
        ax.hist(vals, bins=40, color="C2", alpha=0.85, density=True)
        q10, q50, q90 = np.quantile(vals, [0.10, 0.50, 0.90])
        ax.axvline(q50, color="black", linewidth=2, label=f"median={q50:.3g}y (10–90%: {q10:.3g}–{q90:.3g})")
        ax.axvline(q10, color="black", linewidth=1, linestyle="--")
        ax.axvline(q90, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Doubling time (years)")
        ax.set_ylabel("Posterior density")
        ax.set_title("Implied horizon doubling time (from θ trend and time model)")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.outdir / "doubling_time_posterior.png", dpi=200)
        plt.close(fig)

    out_meta = {
        "fitdir": str(args.fitdir),
        "benchmark_json": str(args.benchmark_json),
        "outdir": str(args.outdir),
        "p": ps,
        "draws_used": int(theta_draws_all.shape[0]),
        "a_summary": args.a_summary,
        "x_transform": fit_meta.x_transform,
        "mean_x": fit_meta.mean_x,
        "t0_minutes": fit_meta.t0_minutes if fit_meta.x_transform == "log1p_minutes" else None,
        "models_with_release_dates": names,
    }
    (args.outdir / "meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    print(f"wrote: {args.outdir / 'theta_vs_date.png'}")
    for p in ps:
        print(f"wrote: {args.outdir / f'horizon_p{int(round(100*p)):02d}.png'}")
    if len(ps) > 1:
        print(f"wrote: {args.outdir / 'horizons_multi.png'}")
    if dt_draws is not None:
        print(f"wrote: {args.outdir / 'doubling_time_posterior.png'}")
    print(f"wrote: {args.outdir / 'doubling_time.csv'}")


if __name__ == "__main__":
    main()
