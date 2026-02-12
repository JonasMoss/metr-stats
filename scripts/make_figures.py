#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_MODELS = ["gpt_4", "gpt_4o_inspect", "claude_3_7_sonnet_inspect", "o3_inspect"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate the core figures for the blog-post pipeline: posterior p_i(t) curves, release-date trend in θ, "
            "horizons vs release date, and an implied doubling-time distribution."
        )
    )
    p.add_argument("--fitdir", type=Path, default=None, help="Fit directory containing meta.json and CmdStan CSVs.")
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--run-id", type=str, default=None, help="Use a run under --runs-root/<run-id>/fit.")
    p.add_argument(
        "--spec",
        type=str,
        default=None,
        help="Model spec folder under --runs-root (default: infer from --fitdir or use time_irt__theta_none).",
    )
    p.add_argument("--outdir", type=Path, default=None, help="Directory to write figures/CSVs (default: run figures dir).")

    p.add_argument("--runs", type=Path, default=Path("data/runs.pkl"))
    p.add_argument("--benchmark-yaml", type=Path, default=Path("data_raw/benchmark_results_1_1.yaml"))

    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--p", type=str, default="0.5,0.8", help="Comma-separated horizon probability levels (default: 0.5,0.8).")

    p.add_argument("--grid-points", type=int, default=200, help="Time grid points for p_i(t) curves (default: 200).")
    p.add_argument("--draws", type=int, default=600, help="Posterior draws to use (subsampled) (default: 600).")
    p.add_argument("--mc", type=int, default=80, help="MC samples per draw for marginal p_i(t) curves (default: 80).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed (default: 123).")
    p.add_argument(
        "--extend-to-year",
        type=int,
        default=2032,
        help="Extend horizon-vs-date trend lines to the end of this year (default: 2032).",
    )
    p.add_argument(
        "--label-preset",
        choices=["none", "blog"],
        default="none",
        help="Optional label preset for horizon plots (default: none).",
    )
    p.add_argument(
        "--label-models",
        type=str,
        default="",
        help="Comma-separated model keys to label on horizon plots (overrides --label-preset).",
    )
    return p.parse_args()


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")
    return math.log(p / (1.0 - p))


def _format_duration_hours(hours: float) -> str:
    if not np.isfinite(hours) or hours <= 0:
        return ""
    seconds = hours * 3600.0
    if seconds < 60:
        v = int(round(seconds))
        return f"{v} sec"
    minutes = seconds / 60.0
    if minutes < 60:
        v = int(round(minutes))
        return f"{v} min"
    if hours < 24:
        v = int(round(hours))
        return f"{v} hour" if v == 1 else f"{v} hours"
    days = hours / 24.0
    if days >= 365:
        years = int(round(days / 365.0))
        return f"{years} year" if years == 1 else f"{years} years"
    if days >= 30:
        months = int(round(days / 30.0))
        return f"{months} month" if months == 1 else f"{months} months"
    v = int(round(days))
    return f"{v} day" if v == 1 else f"{v} days"


def apply_metr_duration_ticks(ax, y_values_hours: np.ndarray) -> None:
    """
    METR-style y-axis with human-friendly tick labels and a roughly x9–x10 spacing.

    We use a fixed "ladder" of rounded durations (in hours) similar to METR plots, but
    with calendar-style months/years:
      4 sec, 36 sec, 6 min, 1 hour, 10 hours, 4 days, 1 month, 6 months, 3 years, 18 years, 50 years, 100 years, 500 years

    Then we keep the subset that spans the plotted y-range.
    """
    y = np.asarray(y_values_hours, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if y.size == 0:
        return
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymin <= 0 or ymax <= 0:
        return

    # Define ladder in hours. Use 30-day months and 365-day years.
    ticks = np.array(
        [
            4 / 3600,  # 4 sec
            36 / 3600,  # 36 sec
            6 / 60,  # 6 min
            1.0,  # 1 hour
            10.0,  # 10 hours
            4 * 24.0,  # 4 days
            30 * 24.0,  # 1 month (30 days)
            6 * 30 * 24.0,  # 6 months
            3 * 365 * 24.0,  # 3 years
            18 * 365 * 24.0,  # 18 years
            50 * 365 * 24.0,  # 50 years
            100 * 365 * 24.0,  # 100 years
            500 * 365 * 24.0,  # 500 years
        ],
        dtype=float,
    )

    lo = ymin / 1.15
    hi = ymax * 1.15
    keep = ticks[(ticks >= lo) & (ticks <= hi)]
    if keep.size < 3:
        # Expand by one neighbor on each side if needed.
        below = ticks[ticks < lo]
        above = ticks[ticks > hi]
        candidates = []
        if below.size:
            candidates.append(below[-1])
        candidates.extend(keep.tolist())
        if above.size:
            candidates.append(above[0])
        keep = np.array(candidates, dtype=float)
    if keep.size == 0:
        return

    ax.set_yticks(keep)
    ax.set_yticklabels([_format_duration_hours(float(t)) for t in keep])
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.25)


def pretty_model_label(model_key: str) -> str:
    mapping = {
        "claude_opus_4_5_inspect": "Opus 4.5",
        "claude_3_5_sonnet_20241022_inspect": "Sonnet 3.5 (new)",
        "claude_3_5_sonnet_20240620_inspect": "Sonnet 3.5 (old)",
        "o3_inspect": "o3",
    }
    return mapping.get(model_key, model_key)


def annotate_models(ax, pts: pd.DataFrame, model_keys: list[str]) -> None:
    if not model_keys:
        return
    if pts.empty:
        return

    sub = pts[pts["model"].isin(model_keys)].copy()
    if sub.empty:
        return

    # Stable offsets (points) to reduce collisions a bit.
    offsets = [(6, 6), (6, -10), (6, 14), (6, -18), (10, 8), (10, -12)]
    for i, (_, r) in enumerate(sub.sort_values("release_date").iterrows()):
        label = pretty_model_label(str(r["model"]))
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            label,
            (r["release_date"], r["t_hours_q50"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
        )


def gaussian_kde_1d(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Simple Gaussian KDE with a plug-in (Silverman) bandwidth.

    Returns an estimate of the density on `grid`.
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.full_like(grid, np.nan, dtype=float)

    n = float(x.size)
    sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.full_like(grid, np.nan, dtype=float)

    h = 1.06 * sd * (n ** (-1.0 / 5.0))
    h = max(h, 1e-6)

    z = (grid[:, None] - x[None, :]) / h
    dens = np.mean(np.exp(-0.5 * z * z), axis=1) / (h * math.sqrt(2.0 * math.pi))
    return dens


def load_release_dates(yaml_path: Path) -> dict[str, pd.Timestamp]:
    obj = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    results = obj.get("results", {})
    out: dict[str, pd.Timestamp] = {}
    for model_name, payload in results.items():
        rd = payload.get("release_date")
        if rd is None:
            continue
        ts = pd.to_datetime(rd, errors="coerce")
        if pd.isna(ts):
            continue
        out[str(model_name)] = ts
    return out


def chain_csv_files(fitdir: Path) -> list[Path]:
    files = sorted(fitdir.glob("*.csv"))
    keep: list[Path] = []
    for f in files:
        if f.name in {"summary.csv", "theta.csv", "b.csv", "a.csv"}:
            continue
        keep.append(f)
    return keep


def read_draws_subset(fitdir: Path, draws: int, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    meta = json.loads((fitdir / "meta.json").read_text(encoding="utf-8"))
    model_names = pd.read_csv(fitdir / "theta.csv")["model"].astype(str).tolist()
    I = len(model_names)
    theta_cols = [f"theta.{i}" for i in range(1, I + 1)]
    scalar_cols = ["alpha", "kappa", "sigma_b", "mu_loga", "sigma_loga"]

    extra_cols: list[str] = []
    trend = str(meta.get("theta_trend", "none"))
    if trend in {"linear", "quadratic", "quadratic_pos", "sqrt", "log1p"}:
        K = int(meta.get("theta_trend_K", 0))
        if K <= 0:
            raise SystemExit("meta.json indicates theta_trend but theta_trend_K is missing/invalid")
        extra_cols = ["gamma0", "sigma_theta"] + [f"gamma.{k}" for k in range(1, K + 1)]
    elif trend == "xpow":
        extra_cols = ["gamma0", "gamma1", "a_pow", "sigma_theta"]
    elif trend == "t50_logistic":
        extra_cols = ["log_t_low", "log_delta_t", "a_t", "b_t", "sigma_theta"]
    elif trend == "theta_logistic":
        extra_cols = ["theta_min", "theta_range", "a_logis", "b_logis", "sigma_theta"]
    elif trend in {"singularity", "singularity_nolinear"}:
        if trend == "singularity":
            extra_cols = ["gamma0", "gamma1", "c_sing", "alpha_sing", "eta_tstar", "sigma_theta"]
        else:
            extra_cols = ["gamma0", "c_sing", "alpha_sing", "eta_tstar", "sigma_theta"]
    usecols = scalar_cols + theta_cols + extra_cols

    chain_files = chain_csv_files(fitdir)
    if not chain_files:
        raise SystemExit(f"No chain CSVs found in {fitdir}")

    frames = [pd.read_csv(f, comment="#", usecols=usecols) for f in chain_files]
    all_draws = pd.concat(frames, ignore_index=True)

    rng = np.random.default_rng(seed)
    if len(all_draws) > draws:
        idx = rng.choice(len(all_draws), size=draws, replace=False)
        all_draws = all_draws.iloc[np.sort(idx)].reset_index(drop=True)

    meta_out = {
        **meta,
        "model_names": model_names,
        "draws_used": int(len(all_draws)),
    }
    return all_draws, meta_out


def time_grid_from_runs(runs_df: pd.DataFrame, grid_points: int) -> np.ndarray:
    t_hours = runs_df.groupby("task_id")["human_minutes"].first().to_numpy(dtype=float) / 60.0
    t_hours = t_hours[np.isfinite(t_hours) & (t_hours > 0)]
    if len(t_hours) == 0:
        raise SystemExit("No valid human_minutes found for task lengths.")
    t_min = float(np.min(t_hours))
    t_max = float(np.max(t_hours))
    return np.exp(np.linspace(np.log(t_min), np.log(t_max), grid_points))


def horizon_hours_approx(
    theta: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
    sigma_b: np.ndarray,
    mu_loga: np.ndarray,
    *,
    mean_log_t_hours: float,
    p: float,
) -> np.ndarray:
    """
    Approximate marginal horizon t_p(theta) for the log-hours transform, using
    a logistic-normal approximation for u ~ N(0, sigma_b) and a plug-in a = exp(mu_loga).
    """
    lp = logit(p)
    a_eff = np.exp(mu_loga)
    c = (math.pi**2) / 3.0
    adj = (lp / a_eff) * np.sqrt(1.0 + c * ((a_eff * sigma_b) ** 2))
    b_star = theta - adj
    z = mean_log_t_hours + (b_star - alpha) / kappa  # log(hours)
    # Avoid overflow in extreme extrapolations (e.g. singularity trend).
    z = np.clip(z, -80.0, 80.0)
    return np.exp(z)


def horizon_hours_typical(
    theta: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
    mu_loga: np.ndarray,
    *,
    mean_log_t_hours: float,
    p: float,
) -> np.ndarray:
    """
    "Typical" horizon: does not integrate over task random effects.

    Uses plug-in discrimination a = exp(mu_loga) and b(t) = alpha + kappa * (log t - mean_log_t_hours).
    """
    lp = logit(p)
    a_eff = np.exp(mu_loga)
    b_star = theta - (lp / a_eff)
    z = mean_log_t_hours + (b_star - alpha) / kappa  # log(hours)
    z = np.clip(z, -80.0, 80.0)
    return np.exp(z)


def summarize_draws(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10, q50, q90 = np.quantile(arr, [0.10, 0.50, 0.90], axis=0)
    return q10, q50, q90


def main() -> None:
    args = parse_args()

    if args.fitdir is None:
        spec = args.spec or "time_irt__theta_none"
        run_id = args.run_id or (args.runs_root / spec / "LATEST").read_text(encoding="utf-8").strip()
        args.fitdir = args.runs_root / spec / run_id / "fit"
    else:
        # .../runs/<spec>/<run_id>/fit
        if args.fitdir.name == "fit":
            run_id = args.fitdir.parent.name
            spec = args.fitdir.parent.parent.name
        else:
            run_id = args.fitdir.name
            spec = args.fitdir.parent.name

    if args.outdir is None:
        if args.fitdir.name == "fit":
            args.outdir = args.fitdir.parent / "figures"
        else:
            args.outdir = args.fitdir / "figures"
    args.outdir.mkdir(parents=True, exist_ok=True)

    draws_df, meta = read_draws_subset(args.fitdir, args.draws, args.seed)
    model_names: list[str] = meta["model_names"]
    model_index = {m: i for i, m in enumerate(model_names)}

    mean_log_t_hours = float(meta.get("mean_log_t_hours", float("nan")))
    if not np.isfinite(mean_log_t_hours):
        raise SystemExit("fit meta.json missing mean_log_t_hours")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    missing = [m for m in models if m not in model_index]
    if missing:
        raise SystemExit(f"Models not found in fit: {missing}")

    ps = [float(s.strip()) for s in args.p.split(",") if s.strip()]

    if args.label_models.strip():
        label_models = [s.strip() for s in args.label_models.split(",") if s.strip()]
    elif args.label_preset == "blog":
        label_models = [
            "claude_opus_4_5_inspect",
            "claude_3_5_sonnet_20241022_inspect",
            "o3_inspect",
        ]
    else:
        label_models = []

    runs_df = pd.read_pickle(args.runs)
    t_grid = time_grid_from_runs(runs_df, args.grid_points)  # hours
    x_grid = np.log(t_grid) - mean_log_t_hours

    # Extract scalar draws
    alpha = draws_df["alpha"].to_numpy(dtype=float)
    kappa = draws_df["kappa"].to_numpy(dtype=float)
    sigma_b = draws_df["sigma_b"].to_numpy(dtype=float)
    mu_loga = draws_df["mu_loga"].to_numpy(dtype=float)
    sigma_loga = draws_df["sigma_loga"].to_numpy(dtype=float)

    # Extract theta draws for selected models
    theta_cols = [f"theta.{i+1}" for i in range(len(model_names))]
    theta_all = draws_df[theta_cols].to_numpy(dtype=float)  # (S, I)

    rng = np.random.default_rng(args.seed)

    # p_i(t) curves for selected models
    curves = []
    S = len(draws_df)
    M = int(args.mc)
    u = rng.normal(0.0, sigma_b[:, None], size=(S, M))

    for model in models:
        i = model_index[model]
        th = theta_all[:, i]  # (S,)

        # Typical curve: u=0 and loga=mu
        a_typ = np.exp(mu_loga[:, None])  # (S, 1)
        b_mean = alpha[:, None] + kappa[:, None] * x_grid[None, :]  # (S, G)
        p_typ = logistic(a_typ * (th[:, None] - b_mean))  # (S, G)

        # Marginal curve: integrate u and loga heterogeneity via MC
        G = len(t_grid)
        p_marg = np.empty((S, G), dtype=float)
        chunk = 60
        for start in range(0, G, chunk):
            end = min(G, start + chunk)
            x_c = x_grid[start:end]
            b_mean_c = alpha[:, None] + kappa[:, None] * x_c[None, :]  # (S, B)
            b_mc = b_mean_c[:, None, :] + u[:, :, None]  # (S, M, B)

            loga = mu_loga[:, None, None] + sigma_loga[:, None, None] * rng.normal(size=(S, M, len(x_c)))
            loga = np.clip(loga, -2.0, 2.0)  # match Stan bounds
            a_mc = np.exp(loga)
            eta = a_mc * (th[:, None, None] - b_mc)
            p_marg[:, start:end] = logistic(eta).mean(axis=1)

        for curve_name, p in [("typical", p_typ), ("marginal", p_marg)]:
            mean = p.mean(axis=0)
            q10, q50, q90 = np.quantile(p, [0.10, 0.50, 0.90], axis=0)
            curves.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "curve": curve_name,
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

    # Release-date trend in theta-space
    release = load_release_dates(args.benchmark_yaml)
    rows = [(i, m, release[m]) for i, m in enumerate(model_names) if m in release and m != "human"]
    if len(rows) < 3:
        raise SystemExit("Need at least 3 models with release dates to fit a trend.")
    idx = np.array([r[0] for r in rows], dtype=int)
    names = [r[1] for r in rows]
    dates = pd.to_datetime([r[2] for r in rows])

    date0 = pd.to_datetime(pd.Series(dates).astype("int64").mean()).normalize()
    x = ((dates - date0).days.to_numpy(dtype=float)) / 365.25  # years since mean release

    theta_sub = theta_all[:, idx]  # (S, N)
    theta_trend_mode = str(meta.get("theta_trend", "none"))

    x_min = float(np.min(x))
    if args.extend_to_year is not None and args.extend_to_year > 0:
        end_date = pd.Timestamp(year=int(args.extend_to_year), month=12, day=31)
        x_max_target = float(((end_date.normalize() - date0).days) / 365.25)
        x_max = max(float(np.max(x)), x_max_target)
    else:
        x_max = float(np.max(x))

    x_grid2 = np.linspace(x_min, x_max, 160)
    date_grid = (date0 + pd.to_timedelta(x_grid2 * 365.25, unit="D")).normalize()

    if theta_trend_mode == "none":
        # Per-draw OLS: theta = a + b*x
        x0 = float(np.mean(x))
        xc = x - x0
        denom = float(np.sum(xc * xc))
        if denom <= 0:
            raise SystemExit("Release dates have zero variance; can't fit trend.")
        b_draws = (theta_sub @ xc) / denom  # (S,)
        a_draws = theta_sub.mean(axis=1) - b_draws * x0  # (S,)
        theta_grid = a_draws[:, None] + b_draws[:, None] * x_grid2[None, :]  # (S, G)
    elif theta_trend_mode in {"linear", "quadratic", "quadratic_pos", "sqrt", "log1p"}:
        K = int(meta.get("theta_trend_K", 0))
        if theta_trend_mode in {"linear", "sqrt", "log1p"} and K != 1:
            raise SystemExit(f"Expected theta_trend_K=1 for linear, got {K}")
        if theta_trend_mode in {"quadratic", "quadratic_pos"} and K != 2:
            raise SystemExit(f"Expected theta_trend_K=2 for quadratic, got {K}")
        gamma0 = draws_df["gamma0"].to_numpy(dtype=float)
        gamma = np.column_stack([draws_df[f"gamma.{k}"].to_numpy(dtype=float) for k in range(1, K + 1)])  # (S, K)
        if K == 1:
            if theta_trend_mode == "sqrt":
                x_min = float(meta.get("theta_trend_x_min", float("nan")))
                eps = float(meta.get("theta_trend_eps", 1e-6))
                if not np.isfinite(x_min):
                    raise SystemExit("meta.json missing theta_trend_x_min for sqrt trend")
                z = np.sqrt(np.clip(x_grid2 - x_min + eps, a_min=eps, a_max=None))
                Xg = z[None, :]
            elif theta_trend_mode == "log1p":
                x_min = float(meta.get("theta_trend_x_min", float("nan")))
                scale = float(meta.get("theta_trend_scale", 1.0))
                if not np.isfinite(x_min):
                    raise SystemExit("meta.json missing theta_trend_x_min for log1p trend")
                z = np.clip(x_grid2 - x_min, a_min=0.0, a_max=None) / max(scale, 1e-9)
                Xg = np.log1p(z)[None, :]
            else:
                Xg = x_grid2[None, :]  # (1, G)
        else:
            Xg = np.vstack([x_grid2, x_grid2 * x_grid2])  # (2, G)
        theta_grid = gamma0[:, None] + gamma @ Xg  # (S, G)
        # For downstream doubling-time calculations, define an "effective" b_draws(x) if needed.
        b_draws = None
    elif theta_trend_mode == "xpow":
        gamma0 = draws_df["gamma0"].to_numpy(dtype=float)
        gamma1 = draws_df["gamma1"].to_numpy(dtype=float)
        a_pow = draws_df["a_pow"].to_numpy(dtype=float)
        x_min = float(meta.get("theta_trend_x_min", float("nan")))
        x_scale = float(meta.get("theta_trend_x_scale", float("nan")))
        x_eps = float(meta.get("theta_trend_x_eps", 1e-6))
        if not np.isfinite(x_min) or not np.isfinite(x_scale):
            raise SystemExit("meta.json missing theta_trend_x_min/x_scale for xpow trend")
        z = np.maximum(x_grid2 - x_min + x_eps, x_eps) / max(x_scale, 1e-9)
        feat = z[None, :] ** a_pow[:, None]
        theta_grid = gamma0[:, None] + gamma1[:, None] * feat
        b_draws = None
    elif theta_trend_mode == "t50_logistic":
        log_t_low = draws_df["log_t_low"].to_numpy(dtype=float)
        log_delta_t = draws_df["log_delta_t"].to_numpy(dtype=float)
        a_t = draws_df["a_t"].to_numpy(dtype=float)
        b_t = draws_df["b_t"].to_numpy(dtype=float)
        t_low = np.exp(log_t_low)
        t_high = t_low + np.exp(log_delta_t)
        s_curve = logistic(a_t[:, None] + b_t[:, None] * x_grid2[None, :])
        t50 = t_low[:, None] + (t_high[:, None] - t_low[:, None]) * s_curve
        theta_grid = alpha[:, None] + kappa[:, None] * (np.log(t50) - mean_log_t_hours)
        b_draws = None
    elif theta_trend_mode == "theta_logistic":
        theta_min = draws_df["theta_min"].to_numpy(dtype=float)
        theta_range = draws_df["theta_range"].to_numpy(dtype=float)
        a_logis = draws_df["a_logis"].to_numpy(dtype=float)
        b_logis = draws_df["b_logis"].to_numpy(dtype=float)
        s_curve = logistic(a_logis[:, None] + b_logis[:, None] * x_grid2[None, :])
        theta_grid = theta_min[:, None] + theta_range[:, None] * s_curve
        b_draws = None
    elif theta_trend_mode == "singularity":
        x_date_max = float(meta.get("x_date_max", float("nan")))
        if not np.isfinite(x_date_max):
            raise SystemExit("meta.json missing x_date_max for singularity trend")
        gamma0 = draws_df["gamma0"].to_numpy(dtype=float)
        gamma1 = draws_df["gamma1"].to_numpy(dtype=float)
        c_sing = draws_df["c_sing"].to_numpy(dtype=float)
        alpha_sing = draws_df["alpha_sing"].to_numpy(dtype=float)
        eta_tstar = draws_df["eta_tstar"].to_numpy(dtype=float)
        t_star = x_date_max + np.exp(eta_tstar) + 0.25
        denom = t_star[:, None] - x_grid2[None, :]
        # Avoid invalid pow for denom<=0: mask those points as NaN so plots break where singularity occurs.
        denom_safe = np.where(denom > 1e-6, denom, np.nan)
        theta_grid = gamma0[:, None] + gamma1[:, None] * x_grid2[None, :] + c_sing[:, None] / (denom_safe ** alpha_sing[:, None])
        b_draws = None
    elif theta_trend_mode == "singularity_nolinear":
        x_date_max = float(meta.get("x_date_max", float("nan")))
        if not np.isfinite(x_date_max):
            raise SystemExit("meta.json missing x_date_max for singularity trend")
        gamma0 = draws_df["gamma0"].to_numpy(dtype=float)
        c_sing = draws_df["c_sing"].to_numpy(dtype=float)
        alpha_sing = draws_df["alpha_sing"].to_numpy(dtype=float)
        eta_tstar = draws_df["eta_tstar"].to_numpy(dtype=float)
        t_star = x_date_max + np.exp(eta_tstar) + 0.25
        denom = t_star[:, None] - x_grid2[None, :]
        denom_safe = np.where(denom > 1e-6, denom, np.nan)
        theta_grid = gamma0[:, None] + c_sing[:, None] / (denom_safe ** alpha_sing[:, None])
        b_draws = None
    else:
        raise SystemExit(f"Unknown theta_trend in meta.json: {theta_trend_mode!r}")

    theta_obs_q10, theta_obs_q50, theta_obs_q90 = summarize_draws(theta_sub)
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

    # Horizons: (a) at each observed model date, and (b) along the trend grid
    horizon_grid_frames = []
    horizon_points_frames = []
    horizon_grid_typ_frames = []
    horizon_points_typ_frames = []
    dt_rows = []

    # Doubling time (still written, but not central to the blog post).
    # We define local doubling time as log(2) / (d/dx log t50(x)), where x is years since mean dated release.
    # Since log t50 is affine in theta, this is log(2) / ( (d/dx theta(x)) / kappa ).
    dt_draws = None

    # Default: estimate slope at x≈0 by finite differences on theta_grid (works for any theta_trend_mode).
    idx0 = int(np.argmin(np.abs(x_grid2)))
    if idx0 <= 0:
        idx_a, idx_b = 0, 1
    elif idx0 >= len(x_grid2) - 1:
        idx_a, idx_b = len(x_grid2) - 2, len(x_grid2) - 1
    else:
        idx_a, idx_b = idx0 - 1, idx0 + 1
    dx = float(x_grid2[idx_b] - x_grid2[idx_a])
    if dx <= 0:
        dx = 1.0
    slope_theta0 = (theta_grid[:, idx_b] - theta_grid[:, idx_a]) / dx  # (S,)
    slope_logt0 = slope_theta0 / kappa
    ok = slope_logt0 > 0
    dt_draws = np.full_like(slope_logt0, np.nan)
    dt_draws[ok] = math.log(2.0) / slope_logt0[ok]
    pd.DataFrame({"slope_log_horizon_per_year": slope_logt0, "doubling_time_years": dt_draws}).to_csv(
        args.outdir / "doubling_time_draws.csv", index=False
    )

    # For quadratic trends, also provide a date-dependent curve (useful as a diagnostic).
    if theta_trend_mode in {"quadratic", "quadratic_pos"}:
        gamma1 = draws_df["gamma.1"].to_numpy(dtype=float)
        gamma2 = draws_df["gamma.2"].to_numpy(dtype=float)
        slope_theta_grid = gamma1[:, None] + 2.0 * gamma2[:, None] * x_grid2[None, :]  # (S, G)
        slope_logt_grid = slope_theta_grid / kappa[:, None]
        dt_grid = np.full_like(slope_logt_grid, np.nan)
        okg = slope_logt_grid > 0
        dt_grid[okg] = math.log(2.0) / slope_logt_grid[okg]
        q10, q50, q90 = summarize_draws(dt_grid)
        pd.DataFrame(
            {"date": date_grid, "x_years_since_mean_release": x_grid2, "dt_years_q10": q10, "dt_years_q50": q50, "dt_years_q90": q90}
        ).to_csv(args.outdir / "doubling_time_curve.csv", index=False)

    for p in ps:
        t_grid2_marg = horizon_hours_approx(
            theta_grid,
            alpha[:, None],
            kappa[:, None],
            sigma_b[:, None],
            mu_loga[:, None],
            mean_log_t_hours=mean_log_t_hours,
            p=p,
        )
        q10, q50, q90 = summarize_draws(t_grid2_marg)
        horizon_grid_frames.append(
            pd.DataFrame(
                {
                    "p": p,
                    "x_years_since_mean_release": x_grid2,
                    "date": date_grid,
                    "t_hours_q10": q10,
                    "t_hours_q50": q50,
                    "t_hours_q90": q90,
                }
            )
        )

        t_points_marg = horizon_hours_approx(
            theta_sub,
            alpha[:, None],
            kappa[:, None],
            sigma_b[:, None],
            mu_loga[:, None],
            mean_log_t_hours=mean_log_t_hours,
            p=p,
        )
        q10p, q50p, q90p = summarize_draws(t_points_marg)
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

        # Typical horizons (no task random effects integration)
        t_grid2_typ = horizon_hours_typical(
            theta_grid,
            alpha[:, None],
            kappa[:, None],
            mu_loga[:, None],
            mean_log_t_hours=mean_log_t_hours,
            p=p,
        )
        q10t, q50t, q90t = summarize_draws(t_grid2_typ)
        horizon_grid_typ_frames.append(
            pd.DataFrame(
                {
                    "p": p,
                    "x_years_since_mean_release": x_grid2,
                    "date": date_grid,
                    "t_hours_q10": q10t,
                    "t_hours_q50": q50t,
                    "t_hours_q90": q90t,
                }
            )
        )

        t_points_typ = horizon_hours_typical(
            theta_sub,
            alpha[:, None],
            kappa[:, None],
            mu_loga[:, None],
            mean_log_t_hours=mean_log_t_hours,
            p=p,
        )
        q10pt, q50pt, q90pt = summarize_draws(t_points_typ)
        horizon_points_typ_frames.append(
            pd.DataFrame(
                {
                    "model": names,
                    "release_date": dates,
                    "p": p,
                    "t_hours_q10": q10pt,
                    "t_hours_q50": q50pt,
                    "t_hours_q90": q90pt,
                }
            )
        )

        dt_rows.append(
            {
                "p": p,
                "doubling_time_years_q10": float(np.nanquantile(dt_draws, 0.10)),
                "doubling_time_years_q50": float(np.nanquantile(dt_draws, 0.50)),
                "doubling_time_years_q90": float(np.nanquantile(dt_draws, 0.90)),
                "note": ("quadratic: reported at mean release date" if theta_trend_mode == "quadratic" else ""),
            }
        )

    horizon_grid_df = pd.concat(horizon_grid_frames, ignore_index=True)
    horizon_points_df = pd.concat(horizon_points_frames, ignore_index=True)
    horizon_grid_typ_df = pd.concat(horizon_grid_typ_frames, ignore_index=True)
    horizon_points_typ_df = pd.concat(horizon_points_typ_frames, ignore_index=True)
    horizon_grid_df.to_csv(args.outdir / "horizon_grid.csv", index=False)
    horizon_points_df.to_csv(args.outdir / "horizon_points.csv", index=False)
    horizon_grid_typ_df.to_csv(args.outdir / "horizon_grid_typical.csv", index=False)
    horizon_points_typ_df.to_csv(args.outdir / "horizon_points_typical.csv", index=False)
    pd.DataFrame(dt_rows).to_csv(args.outdir / "doubling_time.csv", index=False)

    # Plotting
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates

    # p_curves
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
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(model, loc="left", fontsize=10)
        ax.set_ylabel("p(success)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Task length t (hours, log scale)")
    fig.suptitle("Posterior p_i(t) under the time-IRT model", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.outdir / "p_curves.png", dpi=200)
    plt.close(fig)

    # theta vs date
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.0))
    ax.errorbar(
        theta_points["release_date"],
        theta_points["theta_q50"],
        yerr=np.vstack([theta_points["theta_q50"] - theta_points["theta_q10"], theta_points["theta_q90"] - theta_points["theta_q50"]]),
        fmt="o",
        color="black",
        ecolor="gray",
        elinewidth=1,
        capsize=2,
        alpha=0.85,
        label="model θ (median; 10–90%)",
    )
    theta_grid_q10, theta_grid_q50, theta_grid_q90 = summarize_draws(theta_grid)
    pd.DataFrame({
        "date": date_grid,
        "theta_q10": theta_grid_q10,
        "theta_q50": theta_grid_q50,
        "theta_q90": theta_grid_q90,
    }).to_csv(args.outdir / "theta_trend_grid.csv", index=False)
    ax.plot(date_grid, theta_grid_q50, color="C0", linewidth=2, label="θ trend (date → θ)")
    ax.fill_between(date_grid, theta_grid_q10, theta_grid_q90, color="C0", alpha=0.15)
    ax.set_xlabel("Release date")
    ax.set_ylabel("Ability θ (anchored)")
    ax.set_title("Ability vs release date")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "theta_vs_date.png", dpi=200)
    plt.close(fig)

    # horizons per p + combined
    colors = [f"C{i}" for i in range(10)]
    for k, p in enumerate(sorted(ps)):
        g = horizon_grid_df[horizon_grid_df["p"] == p].sort_values("date")
        pts = horizon_points_df[horizon_points_df["p"] == p].sort_values("release_date")
        g_typ = horizon_grid_typ_df[horizon_grid_typ_df["p"] == p].sort_values("date")
        pts_typ = horizon_points_typ_df[horizon_points_typ_df["p"] == p].sort_values("release_date")

        fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2))
        ax.plot(g["date"], g["t_hours_q50"], color="C1", linewidth=2, label="trend-based horizon")
        ax.fill_between(g["date"], g["t_hours_q10"], g["t_hours_q90"], color="C1", alpha=0.15)
        ax.scatter(pts["release_date"], pts["t_hours_q50"], s=35, color="black", alpha=0.8, label="model horizon (median)")
        for _, r in pts.iterrows():
            ax.plot([r["release_date"], r["release_date"]], [r["t_hours_q10"], r["t_hours_q90"]], color="black", alpha=0.20)
        annotate_models(ax, pts, label_models)
        ax.set_yscale("log")
        ax.set_xlabel("Release date")
        ax.set_ylabel(f"Task duration (humans) at t{int(round(100*p))}")
        ax.set_title(f"Horizon vs release date (p={p:g})")
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        apply_metr_duration_ticks(ax, g["t_hours_q50"].to_numpy())
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.outdir / f"horizon_p{int(round(100*p)):02d}.png", dpi=200)
        plt.close(fig)

        # Additional plot: typical vs marginal on the same axes.
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2))
        ax.plot(g_typ["date"], g_typ["t_hours_q50"], color="C0", linewidth=2, label="typical (no task RE)")
        ax.fill_between(g_typ["date"], g_typ["t_hours_q10"], g_typ["t_hours_q90"], color="C0", alpha=0.12)
        ax.plot(g["date"], g["t_hours_q50"], color="C1", linewidth=2, label="marginal (integrate task RE)")
        ax.fill_between(g["date"], g["t_hours_q10"], g["t_hours_q90"], color="C1", alpha=0.12)

        ax.scatter(pts_typ["release_date"], pts_typ["t_hours_q50"], s=22, color="C0", alpha=0.55)
        ax.scatter(pts["release_date"], pts["t_hours_q50"], s=22, color="C1", alpha=0.55)

        ax.set_yscale("log")
        ax.set_xlabel("Release date")
        ax.set_ylabel(f"Task duration (humans) at t{int(round(100*p))}")
        ax.set_title(f"Horizon vs release date (p={p:g}): typical vs marginal")
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        apply_metr_duration_ticks(ax, np.concatenate([g_typ["t_hours_q50"].to_numpy(), g["t_hours_q50"].to_numpy()]))
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.outdir / f"horizon_p{int(round(100*p)):02d}_typical_vs_marginal.png", dpi=200)
        plt.close(fig)

    if len(ps) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.4))
        for k, p in enumerate(sorted(ps)):
            g = horizon_grid_df[horizon_grid_df["p"] == p].sort_values("date")
            pts = horizon_points_df[horizon_points_df["p"] == p].sort_values("release_date")
            color = colors[k % len(colors)]
            ax.plot(g["date"], g["t_hours_q50"], color=color, linewidth=2, label=f"t{int(round(100*p))} trend (median; 10–90%)")
            ax.fill_between(g["date"], g["t_hours_q10"], g["t_hours_q90"], color=color, alpha=0.12)
            ax.scatter(pts["release_date"], pts["t_hours_q50"], s=18, color=color, alpha=0.55)
        ax.set_yscale("log")
        ax.set_xlabel("Release date")
        ax.set_ylabel("Task duration (humans)")
        ax.set_title("Horizon vs release date (multiple p levels)")
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        apply_metr_duration_ticks(ax, horizon_grid_df["t_hours_q50"].to_numpy())
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.outdir / "horizons_multi.png", dpi=200)
        plt.close(fig)

        # Additional plot: typical vs marginal, multi-p, as two panels.
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
        for ax, kind, grid_df in [
            (axes[0], "typical (no task RE)", horizon_grid_typ_df),
            (axes[1], "marginal (integrate task RE)", horizon_grid_df),
        ]:
            for k, p in enumerate(sorted(ps)):
                gk = grid_df[grid_df["p"] == p].sort_values("date")
                color = colors[k % len(colors)]
                ax.plot(gk["date"], gk["t_hours_q50"], color=color, linewidth=2, label=f"t{int(round(100*p))}")
                ax.fill_between(gk["date"], gk["t_hours_q10"], gk["t_hours_q90"], color=color, alpha=0.12)
            ax.set_yscale("log")
            ax.set_xlabel("Release date")
            ax.set_title(kind)
            ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.legend(loc="best", fontsize=8)
        axes[0].set_ylabel("Task duration (humans)")
        fig.suptitle("Horizon vs release date: typical vs marginal", y=0.98)
        apply_metr_duration_ticks(axes[0], np.concatenate([horizon_grid_typ_df["t_hours_q50"].to_numpy(), horizon_grid_df["t_hours_q50"].to_numpy()]))
        fig.tight_layout()
        fig.savefig(args.outdir / "horizons_multi_typical_vs_marginal.png", dpi=200)
        plt.close(fig)

    # doubling time posterior
    vals = dt_draws[np.isfinite(dt_draws)]
    vals_months = 12.0 * vals
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))
    q10, q50, q90 = np.quantile(vals_months, [0.10, 0.50, 0.90])
    lo, hi = np.quantile(vals_months, [0.001, 0.999])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.min(vals_months)), float(np.max(vals_months))
    xg = np.linspace(lo, hi, 350)
    dens = gaussian_kde_1d(vals_months, xg)
    ax.plot(xg, dens, color="C2", linewidth=2.2, label="KDE (Silverman bw)")
    ax.fill_between(xg, 0.0, dens, color="C2", alpha=0.12)

    ax.axvline(q50, color="black", linewidth=2, label=f"median (50th pct) = {q50:.3g} months")
    ax.axvline(q10, color="black", linewidth=1.2, linestyle="--", label="10th percentile (dashed)")
    ax.axvline(q90, color="black", linewidth=1.2, linestyle="--", label="90th percentile (dashed)")
    ax.set_xlabel("Doubling time (months)")
    ax.set_ylabel("Posterior density")
    ax.set_title("Implied horizon doubling time")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "doubling_time_posterior.png", dpi=200)
    plt.close(fig)

    if theta_trend_mode in {"quadratic", "quadratic_pos"} and (args.outdir / "doubling_time_curve.csv").exists():
        dt_curve = pd.read_csv(args.outdir / "doubling_time_curve.csv")
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.8))
        ax.plot(pd.to_datetime(dt_curve["date"]), dt_curve["dt_years_q50"], color="C2", linewidth=2, label="median")
        ax.fill_between(pd.to_datetime(dt_curve["date"]), dt_curve["dt_years_q10"], dt_curve["dt_years_q90"], color="C2", alpha=0.15)
        ax.set_xlabel("Release date")
        ax.set_ylabel("Local doubling time (years)")
        ax.set_title("Implied doubling time vs date (quadratic θ trend)")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.outdir / "doubling_time_vs_date.png", dpi=200)
        plt.close(fig)

    out_meta = {
        "spec": spec,
        "run_id": run_id,
        "fitdir": str(args.fitdir),
        "outdir": str(args.outdir),
        "models": models,
        "p": ps,
        "draws_used": int(len(draws_df)),
        "mc_per_draw": int(args.mc),
        "grid_points": int(args.grid_points),
        "theta_trend": theta_trend_mode,
    }
    (args.outdir / "meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    print(f"wrote: {args.outdir / 'p_curves.png'}")
    print(f"wrote: {args.outdir / 'theta_vs_date.png'}")
    print(f"wrote: {args.outdir / 'horizons_multi.png'}")


if __name__ == "__main__":
    main()
