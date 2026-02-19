#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Diagnostics for a fitted time-IRT run: LOO (from log_lik), residual plots, and "
            "non-cheating calibration/Brier using marginal predictions that integrate over task random effects."
        )
    )
    p.add_argument("--fitdir", type=Path, default=None, help="Fit directory containing meta.json and CmdStan CSVs.")
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--spec", type=str, default="time_irt__theta_linear")
    p.add_argument("--run-id", type=str, default=None, help="Defaults to --runs-root/<spec>/LATEST.")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (default: run diagnostics dir).")

    p.add_argument("--counts", type=Path, default=Path("data/irt_counts_task_id.csv"), help="Aggregated counts table.")
    p.add_argument("--runs", type=Path, default=Path("data/runs.pkl"), help="Runs pickle for task lengths.")

    p.add_argument("--draws", type=int, default=600, help="Posterior draws to use (subsampled) (default: 600).")
    p.add_argument("--mc", type=int, default=120, help="MC samples per draw for marginal predictions (default: 120).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed (default: 123).")
    p.add_argument("--bins", type=int, default=10, help="Calibration bins (default: 10).")
    p.add_argument(
        "--time-bins",
        type=int,
        default=12,
        help="Time-bins for pooled PPC by task length (log-time quantile bins) (default: 12).",
    )
    return p.parse_args()


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def chain_csv_files(fitdir: Path) -> list[Path]:
    files = sorted(fitdir.glob("*.csv"))
    keep: list[Path] = []
    for f in files:
        if f.name in {"summary.csv", "theta.csv", "b.csv", "a.csv"}:
            continue
        keep.append(f)
    return keep


def read_draws_subset(fitdir: Path, draws: int, seed: int) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    meta = json.loads((fitdir / "meta.json").read_text(encoding="utf-8"))
    model_names = pd.read_csv(fitdir / "theta.csv")["model"].astype(str).tolist()
    I = len(model_names)
    theta_cols = [f"theta.{i}" for i in range(1, I + 1)]
    scalar_cols = ["alpha", "kappa", "sigma_b", "mu_loga", "sigma_loga"]

    # log_lik is needed for LOO; include all entries.
    ll_cols = [f"log_lik.{k}" for k in range(1, int(meta.get("N", 0)) + 1)]
    # meta.json from fit_time_model doesn't currently store N; fall back by scanning one chain file.
    if not ll_cols:
        cf = chain_csv_files(fitdir)
        if not cf:
            raise SystemExit(f"No chain CSVs found in {fitdir}")
        header = pd.read_csv(cf[0], comment="#", nrows=0).columns.tolist()
        ll_cols = [c for c in header if c.startswith("log_lik.")]

    extra_cols: list[str] = []
    trend = str(meta.get("theta_trend", "none"))
    if trend in {"linear", "quadratic_pos"}:
        K = int(meta.get("theta_trend_K", 0))
        extra_cols = ["gamma0", "sigma_theta"] + [f"gamma.{k}" for k in range(1, K + 1)]
    elif trend == "xpow":
        extra_cols = ["gamma0", "gamma1", "a_pow", "sigma_theta"]

    usecols = scalar_cols + theta_cols + extra_cols + ll_cols
    chain_files = chain_csv_files(fitdir)
    if not chain_files:
        raise SystemExit(f"No chain CSVs found in {fitdir}")

    # Read and subsample one chain at a time to limit peak memory.
    rng = np.random.default_rng(seed)
    n_chains = len(chain_files)
    per_chain = max(1, draws // n_chains)
    sampled: list[pd.DataFrame] = []
    for f in chain_files:
        # First pass: identify comment and data line numbers.
        comment_lines: list[int] = []
        data_lines: list[int] = []
        with open(f) as fh:
            for i, line in enumerate(fh):
                if line.startswith("#"):
                    comment_lines.append(i)
                else:
                    data_lines.append(i)
        # data_lines[0] = header, data_lines[1:] = draws
        n_data = len(data_lines) - 1
        k = min(per_chain, n_data)
        keep_idx = set(rng.choice(n_data, size=k, replace=False).tolist())
        keep_lines = {data_lines[0]}  # always keep header
        keep_lines |= {data_lines[1 + i] for i in keep_idx}
        df = pd.read_csv(f, usecols=usecols, skiprows=lambda i: i not in keep_lines)
        sampled.append(df.reset_index(drop=True))
        del df
    all_draws = pd.concat(sampled, ignore_index=True)
    del sampled

    return all_draws, meta, model_names


def build_task_x(runs_df: pd.DataFrame, task_ids: np.ndarray, mean_log_t_hours: float) -> np.ndarray:
    task_time = runs_df.groupby("task_id", as_index=True)["human_minutes"].first()
    t_minutes = task_time.reindex(task_ids).to_numpy(dtype=float)
    if np.any(t_minutes <= 0) or np.any(~np.isfinite(t_minutes)):
        raise SystemExit("Non-positive or invalid task lengths encountered.")
    t_hours = t_minutes / 60.0
    return np.log(t_hours) - mean_log_t_hours


def task_hours(runs_df: pd.DataFrame, task_ids: np.ndarray) -> np.ndarray:
    task_time = runs_df.groupby("task_id", as_index=True)["human_minutes"].first()
    t_minutes = task_time.reindex(task_ids).to_numpy(dtype=float)
    if np.any(t_minutes <= 0) or np.any(~np.isfinite(t_minutes)):
        raise SystemExit("Non-positive or invalid task lengths encountered.")
    return t_minutes / 60.0


def marginal_pred_binomial(
    *,
    theta: np.ndarray,  # (S, Nobs)
    x: np.ndarray,  # (Nobs,)
    alpha: np.ndarray,  # (S,)
    kappa: np.ndarray,  # (S,)
    sigma_b: np.ndarray,  # (S,)
    mu_loga: np.ndarray,  # (S,)
    sigma_loga: np.ndarray,  # (S,)
    mc: int,
    rng: np.random.Generator,
    chunk_size: int = 512,
) -> np.ndarray:
    """
    Non-cheating prediction: integrate over task random effects, do NOT condition on b_j / a_j.

      b | x ~ Normal(alpha + kappa*x, sigma_b)
      log_a ~ Normal(mu_loga, sigma_loga), truncated to [-2,2] by clipping draws.

    Returns p_hat for each observation (Nobs,), averaged over posterior draws and MC.
    Processes observations in chunks to limit peak memory.
    """
    S, Nobs = theta.shape
    if x.shape != (Nobs,):
        raise ValueError("x must have shape (Nobs,)")

    # u is shared across observations within each (draw, mc) pair.
    u = rng.normal(0.0, sigma_b[:, None], size=(S, mc))  # (S, M)
    b_mean = alpha[:, None] + kappa[:, None] * x[None, :]  # (S, Nobs)

    p_hat = np.empty(Nobs, dtype=float)
    for start in range(0, Nobs, chunk_size):
        end = min(start + chunk_size, Nobs)
        C = end - start

        # Fresh loga per (draw, mc, obs) from the population distribution.
        loga = mu_loga[:, None, None] + sigma_loga[:, None, None] * rng.normal(size=(S, mc, C))
        np.clip(loga, -2.0, 2.0, out=loga)
        a = np.exp(loga)

        b = b_mean[:, None, start:end] + u[:, :, None]  # (S, M, C)
        eta = a * (theta[:, None, start:end] - b)  # (S, M, C)
        p_hat[start:end] = logistic(eta).mean(axis=1).mean(axis=0)

        del loga, a, b, eta

    return p_hat


def weighted_brier_and_logscore(s: np.ndarray, n: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = s / n
    w = n / np.sum(n)
    brier = float(np.sum(w * (y - p) ** 2))
    logscore = float(np.sum((s * np.log(p) + (n - s) * np.log(1.0 - p)) / np.sum(n)))
    return {"brier_weighted": brier, "logscore_per_trial": logscore}


def calibration_bins(s: np.ndarray, n: np.ndarray, p: np.ndarray, bins: int) -> pd.DataFrame:
    df = pd.DataFrame({"s": s, "n": n, "p": p})
    df["bin"] = pd.qcut(df["p"], q=bins, labels=False, duplicates="drop")
    out = (
        df.groupby("bin", as_index=False)
        .agg(n=("n", "sum"), s=("s", "sum"), p_mean=("p", "mean"))
        .assign(obs_rate=lambda d: d["s"] / d["n"])
    )
    # Binomial SE for observed rate (for rough error bars).
    out["se"] = np.sqrt(out["obs_rate"] * (1 - out["obs_rate"]) / out["n"])
    return out


def time_binned_ppc(t_hours: np.ndarray, s: np.ndarray, n: np.ndarray, p: np.ndarray, bins: int) -> pd.DataFrame:
    df = pd.DataFrame({"t_hours": t_hours, "s": s, "n": n, "p": p})
    if np.any(df["t_hours"] <= 0) or np.any(~np.isfinite(df["t_hours"])):
        raise ValueError("t_hours must be positive and finite.")
    df["logt"] = np.log(df["t_hours"])
    df["bin"] = pd.qcut(df["logt"], q=bins, labels=False, duplicates="drop")
    df["npred"] = df["n"] * df["p"]
    df["wlogt"] = df["n"] * df["logt"]

    out = (
        df.groupby("bin", as_index=False)
        .agg(
            t_min_hours=("t_hours", "min"),
            t_max_hours=("t_hours", "max"),
            n=("n", "sum"),
            s=("s", "sum"),
            npred=("npred", "sum"),
            wlogt=("wlogt", "sum"),
            n_cells=("t_hours", "size"),
        )
        .assign(
            pred_rate=lambda d: d["npred"] / d["n"].clip(lower=1.0),
            obs_rate=lambda d: d["s"] / d["n"].clip(lower=1.0),
            obs_se=lambda d: np.sqrt(d["obs_rate"] * (1.0 - d["obs_rate"]) / d["n"].clip(lower=1.0)),
            t_geo_hours=lambda d: np.exp(d["wlogt"] / d["n"].clip(lower=1.0)),
        )
        .drop(columns=["npred", "wlogt"])
    )
    return out.sort_values("t_geo_hours").reset_index(drop=True)


def compute_loo(log_lik: np.ndarray, outdir: Path) -> pd.DataFrame:
    """
    PSIS-LOO via arviz. Writes loo.csv and pareto_k.csv.
    """
    # Keep arviz cache local to avoid permission issues in some environments.
    os.environ["XDG_CACHE_HOME"] = str(outdir / ".cache")
    try:
        import arviz as az
    except Exception as e:
        raise SystemExit(f"arviz import failed ({e}); can't compute LOO.")

    outdir.mkdir(parents=True, exist_ok=True)

    # ArviZ expects (chain, draw, obs). We have a flattened draw dimension; treat it as a single chain.
    log_lik_c = log_lik[None, :, :]  # (1, S, Nobs)
    posterior_dummy = {"_": np.zeros((1, log_lik.shape[0], 1), dtype=float)}
    idata = az.from_dict(posterior=posterior_dummy, log_likelihood={"y": log_lik_c})
    loo = az.loo(idata, pointwise=True)
    loo_df = pd.DataFrame(
        {
            "elpd_loo": [float(loo.elpd_loo)],
            "se": [float(loo.se)],
            "p_loo": [float(loo.p_loo)],
            "n_obs": [int(log_lik.shape[1])],
        }
    )
    loo_df.to_csv(outdir / "loo.csv", index=False)

    pareto_k = np.asarray(loo.pareto_k)
    pk = pd.DataFrame({"pareto_k": pareto_k})
    pk.to_csv(outdir / "pareto_k.csv", index=False)
    return loo_df


def main() -> None:
    args = parse_args()

    if args.fitdir is None:
        run_id = args.run_id or (args.runs_root / args.spec / "LATEST").read_text(encoding="utf-8").strip()
        args.fitdir = args.runs_root / args.spec / run_id / "fit"
    else:
        run_id = args.fitdir.parent.name if args.fitdir.name == "fit" else args.fitdir.name

    if args.outdir is None:
        args.outdir = args.fitdir.parent / "diagnostics"
    args.outdir.mkdir(parents=True, exist_ok=True)

    draws_df, meta, model_names = read_draws_subset(args.fitdir, args.draws, args.seed)
    spec = str(meta.get("spec") or args.spec)

    # LOO from log_lik
    ll_cols = [c for c in draws_df.columns if c.startswith("log_lik.")]
    log_lik = draws_df[ll_cols].to_numpy(dtype=float)  # (S, Ncells)
    loo_df = compute_loo(log_lik, args.outdir)

    # Residual plot inputs (uses b_j posterior mean from fit; diagnostic only).
    b_df = pd.read_csv(args.fitdir / "b.csv")
    mean_log_t_hours = float(meta.get("mean_log_t_hours", float("nan")))
    if not np.isfinite(mean_log_t_hours):
        raise SystemExit("meta.json missing mean_log_t_hours")
    runs_df = pd.read_pickle(args.runs)
    x_task = build_task_x(runs_df, b_df["task_id"].to_numpy(), mean_log_t_hours)
    b_mean = b_df["mean"].to_numpy(dtype=float)
    alpha_mean = float(meta["scalar_params"]["alpha"]["mean"])
    kappa_mean = float(meta["scalar_params"]["kappa"]["mean"])
    u_hat = b_mean - (alpha_mean + kappa_mean * x_task)
    resid_df = pd.DataFrame({"task_id": b_df["task_id"], "x": x_task, "u_hat": u_hat})
    resid_df.to_csv(args.outdir / "difficulty_residuals.csv", index=False)

    # Calibration / Brier using marginal predictions that do not condition on task residuals.
    counts = pd.read_csv(args.counts)
    required = {"model", "task_id", "n", "s"}
    missing = required - set(counts.columns)
    if missing:
        raise SystemExit(f"--counts missing columns: {sorted(missing)}")

    model_index = {m: i for i, m in enumerate(model_names)}
    counts = counts[counts["model"].isin(model_names)].reset_index(drop=True)

    task_ids = counts["task_id"].astype(str).to_numpy()
    x_obs = build_task_x(runs_df, task_ids, mean_log_t_hours)

    theta_cols = [f"theta.{i}" for i in range(1, len(model_names) + 1)]
    theta_all = draws_df[theta_cols].to_numpy(dtype=float)  # (S, I)
    ii = counts["model"].map(model_index).to_numpy(dtype=int)
    theta_obs = theta_all[:, ii]  # (S, Nobs)

    alpha = draws_df["alpha"].to_numpy(dtype=float)
    kappa = draws_df["kappa"].to_numpy(dtype=float)
    sigma_b = draws_df["sigma_b"].to_numpy(dtype=float)
    mu_loga = draws_df["mu_loga"].to_numpy(dtype=float)
    sigma_loga = draws_df["sigma_loga"].to_numpy(dtype=float)

    rng = np.random.default_rng(args.seed)
    p_hat = marginal_pred_binomial(
        theta=theta_obs,
        x=x_obs,
        alpha=alpha,
        kappa=kappa,
        sigma_b=sigma_b,
        mu_loga=mu_loga,
        sigma_loga=sigma_loga,
        mc=int(args.mc),
        rng=rng,
    )

    s = counts["s"].to_numpy(dtype=float)
    n = counts["n"].to_numpy(dtype=float)
    metrics = weighted_brier_and_logscore(s=s, n=n, p=p_hat)
    metrics.update({"run_id": run_id, "spec": spec, "theta_trend": str(meta.get("theta_trend", "none"))})
    pd.DataFrame([metrics]).to_csv(args.outdir / "calibration_metrics.csv", index=False)

    bins_df = calibration_bins(s=s, n=n, p=p_hat, bins=int(args.bins))
    bins_df.to_csv(args.outdir / "calibration_bins.csv", index=False)

    # Pooled PPC by task length (time-binned).
    t_hours = task_hours(runs_df, task_ids)
    ppc_df = time_binned_ppc(t_hours=t_hours, s=s, n=n, p=p_hat, bins=int(args.time_bins))
    ppc_df.to_csv(args.outdir / "ppc_time_bins.csv", index=False)

    # Plots
    from matplotlib import pyplot as plt

    # Reliability curve
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.errorbar(
        bins_df["p_mean"],
        bins_df["obs_rate"],
        yerr=bins_df["se"],
        fmt="o-",
        color="C0",
        ecolor="C0",
        capsize=2,
        linewidth=2,
    )
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed success rate")
    ax.set_title("Calibration (marginal over task random effects)")
    fig.tight_layout()
    fig.savefig(args.outdir / "calibration.png", dpi=200)
    plt.close(fig)

    # Difficulty residuals vs x
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8))
    ax.scatter(resid_df["x"], resid_df["u_hat"], s=14, alpha=0.45)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Centered log task length  $x = \log(t_{hours}) - \overline{\log t}$")
    ax.set_ylabel(r"Residual difficulty  $\hat u_j = \hat b_j - (\hat\alpha + \hat\kappa x_j)$")
    ax.set_title("Difficulty residuals vs task length (diagnostic)")
    fig.tight_layout()
    fig.savefig(args.outdir / "difficulty_residuals.png", dpi=200)
    plt.close(fig)

    # Pooled PPC by time bins
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8))
    ax.plot(ppc_df["t_geo_hours"], ppc_df["pred_rate"], "o-", color="C1", linewidth=2, label="predicted (marginal)")
    ax.errorbar(
        ppc_df["t_geo_hours"],
        ppc_df["obs_rate"],
        yerr=ppc_df["obs_se"],
        fmt="o",
        color="C0",
        ecolor="C0",
        capsize=2,
        label="observed",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Task length (hours, log scale)")
    ax.set_ylabel("Success rate")
    ax.set_title("Posterior predictive check by task length (pooled)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(args.outdir / "ppc_time_bins.png", dpi=200)
    plt.close(fig)

    out_meta = {
        "fitdir": str(args.fitdir),
        "outdir": str(args.outdir),
        "run_id": run_id,
        "spec": spec,
        "draws_used": int(len(draws_df)),
        "mc": int(args.mc),
        "bins": int(args.bins),
        "time_bins": int(args.time_bins),
        "loo": loo_df.to_dict(orient="records")[0],
        "metrics": metrics,
    }
    (args.outdir / "meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    print(f"wrote: {args.outdir / 'loo.csv'}")
    print(f"wrote: {args.outdir / 'calibration.png'}")
    print(f"wrote: {args.outdir / 'difficulty_residuals.png'}")
    print(f"wrote: {args.outdir / 'ppc_time_bins.png'}")


if __name__ == "__main__":
    main()
