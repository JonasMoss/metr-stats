#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd
import yaml
from cmdstanpy import CmdStanModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit the core time-IRT model (2PL with log-linear difficulty vs log task length, constant discrimination) "
            "using Stan (cmdstanpy)."
        )
    )
    p.add_argument("--counts", type=Path, default=Path("data/irt_counts_task_id.csv"))
    p.add_argument("--runs", type=Path, default=Path("data/runs.pkl"))
    p.add_argument(
        "--benchmark-yaml",
        type=Path,
        default=Path("data_raw/benchmark_results_1_1.yaml"),
        help="Benchmark YAML with release dates (default: data_raw/benchmark_results_1_1.yaml).",
    )
    p.add_argument(
        "--theta-trend",
        choices=[
            "none",
            "linear",
            "quadratic",
            "quadratic_pos",
            "sqrt",
            "log1p",
            "xpow",
            "t50_logistic",
            "singularity",
            "singularity_nolinear",
        ],
        default="none",
        help="Optional joint ability trend θ(d) inside Stan (default: none).",
    )
    p.add_argument(
        "--stan",
        type=Path,
        default=None,
        help="Override Stan file (default depends on --theta-trend).",
    )

    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (default: timestamp like 20260206_130000).",
    )

    p.add_argument("--anchor-low", type=str, default=None)
    p.add_argument("--anchor-high", type=str, default=None)
    p.add_argument("--theta-low", type=float, default=-1.0)
    p.add_argument("--theta-high", type=float, default=1.0)

    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--iter-warmup", type=int, default=1000)
    p.add_argument("--iter-sampling", type=int, default=1000)
    p.add_argument("--adapt-delta", type=float, default=0.95)
    p.add_argument("--max-treedepth", type=int, default=12)
    p.add_argument("--refresh", type=int, default=100)

    p.add_argument("--list-models", action="store_true", help="Print pass rates for models in --counts, then exit.")
    return p.parse_args()


def load_counts(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    required = {"model", "n", "s"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"--counts missing columns: {sorted(missing)}")
    item_cols = [c for c in df.columns if c not in {"model", "n", "s", "f"}]
    if item_cols != ["task_id"]:
        raise SystemExit(f"--counts must be task-level counts with column task_id (got item cols={item_cols})")
    return df, "task_id"


def pick_anchors(counts_df: pd.DataFrame, model_names: list[str], anchor_low: str | None, anchor_high: str | None) -> tuple[str, str]:
    totals = counts_df.groupby("model", as_index=True).agg(n=("n", "sum"), s=("s", "sum"))
    rates = (totals["s"] / totals["n"]).sort_values()
    preferred_low = "gpt_4"
    preferred_high = "claude_3_7_sonnet_inspect"

    low = anchor_low or (preferred_low if preferred_low in set(model_names) else str(rates.index[0]))
    high = anchor_high or (preferred_high if preferred_high in set(model_names) else str(rates.index[-1]))

    if low not in set(model_names):
        raise SystemExit(f"--anchor-low {low!r} not found in counts")
    if high not in set(model_names):
        raise SystemExit(f"--anchor-high {high!r} not found in counts")
    if low == high:
        raise SystemExit("anchor-low and anchor-high must be different models")
    return low, high


def build_x_log_hours(runs_df: pd.DataFrame, task_ids: list[str]) -> tuple[np.ndarray, float]:
    task_time = runs_df.groupby("task_id", as_index=True)["human_minutes"].first()
    missing = [t for t in task_ids if t not in task_time.index]
    if missing:
        raise SystemExit(f"Missing human_minutes for {len(missing)} task_ids (example: {missing[:5]})")
    t_minutes = task_time.reindex(task_ids).to_numpy(dtype=float)
    if np.any(t_minutes <= 0) or np.any(~np.isfinite(t_minutes)):
        raise SystemExit("Non-positive or invalid task lengths encountered.")
    t_hours = t_minutes / 60.0
    x_raw = np.log(t_hours)
    mean_x = float(np.mean(x_raw))
    x = x_raw - mean_x
    return x, mean_x


def summarize(draws: np.ndarray, names: list[str], out_path: Path, key: str) -> None:
    mean = draws.mean(axis=0)
    sd = draws.std(axis=0, ddof=1)
    q05, q50, q95 = np.quantile(draws, [0.05, 0.50, 0.95], axis=0)
    pd.DataFrame({key: names, "mean": mean, "sd": sd, "q05": q05, "q50": q50, "q95": q95}).to_csv(out_path, index=False)


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


def build_release_x(
    model_names: list[str], benchmark_yaml: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    release = load_release_dates(benchmark_yaml)
    has_date = np.array([1 if m in release and m != "human" else 0 for m in model_names], dtype=int)
    dates = [release.get(m) for m in model_names]
    dated = [d for d, ok in zip(dates, has_date) if ok == 1]
    if len(dated) < 3:
        raise SystemExit("Need at least 3 models with release dates to fit a θ(d) trend.")
    dated_s = pd.to_datetime(pd.Series(dated))
    date0 = pd.to_datetime(dated_s.astype("int64").mean()).normalize()

    x = np.zeros(len(model_names), dtype=float)
    for i, (d, ok) in enumerate(zip(dates, has_date)):
        if ok == 1 and d is not None:
            x[i] = (pd.to_datetime(d).normalize() - date0).days / 365.25

    meta = {"theta_trend_date0": str(date0.date())}
    return x, has_date, meta


def build_theta_trend_design(
    model_names: list[str], benchmark_yaml: Path, theta_trend: str
) -> tuple[int, np.ndarray, np.ndarray, dict[str, object]]:
    if theta_trend == "none":
        raise ValueError("build_theta_trend_design called with theta_trend=none")
    x, has_date, meta0 = build_release_x(model_names, benchmark_yaml)

    if theta_trend == "linear":
        K = 1
        X = x[:, None]
    elif theta_trend in {"quadratic", "quadratic_pos"}:
        K = 2
        X = np.column_stack([x, x * x])
    elif theta_trend == "sqrt":
        dated_x = x[has_date == 1]
        x_min = float(np.min(dated_x))
        eps = 1e-6
        z = np.sqrt(np.clip(x - x_min + eps, a_min=eps, a_max=None))
        K = 1
        X = z[:, None]
        meta0 = {**meta0, "theta_trend_x_min": x_min, "theta_trend_eps": eps}
    elif theta_trend == "log1p":
        dated_x = x[has_date == 1]
        x_min = float(np.min(dated_x))
        scale = 1.0  # 1 year; keeps feature in a reasonable numeric range
        z = np.clip(x - x_min, a_min=0.0, a_max=None) / scale
        feat = np.log1p(z)
        K = 1
        X = feat[:, None]
        meta0 = {**meta0, "theta_trend_x_min": x_min, "theta_trend_scale": scale}
    else:
        raise SystemExit(f"Unknown theta_trend: {theta_trend!r}")

    meta = {"theta_trend": theta_trend, **meta0, "theta_trend_K": K}
    return K, X, has_date, meta


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

    low, high = pick_anchors(counts_df, model_names, args.anchor_low, args.anchor_high)
    x, mean_log_t_hours = build_x_log_hours(runs_df, task_ids)

    model_index = {m: i + 1 for i, m in enumerate(model_names)}  # Stan 1-based
    task_index = {t: j + 1 for j, t in enumerate(task_ids)}
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

    theta_trend_meta: dict[str, object] = {"theta_trend": "none"}
    if args.theta_trend != "none":
        if args.theta_trend in {"linear", "quadratic", "quadratic_pos", "sqrt", "log1p"}:
            K, X_date, has_date, theta_trend_meta = build_theta_trend_design(model_names, args.benchmark_yaml, args.theta_trend)
            stan_data.update(
                {
                    "K": int(K),
                    "X_date": X_date.tolist(),
                    "has_date": has_date.tolist(),
                }
            )
        elif args.theta_trend == "xpow":
            x_date, has_date, theta_trend_meta0 = build_release_x(model_names, args.benchmark_yaml)
            dated = x_date[has_date == 1]
            x_min = float(np.min(dated))
            x_max = float(np.max(dated))
            x_scale = float(max(x_max - x_min, 1.0))
            x_eps = 1e-6
            theta_trend_meta = {**theta_trend_meta0, "theta_trend": args.theta_trend, "theta_trend_x_min": x_min, "theta_trend_x_scale": x_scale, "theta_trend_x_eps": x_eps}
            stan_data.update(
                {
                    "x_date": x_date.tolist(),
                    "x_min": x_min,
                    "x_scale": x_scale,
                    "x_eps": x_eps,
                    "has_date": has_date.tolist(),
                }
            )
        elif args.theta_trend == "t50_logistic":
            x_date, has_date, theta_trend_meta0 = build_release_x(model_names, args.benchmark_yaml)
            theta_trend_meta = {"theta_trend": args.theta_trend, **theta_trend_meta0}
            stan_data.update(
                {
                    "x_date": x_date.tolist(),
                    "has_date": has_date.tolist(),
                    "mean_log_t_hours": float(mean_log_t_hours),
                }
            )
        elif args.theta_trend in {"singularity", "singularity_nolinear"}:
            x_date, has_date, theta_trend_meta0 = build_release_x(model_names, args.benchmark_yaml)
            x_date_max = float(np.max(x_date[has_date == 1]))
            theta_trend_meta = {"theta_trend": args.theta_trend, **theta_trend_meta0, "x_date_max": x_date_max}
            stan_data.update(
                {
                    "x_date": x_date.tolist(),
                    "x_date_max": x_date_max,
                    "has_date": has_date.tolist(),
                }
            )
        else:
            raise SystemExit(f"Unknown --theta-trend {args.theta_trend!r}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    spec = f"time_irt__theta_{args.theta_trend}"
    run_dir = args.runs_root / spec / run_id
    fit_dir = run_dir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)

    if args.stan is None:
        if args.theta_trend == "none":
            args.stan = Path("stan/2pl_time_loglinear.stan")
        elif args.theta_trend in {"linear", "quadratic", "sqrt", "log1p"}:
            args.stan = Path("stan/2pl_time_loglinear_theta_trend.stan")
        elif args.theta_trend == "quadratic_pos":
            args.stan = Path("stan/2pl_time_loglinear_theta_trend_quadratic_pos.stan")
        elif args.theta_trend == "xpow":
            args.stan = Path("stan/2pl_time_loglinear_theta_trend_xpow.stan")
        elif args.theta_trend == "t50_logistic":
            args.stan = Path("stan/2pl_time_loglinear_theta_trend_t50_logistic.stan")
        elif args.theta_trend == "singularity":
            args.stan = Path("experiments/stan/2pl_time_loglinear_theta_singularity.stan")
        elif args.theta_trend == "singularity_nolinear":
            args.stan = Path("experiments/stan/2pl_time_loglinear_theta_singularity_nolinear.stan")
        else:
            raise SystemExit(f"Unknown --theta-trend {args.theta_trend!r}")

    model = CmdStanModel(stan_file=str(args.stan))
    fit = model.sample(
        data=stan_data,
        seed=args.seed,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        refresh=args.refresh,
        output_dir=str(fit_dir),
    )

    fit.summary().to_csv(fit_dir / "summary.csv")
    summarize(fit.stan_variable("theta"), model_names, fit_dir / "theta.csv", "model")
    summarize(fit.stan_variable("b"), task_ids, fit_dir / "b.csv", "task_id")
    summarize(fit.stan_variable("a"), task_ids, fit_dir / "a.csv", "task_id")

    scalar_params = {}
    for name in ["alpha", "kappa", "sigma_b", "mu_loga", "sigma_loga"]:
        draws = fit.stan_variable(name).reshape(-1)
        scalar_params[name] = {
            "mean": float(draws.mean()),
            "sd": float(draws.std(ddof=1)),
            "q05": float(np.quantile(draws, 0.05)),
            "q50": float(np.quantile(draws, 0.50)),
            "q95": float(np.quantile(draws, 0.95)),
        }

    meta = {
        "run_id": run_id,
        "spec": spec,
        "counts": str(args.counts),
        "runs": str(args.runs),
        "benchmark_yaml": str(args.benchmark_yaml),
        "stan_file": str(args.stan),
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
        "x_transform": "log_hours",
        "mean_log_t_hours": mean_log_t_hours,
        **theta_trend_meta,
        "scalar_params": scalar_params,
        "model_names": model_names,
        "task_ids": task_ids,
        "N": int(len(counts_df)),
    }
    (fit_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (args.runs_root / spec / "LATEST").write_text(run_id + "\n", encoding="utf-8")

    print(f"wrote: {fit_dir / 'theta.csv'}")
    print(f"wrote: {fit_dir / 'summary.csv'}")
    print(f"wrote: {(args.runs_root / spec / 'LATEST')}")


if __name__ == "__main__":
    main()
