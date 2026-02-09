#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute posterior mean + 95% credible interval for the release date when the trend-based "
            "horizon reaches 1 month (30 days), for multiple θ(d) trend specs."
        )
    )
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument(
        "--specs",
        type=str,
        default="time_irt__theta_linear,time_irt__theta_quadratic,time_irt__theta_xpow,time_irt__theta_t50_logistic",
        help="Comma-separated specs (default: linear,quadratic,xpow,t50_logistic).",
    )
    p.add_argument("--run-id", type=str, default=None, help="If set, use this run-id for all specs (otherwise LATEST).")
    p.add_argument("--p", type=float, default=0.5, help="Horizon probability level (default: 0.5).")
    p.add_argument(
        "--kind",
        choices=["marginal", "typical"],
        default="marginal",
        help="Horizon definition for the threshold calculation (default: marginal).",
    )
    p.add_argument(
        "--benchmark-yaml",
        type=Path,
        default=Path("data_raw/benchmark_results_1_1.yaml"),
        help="Benchmark YAML with release dates (default: data_raw/benchmark_results_1_1.yaml).",
    )
    p.add_argument("--draws", type=int, default=1500, help="Posterior draws to use (subsampled) (default: 1500).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed (default: 123).")
    p.add_argument(
        "--max-year",
        type=int,
        default=2200,
        help="Treat crossings after Dec 31 of this year as 'not reached' (default: 2200).",
    )
    p.add_argument(
        "--threshold-days",
        type=float,
        default=30.0,
        help="Horizon threshold in days (default: 30 = 1 month).",
    )
    p.add_argument(
        "--threshold-label",
        type=str,
        default=None,
        help="Human-readable label for the threshold (auto-derived if omitted).",
    )
    p.add_argument("--out-csv", type=Path, default=Path("blog/_generated/one_month_horizon.csv"))
    p.add_argument("--out-md", type=Path, default=Path("blog/_generated/one_month_horizon.md"))
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip specs that are missing under outputs/runs/ (default: false).",
    )
    return p.parse_args()


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")
    return math.log(p / (1.0 - p))


def chain_csv_files(fitdir: Path) -> list[Path]:
    files = sorted(fitdir.glob("*.csv"))
    keep: list[Path] = []
    for f in files:
        if f.name in {"summary.csv", "theta.csv", "b.csv", "a.csv"}:
            continue
        keep.append(f)
    return keep


def read_draws_subset(
    fitdir: Path, *, draws: int, seed: int, cols: list[str]
) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    meta = json.loads((fitdir / "meta.json").read_text(encoding="utf-8"))
    model_names = pd.read_csv(fitdir / "theta.csv")["model"].astype(str).tolist()

    chain_files = chain_csv_files(fitdir)
    if not chain_files:
        raise SystemExit(f"No chain CSVs found in {fitdir}")

    frames = [pd.read_csv(f, comment="#", usecols=cols) for f in chain_files]
    all_draws = pd.concat(frames, ignore_index=True)

    rng = np.random.default_rng(seed)
    if len(all_draws) > draws:
        idx = rng.choice(len(all_draws), size=draws, replace=False)
        all_draws = all_draws.iloc[np.sort(idx)].reset_index(drop=True)

    return all_draws, meta, model_names


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
        out[str(model_name)] = ts.normalize()
    return out


def resolve_run_id(runs_root: Path, spec: str, run_id: str | None) -> str:
    if run_id:
        return run_id
    latest = runs_root / spec / "LATEST"
    if not latest.exists():
        raise SystemExit(f"Missing {latest}")
    return latest.read_text(encoding="utf-8").strip()


def crossing_x_from_poly(
    *, A: np.ndarray, B: np.ndarray, C: np.ndarray | None, target: float, x_start: float
) -> np.ndarray:
    """
    Solve for x where A + B x (+ C x^2) = target, returning the earliest solution >= x_start per draw.
    A,B,C are arrays over draws.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if C is None:
        C0 = np.zeros_like(A)
    else:
        C0 = np.asarray(C, dtype=float)
    out = np.full_like(A, np.nan)

    # Quadratic when C != 0; otherwise linear.
    eps = 1e-10
    is_quad = np.abs(C0) > eps
    is_lin = ~is_quad

    # Linear case: A + Bx = target
    if np.any(is_lin):
        b = B[is_lin]
        a = A[is_lin]
        x = np.full_like(a, np.nan)
        good = np.abs(b) > eps
        x[good] = (target - a[good]) / b[good]
        # If trend is flat/negative, treat as "no crossing" unless already at/above target at x_start.
        # Enforce the reporting convention x_cross >= x_start.
        x = np.where(np.isfinite(x) & (x >= x_start), x, np.nan)
        out[is_lin] = x

    # Quadratic case: Cx^2 + Bx + (A-target)=0
    if np.any(is_quad):
        Aq = A[is_quad] - target
        Bq = B[is_quad]
        Cq = C0[is_quad]
        D = Bq * Bq - 4.0 * Cq * Aq
        ok = D >= 0
        xq = np.full_like(Aq, np.nan)
        if np.any(ok):
            sqrtD = np.sqrt(D[ok])
            denom = 2.0 * Cq[ok]
            r1 = (-Bq[ok] - sqrtD) / denom
            r2 = (-Bq[ok] + sqrtD) / denom
            # choose the smallest root >= x_start
            r1_ok = np.isfinite(r1) & (r1 >= x_start)
            r2_ok = np.isfinite(r2) & (r2 >= x_start)
            chosen = np.where(r1_ok & r2_ok, np.minimum(r1, r2), np.where(r1_ok, r1, np.where(r2_ok, r2, np.nan)))
            xq[ok] = chosen
        out[is_quad] = xq

    return out


def summarize_dates(date0: pd.Timestamp, x_cross: np.ndarray, *, x_start: float, x_max: float) -> dict[str, object]:
    x = np.asarray(x_cross, dtype=float)
    ok = np.isfinite(x) & (x >= x_start) & (x <= x_max)
    frac = float(np.mean(ok)) if x.size else float("nan")
    if not np.any(ok):
        return {
            "draws": int(x.size),
            "frac_crossing": frac,
            "x_start_years": float(x_start),
            "x_max_years": float(x_max),
            "mean_date": None,
            "ci95_low": None,
            "ci95_high": None,
            "mean_years_after_last": None,
            "ci95_years_after_last_low": None,
            "ci95_years_after_last_high": None,
        }

    x_ok = x[ok]
    mean_x = float(np.mean(x_ok))
    q025, q975 = np.quantile(x_ok, [0.025, 0.975])

    def x_to_date_str(xv: float) -> str | None:
        # Pandas Timestamps overflow outside roughly [1677, 2262].
        try:
            d = (date0 + pd.to_timedelta(float(xv) * 365.25, unit="D")).normalize()
            return str(d.date())
        except Exception:
            approx_year = float(date0.year) + float(xv)
            if not np.isfinite(approx_year):
                return None
            if approx_year > 9999:
                return "~far future"
            if approx_year < 0:
                return "~far past"
            return f"~{int(round(approx_year))}"

    mean_date = x_to_date_str(mean_x)
    low_date = x_to_date_str(float(q025))
    high_date = x_to_date_str(float(q975))

    yrs_after = x_ok - x_start
    mean_ya = float(np.mean(yrs_after))
    ya025, ya975 = np.quantile(yrs_after, [0.025, 0.975])

    return {
        "draws": int(x.size),
        "frac_crossing": frac,
        "x_start_years": float(x_start),
        "x_max_years": float(x_max),
        "mean_date": mean_date,
        "ci95_low": low_date,
        "ci95_high": high_date,
        "mean_years_after_last": mean_ya,
        "ci95_years_after_last_low": float(ya025),
        "ci95_years_after_last_high": float(ya975),
    }


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    specs = [s.strip() for s in args.specs.split(",") if s.strip()]
    thr_hours = args.threshold_days * 24.0
    target_logt = float(np.log(thr_hours))

    # Human-readable label for the markdown output.
    if args.threshold_label is not None:
        thr_label = args.threshold_label
    else:
        days = args.threshold_days
        if abs(days - 30.0) < 0.1:
            thr_label = "1 month"
        elif days < 1:
            thr_label = f"{days * 24:.0f} hours"
        elif days < 60:
            thr_label = f"{days:.0f} days"
        elif days < 365.25:
            thr_label = f"{days / 30:.0f} months"
        else:
            yrs = days / 365.25
            thr_label = f"{yrs:.0f} years" if yrs == int(yrs) else f"{yrs:.1f} years"

    release = load_release_dates(args.benchmark_yaml)

    rows = []
    for spec in specs:
        try:
            run_id = resolve_run_id(args.runs_root, spec, args.run_id)
        except SystemExit:
            if args.skip_missing:
                continue
            raise
        fitdir = args.runs_root / spec / run_id / "fit"
        if not fitdir.exists():
            if args.skip_missing:
                continue
            raise SystemExit(f"Missing fitdir: {fitdir}")

        meta = json.loads((fitdir / "meta.json").read_text(encoding="utf-8"))
        trend = str(meta.get("theta_trend", "none"))
        if trend not in {"linear", "quadratic", "quadratic_pos", "sqrt", "log1p", "xpow", "t50_logistic"}:
            raise SystemExit(f"{spec}: expected linear/quadratic/xpow/t50_logistic trend, got theta_trend={trend!r}")

        K = int(meta.get("theta_trend_K", 0)) if "theta_trend_K" in meta else 0
        if trend in {"linear", "sqrt", "log1p"} and K not in {0, 1}:
            raise SystemExit(f"{spec}: expected K=1, got {K}")
        if trend in {"quadratic", "quadratic_pos"} and K not in {0, 2}:
            raise SystemExit(f"{spec}: expected K=2, got {K}")

        date0 = pd.to_datetime(str(meta["theta_trend_date0"])).normalize()

        # Determine last-dated model time (x_start) using the same centering as the fit.
        model_names = meta.get("model_names")
        if not isinstance(model_names, list) or not model_names:
            # fall back to theta.csv if meta missing it
            model_names = pd.read_csv(fitdir / "theta.csv")["model"].astype(str).tolist()
        xs = []
        dates = []
        for m in model_names:
            if m == "human":
                continue
            d = release.get(str(m))
            if d is None:
                continue
            dates.append(d)
            xs.append((d - date0).days / 365.25)
        if not xs:
            raise SystemExit(f"{spec}: no dated models found in {args.benchmark_yaml}")
        x_start = float(np.max(xs))
        last_date = date0 + pd.to_timedelta(x_start * 365.25, unit="D")
        end_date = pd.Timestamp(year=int(args.max_year), month=12, day=31)
        x_max = float(((end_date.normalize() - date0).days) / 365.25)

        mean_log_t_hours = float(meta.get("mean_log_t_hours", float("nan")))
        if not np.isfinite(mean_log_t_hours):
            raise SystemExit(f"{spec}: fit meta.json missing mean_log_t_hours")

        # Columns we need.
        cols = ["alpha", "kappa", "sigma_b", "mu_loga"]
        if trend in {"linear", "quadratic", "quadratic_pos", "sqrt", "log1p"}:
            if K <= 0:
                K = 1 if trend in {"linear", "sqrt", "log1p"} else 2
            cols += ["gamma0"] + [f"gamma.{k}" for k in range(1, K + 1)]
        elif trend == "xpow":
            cols += ["gamma0", "gamma1", "a_pow"]
        elif trend == "t50_logistic":
            cols += ["log_t_low", "log_delta_t", "a_t", "b_t"]
        draws_df, _, _ = read_draws_subset(fitdir, draws=args.draws, seed=args.seed, cols=cols)

        alpha = draws_df["alpha"].to_numpy(dtype=float)
        kappa = draws_df["kappa"].to_numpy(dtype=float)
        sigma_b = draws_df["sigma_b"].to_numpy(dtype=float)
        mu_loga = draws_df["mu_loga"].to_numpy(dtype=float)

        lp = logit(float(args.p))
        a_eff = np.exp(mu_loga)
        if args.kind == "typical":
            adj = lp / a_eff
        else:
            c = (math.pi**2) / 3.0
            adj = (lp / a_eff) * np.sqrt(1.0 + c * ((a_eff * sigma_b) ** 2))

        if trend == "t50_logistic":
            if abs(float(args.p) - 0.5) > 1e-12:
                raise SystemExit("t50_logistic trend is defined in terms of the t50 horizon; use --p 0.5.")
            t_low = np.exp(draws_df["log_t_low"].to_numpy(dtype=float))
            t_high = t_low + np.exp(draws_df["log_delta_t"].to_numpy(dtype=float))
            a_t = draws_df["a_t"].to_numpy(dtype=float)
            b_t = draws_df["b_t"].to_numpy(dtype=float)
            # Solve t_low + (t_high-t_low)*sigmoid(a + b x) = threshold for x >= x_start.
            y = (thr_hours - t_low) / (t_high - t_low)
            ok = np.isfinite(y) & (y > 0) & (y < 1) & np.isfinite(b_t) & (b_t > 1e-9)
            x_cross = np.full_like(y, np.nan, dtype=float)
            x_cross[ok] = (np.log(y[ok] / (1.0 - y[ok])) - a_t[ok]) / b_t[ok]
            x_cross = np.where(np.isfinite(x_cross) & (x_cross >= x_start), x_cross, np.nan)
        elif trend == "sqrt":
            gamma0 = draws_df["gamma0"].to_numpy(dtype=float)
            gamma1 = draws_df["gamma.1"].to_numpy(dtype=float)
            x_min = float(meta.get("theta_trend_x_min", float("nan")))
            eps = float(meta.get("theta_trend_eps", 1e-6))
            if not np.isfinite(x_min):
                raise SystemExit(f"{spec}: meta.json missing theta_trend_x_min for sqrt trend")
            # Target theta for the requested horizon.
            theta_star = alpha + kappa * (target_logt - mean_log_t_hours) + adj
            # Solve gamma0 + gamma1*sqrt(x-x_min+eps) = theta_star.
            ok = np.isfinite(gamma1) & (gamma1 > 1e-9)
            z = np.full_like(theta_star, np.nan, dtype=float)
            z[ok] = (theta_star[ok] - gamma0[ok]) / gamma1[ok]
            ok2 = ok & np.isfinite(z) & (z >= 0)
            x_cross = np.full_like(theta_star, np.nan, dtype=float)
            x_cross[ok2] = x_min - eps + (z[ok2] ** 2)
            x_cross = np.where(np.isfinite(x_cross) & (x_cross >= x_start), x_cross, np.nan)
        elif trend == "log1p":
            gamma0 = draws_df["gamma0"].to_numpy(dtype=float)
            gamma1 = draws_df["gamma.1"].to_numpy(dtype=float)
            x_min = float(meta.get("theta_trend_x_min", float("nan")))
            scale = float(meta.get("theta_trend_scale", 1.0))
            if not np.isfinite(x_min):
                raise SystemExit(f"{spec}: meta.json missing theta_trend_x_min for log1p trend")
            scale = max(scale, 1e-9)
            theta_star = alpha + kappa * (target_logt - mean_log_t_hours) + adj
            ok = np.isfinite(gamma1) & (gamma1 > 1e-9)
            z = np.full_like(theta_star, np.nan, dtype=float)
            z[ok] = (theta_star[ok] - gamma0[ok]) / gamma1[ok]
            ok2 = ok & np.isfinite(z) & (z >= 0)
            x_cross = np.full_like(theta_star, np.nan, dtype=float)
            x_cross[ok2] = x_min + scale * (np.exp(z[ok2]) - 1.0)
            x_cross = np.where(np.isfinite(x_cross) & (x_cross >= x_start), x_cross, np.nan)
        elif trend == "xpow":
            gamma0 = draws_df["gamma0"].to_numpy(dtype=float)
            gamma1 = draws_df["gamma1"].to_numpy(dtype=float)
            a_pow = draws_df["a_pow"].to_numpy(dtype=float)
            x_min = float(meta.get("theta_trend_x_min", float("nan")))
            x_scale = float(meta.get("theta_trend_x_scale", float("nan")))
            x_eps = float(meta.get("theta_trend_x_eps", 1e-6))
            if not np.isfinite(x_min) or not np.isfinite(x_scale):
                raise SystemExit(f"{spec}: meta.json missing theta_trend_x_min/x_scale for xpow trend")
            theta_star = alpha + kappa * (target_logt - mean_log_t_hours) + adj
            ok = np.isfinite(gamma1) & (gamma1 > 1e-9) & np.isfinite(a_pow) & (a_pow > 1e-6)
            rhs = np.full_like(theta_star, np.nan, dtype=float)
            rhs[ok] = (theta_star[ok] - gamma0[ok]) / gamma1[ok]
            ok2 = ok & np.isfinite(rhs) & (rhs >= 0)
            z = np.full_like(theta_star, np.nan, dtype=float)
            z[ok2] = rhs[ok2] ** (1.0 / a_pow[ok2])
            x_cross = np.full_like(theta_star, np.nan, dtype=float)
            x_cross[ok2] = x_min - x_eps + x_scale * z[ok2]
            x_cross = np.where(np.isfinite(x_cross) & (x_cross >= x_start), x_cross, np.nan)
        else:
            gamma0 = draws_df["gamma0"].to_numpy(dtype=float)
            gamma = np.column_stack([draws_df[f"gamma.{k}"].to_numpy(dtype=float) for k in range(1, K + 1)])
            # log t(x) = mean_log_t + (theta(x) - adj - alpha)/kappa.
            A = mean_log_t_hours + (gamma0 - adj - alpha) / kappa
            B = gamma[:, 0] / kappa
            C = (gamma[:, 1] / kappa) if K == 2 else None
            x_cross = crossing_x_from_poly(A=A, B=B, C=C, target=target_logt, x_start=x_start)

        # Apply a max-year cutoff to avoid wild extrapolations dominating means.
        x_cross = np.where(np.isfinite(x_cross) & (x_cross <= x_max), x_cross, np.nan)
        summ = summarize_dates(date0, x_cross, x_start=x_start, x_max=x_max)
        rows.append(
            {
                "spec": spec,
                "run_id": run_id,
                "theta_trend": trend,
                "p": float(args.p),
                "kind": str(args.kind),
                "threshold_hours": thr_hours,
                "max_year": int(args.max_year),
                "date0": str(date0.date()),
                "last_dated_model_date": str(last_date.date()),
                **summ,
            }
        )

    out = pd.DataFrame(rows).sort_values(["spec"]).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)

    # Blog-friendly markdown
    md_lines: list[str] = []
    md_lines.append(f"## When do we hit a {thr_label} horizon?")
    md_lines.append("")
    md_lines.append(
        f"Threshold: **{thr_label}** ({args.threshold_days:.0f} days = {thr_hours:.0f} hours). "
        f"We report the predicted **release date** when the *trend-based* "
        f"**t{int(round(100*args.p))} horizon** reaches {thr_label}, using the **{args.kind}** horizon definition."
    )
    md_lines.append(f"`P(cross)` is the posterior probability of crossing **by {int(args.max_year)}-12-31**.")
    md_lines.append("")
    show = out[
        [
            "theta_trend",
            "frac_crossing",
            "mean_date",
            "ci95_low",
            "ci95_high",
            "mean_years_after_last",
            "ci95_years_after_last_low",
            "ci95_years_after_last_high",
        ]
    ].copy()
    show = show.rename(
        columns={
            "theta_trend": "Trend model",
            "frac_crossing": "P(cross)",
            "mean_date": "Mean date",
            "ci95_low": "95% CrI (low)",
            "ci95_high": "95% CrI (high)",
            "mean_years_after_last": "Mean years after last model",
            "ci95_years_after_last_low": "95% CrI years after (low)",
            "ci95_years_after_last_high": "95% CrI years after (high)",
        }
    )
    show["Mean date"] = show["Mean date"].fillna("not reached")
    show["95% CrI (low)"] = show["95% CrI (low)"].fillna("")
    show["95% CrI (high)"] = show["95% CrI (high)"].fillna("")
    show["P(cross)"] = show["P(cross)"].map(lambda v: f"{float(v):.2f}" if pd.notna(v) else "")
    for c in ["Mean years after last model", "95% CrI years after (low)", "95% CrI years after (high)"]:
        show[c] = show[c].map(lambda v: f"{float(v):.2f}" if pd.notna(v) and np.isfinite(v) else "")
    md_lines.append(show.to_markdown(index=False, disable_numparse=True))
    md_lines.append("")
    args.out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_md}")


if __name__ == "__main__":
    main()
