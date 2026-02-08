#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot a side-by-side comparison of horizon-vs-date curves for two specs (e.g. linear vs quadratic)."
    )
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--spec-a", type=str, default="time_irt__theta_linear")
    p.add_argument("--spec-b", type=str, default="time_irt__theta_quadratic")
    p.add_argument("--run-a", type=str, default=None, help="Run id for spec-a (default: spec-a/LATEST).")
    p.add_argument("--run-b", type=str, default=None, help="Run id for spec-b (default: spec-b/LATEST).")
    p.add_argument("--p", type=float, default=0.5, help="Horizon level (default: 0.5).")
    p.add_argument(
        "--kind",
        choices=["marginal", "typical"],
        default="typical",
        help="Which horizon definition to compare (default: typical).",
    )
    p.add_argument("--outdir", type=Path, default=Path("outputs/compare_trends"))
    p.add_argument(
        "--extend-to-year",
        type=int,
        default=2032,
        help="Extend x-axis to end of this year (default: 2032).",
    )
    return p.parse_args()


def _format_duration_hours(hours: float) -> str:
    if not np.isfinite(hours) or hours <= 0:
        return ""
    seconds = hours * 3600.0
    if seconds < 60:
        return f"{int(round(seconds))} sec"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{int(round(minutes))} min"
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
    y = np.asarray(y_values_hours, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if y.size == 0:
        return
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymin <= 0 or ymax <= 0:
        return

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


def resolve_run_id(runs_root: Path, spec: str, run_id: str | None) -> str:
    if run_id:
        return run_id
    return (runs_root / spec / "LATEST").read_text(encoding="utf-8").strip()


def load_horizon_grid(runs_root: Path, spec: str, run_id: str, kind: str) -> pd.DataFrame:
    figdir = runs_root / spec / run_id / "figures"
    path = figdir / ("horizon_grid_typical.csv" if kind == "typical" else "horizon_grid.csv")
    if not path.exists():
        raise SystemExit(f"Missing horizon grid: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    run_a = resolve_run_id(args.runs_root, args.spec_a, args.run_a)
    run_b = resolve_run_id(args.runs_root, args.spec_b, args.run_b)

    a = load_horizon_grid(args.runs_root, args.spec_a, run_a, args.kind)
    b = load_horizon_grid(args.runs_root, args.spec_b, run_b, args.kind)

    a = a[a["p"] == args.p].sort_values("date")
    b = b[b["p"] == args.p].sort_values("date")
    if a.empty or b.empty:
        raise SystemExit(f"No rows found for p={args.p:g} in one of the grids.")

    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(1, 1, figsize=(9.8, 5.4))
    ax.plot(a["date"], a["t_hours_q50"], color="C0", linewidth=2, label=f"{args.spec_a} ({args.kind})")
    ax.fill_between(a["date"], a["t_hours_q10"], a["t_hours_q90"], color="C0", alpha=0.12)
    ax.plot(b["date"], b["t_hours_q50"], color="C1", linewidth=2, label=f"{args.spec_b} ({args.kind})")
    ax.fill_between(b["date"], b["t_hours_q10"], b["t_hours_q90"], color="C1", alpha=0.12)

    ax.set_yscale("log")
    ax.set_xlabel("Release date")
    ax.set_ylabel(f"Task duration (humans) at t{int(round(100*args.p))}")
    ax.set_title(f"Horizon comparison (p={args.p:g}): linear vs quadratic θ(d)")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if args.extend_to_year is not None and args.extend_to_year > 0:
        end_date = pd.Timestamp(year=int(args.extend_to_year), month=12, day=31)
        ax.set_xlim(left=min(a["date"].min(), b["date"].min()), right=end_date)

    apply_metr_duration_ticks(ax, np.concatenate([a["t_hours_q50"].to_numpy(), b["t_hours_q50"].to_numpy()]))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    outpath = args.outdir / f"horizon_p{int(round(100*args.p)):02d}_{args.kind}_{args.spec_a}_vs_{args.spec_b}.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"wrote: {outpath}")


if __name__ == "__main__":
    main()
