#!/usr/bin/env python3
"""Generate the two PDF figures for the IWSM 2026 paper.

Figure 1 (Moss_horizon_fan.pdf):
    50%-success horizon vs release date under four trajectory models.
    Composites all four specs onto one plot with distinct colors.

Figure 2 (Moss_marginal_typical.pdf):
    Two panels (linear + quadratic) showing 50% reference, 80% typical,
    and 80% marginal horizon curves, with shaded gap.

Reads CSV files from outputs/runs/<spec>/LATEST/../figures/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path so we can import from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_figures import (
    _format_duration_hours,
)

RUNS_ROOT = PROJECT_ROOT / "outputs" / "runs"

SPECS = {
    "Linear": "time_irt__theta_linear",
    "Quadratic": "time_irt__theta_quadratic_pos",
    "Power-law": "time_irt__theta_xpow",
    "Saturating": "time_irt__theta_theta_logistic",
}

SPEC_COLORS = {
    "Linear": "#1f77b4",       # blue
    "Quadratic": "#d62728",    # red
    "Power-law": "#2ca02c",    # green
    "Saturating": "#9467bd",   # purple
}

LINE_STYLES = {
    "Linear": "-",
    "Quadratic": "--",
    "Power-law": ":",
    "Saturating": "-.",
}


def resolve_latest(spec: str) -> Path:
    spec_dir = RUNS_ROOT / spec
    run_id = (spec_dir / "LATEST").read_text().strip()
    return spec_dir / run_id


def figdir(spec: str) -> Path:
    return resolve_latest(spec) / "figures"


def load_horizon_grid(spec: str, kind: str = "marginal") -> pd.DataFrame:
    fname = "horizon_grid.csv" if kind == "marginal" else "horizon_grid_typical.csv"
    df = pd.read_csv(figdir(spec) / fname)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_horizon_points(spec: str, kind: str = "marginal") -> pd.DataFrame:
    fname = "horizon_points.csv" if kind == "marginal" else "horizon_points_typical.csv"
    df = pd.read_csv(figdir(spec) / fname)
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate IWSM 2026 paper figures.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for PDFs (default: iwsm/).",
    )
    return p.parse_args()


def make_figure1(outdir: Path) -> None:
    """Horizon fan plot: p=0.5 marginal horizon from all four specs."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    x_max = pd.Timestamp("2029-01-01")

    for label, spec_name in SPECS.items():
        g = load_horizon_grid(spec_name).query("p == 0.5").sort_values("date")
        g = g[g["date"] <= x_max]
        color = SPEC_COLORS[label]
        ax.plot(
            g["date"],
            g["t_hours_q50"],
            color=color,
            linewidth=1.4,
            linestyle=LINE_STYLES[label],
            label=label,
        )
        ax.fill_between(
            g["date"],
            g["t_hours_q025"],
            g["t_hours_q975"],
            color=color,
            alpha=0.10,
        )

    # Observed model points from one spec (all give the same points)
    hpts = load_horizon_points(list(SPECS.values())[0]).query("p == 0.5")
    ax.errorbar(
        hpts["release_date"],
        hpts["t_hours_q50"],
        yerr=np.vstack([
            hpts["t_hours_q50"] - hpts["t_hours_q025"],
            hpts["t_hours_q975"] - hpts["t_hours_q50"],
        ]),
        fmt="o",
        color="black",
        ecolor="0.4",
        elinewidth=0.8,
        capsize=1.5,
        alpha=0.7,
        zorder=5,
        markersize=2.5,
    )

    # 1-month reference line
    ax.axhline(30 * 24, color="0.45", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(
        pd.Timestamp("2023-02-01"), 30 * 24 * 1.35, "1 month",
        fontsize=6, color="0.45", va="bottom",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Release date")
    ax.set_ylabel("50% horizon (human task time)")
    ax.set_xlim(pd.Timestamp("2023-01-01"), x_max)
    ax.set_ylim(0.03, 365 * 24)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    tick_hours = np.array([6 / 60, 1.0, 10.0, 4 * 24.0, 30 * 24.0, 6 * 30 * 24.0])
    ax.set_yticks(tick_hours)
    ax.set_yticklabels([_format_duration_hours(float(t)) for t in tick_hours])
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.2)

    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout(pad=0.4)

    path = outdir / "Moss_horizon_fan.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {path}")


# Colors for Figure 2
COL_REF = "#888888"       # gray for 50% reference
COL_TYPICAL = "#1f77b4"   # blue for 80% typical
COL_MARGINAL = "#d62728"  # red for 80% marginal
COL_GAP = "#d62728"       # red fill for the gap


def make_figure2(outdir: Path) -> None:
    """Marginal vs typical: two panels (linear + quadratic)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
    })

    panel_specs = [
        ("Linear", "time_irt__theta_linear"),
        ("Quadratic", "time_irt__theta_quadratic_pos"),
    ]
    x_max = pd.Timestamp("2029-01-01")

    fig, axes = plt.subplots(2, 1, figsize=(4.4, 3.2), sharex=True, sharey=True)

    for ax, (panel_label, spec_name) in zip(axes, panel_specs):
        marg = load_horizon_grid(spec_name, kind="marginal")
        typ = load_horizon_grid(spec_name, kind="typical")

        # 50% marginal as reference
        g50 = marg.query("p == 0.5").sort_values("date")
        g50 = g50[g50["date"] <= x_max]
        ax.plot(
            g50["date"], g50["t_hours_q50"],
            color=COL_REF, linewidth=1.0, label="50% marginal",
        )

        # 80% typical
        g80t = typ.query("p == 0.8").sort_values("date")
        g80t = g80t[g80t["date"] <= x_max]
        ax.plot(
            g80t["date"], g80t["t_hours_q50"],
            color=COL_TYPICAL, linewidth=1.4, linestyle="--", label="80% typical",
        )

        # 80% marginal
        g80m = marg.query("p == 0.8").sort_values("date")
        g80m = g80m[g80m["date"] <= x_max]
        ax.plot(
            g80m["date"], g80m["t_hours_q50"],
            color=COL_MARGINAL, linewidth=1.4, label="80% marginal",
        )

        # Shade the gap
        dates_common = g80t["date"].values
        typ_vals = g80t["t_hours_q50"].values
        marg_vals = g80m["t_hours_q50"].values
        ax.fill_between(
            dates_common, typ_vals, marg_vals,
            color=COL_GAP, alpha=0.12, label="Gap",
        )

        ax.set_yscale("log")
        ax.set_title(panel_label, fontsize=9, loc="left")
        ax.set_xlim(pd.Timestamp("2023-01-01"), x_max)
        ax.set_ylim(1e-3, 365 * 24)

        tick_hours = np.array([1 / 60, 6 / 60, 1.0, 10.0, 4 * 24.0, 30 * 24.0, 6 * 30 * 24.0])
        ax.set_yticks(tick_hours)
        ax.set_yticklabels([_format_duration_hours(float(t)) for t in tick_hours])
        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.2)

        ax.legend(loc="upper left", framealpha=0.9)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Release date")
    axes[0].set_ylabel("Horizon (human task time)")
    axes[1].set_ylabel("Horizon (human task time)")

    fig.tight_layout(pad=0.4)

    path = outdir / "Moss_marginal_typical.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {path}")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    make_figure1(args.outdir)
    make_figure2(args.outdir)


if __name__ == "__main__":
    main()
