#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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
            "Fit monotone (decreasing) PAVA curves of p(success) vs task length and "
            "extract 50% horizons. Option A uses observed success rates; Option B uses "
            "plug-in 2PL probabilities from a fitted Stan model."
        )
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/runs.pkl"),
        help="Runs DataFrame pickle produced by scripts/build_data.py (default: data/runs.pkl).",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("data_raw/benchmark_results_1_1.yaml"),
        help="YAML with release dates (default: data_raw/benchmark_results_1_1.yaml).",
    )
    parser.add_argument(
        "--theta-csv",
        type=Path,
        default=Path("analysis/out_stan/theta.csv"),
        help="Theta summary CSV from fit_2pl_stan.py (for Option B) (default: analysis/out_stan/theta.csv).",
    )
    parser.add_argument(
        "--items-csv",
        type=Path,
        default=Path("analysis/out_stan/items.csv"),
        help="Items summary CSV from fit_2pl_stan.py (for Option B) (default: analysis/out_stan/items.csv).",
    )
    parser.add_argument(
        "--exclude-fatal",
        action="append",
        default=["usageLimits"],
        help="Drop runs with fatal_error_from == VALUE (repeatable; default: usageLimits).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model names for example panels (default: {','.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=12,
        help="Number of log-length bins for optional binned summary plots (default: 12).",
    )
    parser.add_argument(
        "--time-units",
        choices=["hours", "minutes"],
        default="hours",
        help="Units for horizons/plots (default: hours).",
    )
    parser.add_argument(
        "--label-top",
        type=int,
        default=10,
        help="Label this many highest-horizon models on horizon plots (default: 10).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/out_isotonic"),
        help="Output directory (default: analysis/out_isotonic).",
    )
    return parser.parse_args()


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pava_decreasing(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted PAVA for a non-increasing fit yhat(x) with x in increasing order.
    Returns yhat at each input x (piecewise constant on blocks).
    """
    if not (len(x) == len(y) == len(w)):
        raise ValueError("x, y, w must have same length")
    if len(x) == 0:
        return np.array([], dtype=float)

    # Each block stores (start_idx, end_idx, sum_w, sum_wy)
    blocks: list[list[float]] = []
    for idx, (wi, yi) in enumerate(zip(w, y)):
        blocks.append([idx, idx, float(wi), float(wi * yi)])
        # enforce non-increasing: prev_avg >= curr_avg
        while len(blocks) >= 2:
            s0, e0, w0, wy0 = blocks[-2]
            s1, e1, w1, wy1 = blocks[-1]
            avg0 = wy0 / w0
            avg1 = wy1 / w1
            if avg0 >= avg1:
                break
            # merge last two blocks
            blocks[-2] = [s0, e1, w0 + w1, wy0 + wy1]
            blocks.pop()

    yhat = np.empty(len(x), dtype=float)
    for s, e, sw, swy in blocks:
        avg = swy / sw
        yhat[int(s) : int(e) + 1] = avg
    return yhat


@dataclass(frozen=True)
class HorizonResult:
    t50: float
    censored: str  # "none", "left", "right"
    x50: float


def extract_t50_from_step(log_t: np.ndarray, p_hat: np.ndarray, threshold: float = 0.5) -> HorizonResult:
    """
    Given x=log(t) increasing and a non-increasing fitted p_hat(x), find the 50% horizon t50.
    Uses linear interpolation between adjacent fitted points when it jumps across the threshold.
    If entirely above/below threshold, returns boundary with censoring flag.
    """
    if len(log_t) == 0:
        return HorizonResult(t50=float("nan"), censored="none", x50=float("nan"))

    # Ensure sorted by x
    order = np.argsort(log_t)
    x = log_t[order]
    p = p_hat[order]

    if np.all(p > threshold):
        x50 = float(x.max())
        return HorizonResult(t50=float(np.exp(x50)), censored="right", x50=x50)
    if np.all(p < threshold):
        x50 = float(x.min())
        return HorizonResult(t50=float(np.exp(x50)), censored="left", x50=x50)

    # Find last index with p >= threshold and first with p <= threshold
    ge = np.where(p >= threshold)[0]
    le = np.where(p <= threshold)[0]
    left = int(ge.max())
    right = int(le.min())
    if left == right:
        x50 = float(x[left])
        return HorizonResult(t50=float(np.exp(x50)), censored="none", x50=x50)
    if left > right:
        # Shouldn't happen for non-increasing p, but guard anyway.
        x50 = float(x[right])
        return HorizonResult(t50=float(np.exp(x50)), censored="none", x50=x50)

    x0, p0 = float(x[left]), float(p[left])
    x1, p1 = float(x[right]), float(p[right])
    if p0 == p1:
        # Flat segment spanning threshold; pick midpoint in log space.
        x50 = 0.5 * (x0 + x1)
    else:
        # Linear interpolation in (x, p) space.
        x50 = x0 + (threshold - p0) * (x1 - x0) / (p1 - p0)
    return HorizonResult(t50=float(np.exp(x50)), censored="none", x50=float(x50))


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


def prepare_task_table(runs_df: pd.DataFrame, exclude_fatal: list[str]) -> pd.DataFrame:
    df = runs_df.copy()
    if exclude_fatal:
        df = df[~df["fatal_error_from"].isin(exclude_fatal)]
    df = df.dropna(subset=["model", "task_id", "score_binarized", "human_minutes"])
    df["score_binarized"] = df["score_binarized"].astype(int)

    agg = (
        df.groupby(["model", "task_id"], as_index=False)["score_binarized"]
        .agg(n="count", s="sum")
        .astype({"n": int, "s": int})
    )
    task_time = df.groupby("task_id", as_index=False)["human_minutes"].first().rename(columns={"human_minutes": "t_minutes"})
    out = agg.merge(task_time, on="task_id", how="left")
    return out


def load_plugin_probs(theta_csv: Path, items_csv: Path, task_tbl: pd.DataFrame) -> pd.DataFrame:
    theta = pd.read_csv(theta_csv).rename(columns={"name": "model"})
    items = pd.read_csv(items_csv)

    # Items.csv from fit_2pl_stan.py should have exactly one of {task_id, task_family}.
    item_cols = [c for c in items.columns if c in {"task_id", "task_family"}]
    if len(item_cols) != 1 or item_cols[0] != "task_id":
        raise SystemExit(
            f"{items_csv} must be task-level (contain task_id). "
            f"Re-fit Stan on data/irt_counts_task_id.csv if needed."
        )
    needed_item = {"task_id", "mean_a", "mean_b"}
    if not needed_item.issubset(items.columns):
        raise SystemExit(f"{items_csv} missing required columns: {sorted(needed_item - set(items.columns))}")

    needed_theta = {"model", "mean"}
    if not needed_theta.issubset(theta.columns):
        raise SystemExit(f"{theta_csv} missing required columns: {sorted(needed_theta - set(theta.columns))}")

    theta = theta[["model", "mean"]].rename(columns={"mean": "theta"})
    items = items[["task_id", "mean_a", "mean_b"]].rename(columns={"mean_a": "a", "mean_b": "b"})

    # Join to model-task attempt table so we can weight by n_ij and keep task lengths.
    df = task_tbl.merge(theta, on="model", how="inner").merge(items, on="task_id", how="inner")
    eta = df["a"].to_numpy(dtype=float) * (df["theta"].to_numpy(dtype=float) - df["b"].to_numpy(dtype=float))
    df["p_plugin"] = logistic(eta)
    return df


def fit_isotonic_and_horizon(df: pd.DataFrame, y_col: str, time_units: str) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for model, g in df.groupby("model", as_index=False):
        if g.empty:
            continue
        t = g["t_minutes"].to_numpy(dtype=float)
        if time_units == "hours":
            t = t / 60.0
        log_t = np.log(t)

        y = g[y_col].to_numpy(dtype=float)
        w = g["n"].to_numpy(dtype=float)
        order = np.argsort(log_t)
        log_t = log_t[order]
        y = y[order]
        w = w[order]

        # Collapse identical x by weighted average.
        uniq_x, inv = np.unique(log_t, return_inverse=True)
        w_sum = np.bincount(inv, weights=w)
        y_wsum = np.bincount(inv, weights=w * y)
        y_mean = y_wsum / w_sum

        p_hat = pava_decreasing(uniq_x, y_mean, w_sum)
        hz = extract_t50_from_step(uniq_x, p_hat, threshold=0.5)

        out_rows.append(
            {
                "model": model,
                "t50": hz.t50,
                "t50_units": time_units,
                "censored": hz.censored,
                "x50": hz.x50,
                "n_tasks": int(g["task_id"].nunique()),
                "n_attempts": int(g["n"].sum()),
            }
        )
    return pd.DataFrame(out_rows)


def plot_horizon_vs_date(horizons: pd.DataFrame, release_dates: dict[str, pd.Timestamp], outpath: Path, title: str, label_top: int) -> None:
    from matplotlib import pyplot as plt

    df = horizons.copy()
    df["release_date"] = df["model"].map(release_dates)
    df = df.dropna(subset=["release_date", "t50"]).copy()
    df = df.sort_values("t50", ascending=False)

    plt.figure(figsize=(9.5, 5.5))
    plt.scatter(df["release_date"], df["t50"], s=45, alpha=0.85)
    plt.yscale("log")
    plt.xlabel("Model release date")
    plt.ylabel(f"50% horizon t50 ({df['t50_units'].iloc[0] if len(df) else ''}, log scale)")
    plt.title(title)

    to_label = df.head(label_top)
    for _, row in to_label.iterrows():
        plt.annotate(
            row["model"],
            (row["release_date"], row["t50"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_example_curves(
    df_obs: pd.DataFrame,
    df_plugin: pd.DataFrame,
    models: list[str],
    time_units: str,
    outpath: Path,
) -> None:
    from matplotlib import pyplot as plt

    nrows = len(models)
    fig, axes = plt.subplots(nrows, 2, figsize=(12.0, 2.6 * nrows), sharex=True, sharey=True)
    if nrows == 1:
        axes = np.array([axes])

    def _panel(ax, g: pd.DataFrame, y_col: str, title: str) -> None:
        t = g["t_minutes"].to_numpy(dtype=float)
        if time_units == "hours":
            t = t / 60.0
        log_t = np.log(t)
        y = g[y_col].to_numpy(dtype=float)
        w = g["n"].to_numpy(dtype=float)

        order = np.argsort(log_t)
        log_t = log_t[order]
        y = y[order]
        w = w[order]

        uniq_x, inv = np.unique(log_t, return_inverse=True)
        w_sum = np.bincount(inv, weights=w)
        y_wsum = np.bincount(inv, weights=w * y)
        y_mean = y_wsum / w_sum

        p_hat = pava_decreasing(uniq_x, y_mean, w_sum)
        hz = extract_t50_from_step(uniq_x, p_hat, threshold=0.5)

        ax.scatter(np.exp(log_t), y, s=12, alpha=0.18)
        ax.step(np.exp(uniq_x), p_hat, where="post", color="black", linewidth=2)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.axvline(hz.t50, color="gray", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{title} (t50≈{hz.t50:.2g} {time_units})", loc="left", fontsize=10)

    for r, model in enumerate(models):
        g_obs = df_obs[df_obs["model"] == model]
        g_pl = df_plugin[df_plugin["model"] == model]
        _panel(axes[r, 0], g_obs, "p_obs", model)
        _panel(axes[r, 1], g_pl, "p_plugin", model)

    axes[-1, 0].set_xlabel(f"Task length t ({time_units}, log scale)")
    axes[-1, 1].set_xlabel(f"Task length t ({time_units}, log scale)")
    axes[0, 0].set_ylabel("p(success)")
    for r in range(1, nrows):
        axes[r, 0].set_ylabel("p(success)")
    axes[0, 0].text(0.02, 1.02, "Observed (Option A)", transform=axes[0, 0].transAxes, fontsize=10)
    axes[0, 1].text(0.02, 1.02, "2PL plug-in (Option B)", transform=axes[0, 1].transAxes, fontsize=10)
    fig.suptitle("Monotone PAVA fits vs task length (examples)", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.read_pickle(args.runs)
    release_dates = load_release_dates(args.yaml)

    task_tbl = prepare_task_table(runs_df, exclude_fatal=args.exclude_fatal)
    task_tbl["p_obs"] = task_tbl["s"] / task_tbl["n"]

    plugin_tbl = load_plugin_probs(args.theta_csv, args.items_csv, task_tbl)

    horizons_obs = fit_isotonic_and_horizon(task_tbl, y_col="p_obs", time_units=args.time_units)
    horizons_pl = fit_isotonic_and_horizon(plugin_tbl, y_col="p_plugin", time_units=args.time_units)

    horizons_obs["release_date"] = horizons_obs["model"].map(release_dates)
    horizons_pl["release_date"] = horizons_pl["model"].map(release_dates)

    horizons_obs.to_csv(args.outdir / "horizons_isotonic_observed.csv", index=False)
    horizons_pl.to_csv(args.outdir / "horizons_isotonic_plugin.csv", index=False)

    plot_horizon_vs_date(
        horizons_obs,
        release_dates,
        args.outdir / "horizon_vs_date_observed.png",
        title="50% horizon vs release date (monotone isotonic on observed success)",
        label_top=args.label_top,
    )
    plot_horizon_vs_date(
        horizons_pl,
        release_dates,
        args.outdir / "horizon_vs_date_plugin.png",
        title="50% horizon vs release date (monotone isotonic on 2PL plug-in probabilities)",
        label_top=args.label_top,
    )

    example_models = [m.strip() for m in args.models.split(",") if m.strip()]
    example_models = [m for m in example_models if m in set(task_tbl["model"].unique())]
    if example_models:
        plot_example_curves(
            df_obs=task_tbl,
            df_plugin=plugin_tbl,
            models=example_models,
            time_units=args.time_units,
            outpath=args.outdir / "examples_curves.png",
        )

    meta = {
        "runs": str(args.runs),
        "yaml": str(args.yaml),
        "theta_csv": str(args.theta_csv),
        "items_csv": str(args.items_csv),
        "exclude_fatal": args.exclude_fatal,
        "time_units": args.time_units,
        "example_models": example_models,
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote: {args.outdir / 'horizons_isotonic_observed.csv'}")
    print(f"wrote: {args.outdir / 'horizon_vs_date_observed.png'}")
    print(f"wrote: {args.outdir / 'horizon_vs_date_plugin.png'}")
    if example_models:
        print(f"wrote: {args.outdir / 'examples_curves.png'}")


if __name__ == "__main__":
    main()

