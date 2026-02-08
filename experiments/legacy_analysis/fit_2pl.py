#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class PreparedData:
    agg: pd.DataFrame
    model_names: list[str]
    item_names: list[str]
    model_index: dict[str, int]
    item_index: dict[str, int]
    i_idx: np.ndarray
    j_idx: np.ndarray
    n: np.ndarray
    s: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a 2PL IRT model to runs.jsonl.")
    parser.add_argument(
        "--infile",
        type=Path,
        default=Path("data_raw/runs.jsonl"),
        help="Path to runs.jsonl (default: data_raw/runs.jsonl).",
    )
    parser.add_argument(
        "--item-col",
        choices=["task_id", "task_family"],
        default="task_id",
        help="Whether items are tasks or task families (default: task_id).",
    )
    parser.add_argument(
        "--exclude-fatal",
        action="append",
        default=[],
        help="Drop runs with fatal_error_from == VALUE (repeatable).",
    )
    parser.add_argument(
        "--min-item-success",
        type=int,
        default=1,
        help="Drop items with fewer than this many total successes (default: 1).",
    )
    parser.add_argument(
        "--min-item-fail",
        type=int,
        default=1,
        help="Drop items with fewer than this many total failures (default: 1).",
    )
    parser.add_argument(
        "--anchor-low",
        type=str,
        default=None,
        help="Model name to anchor at --theta-low (default: lowest pass-rate model).",
    )
    parser.add_argument(
        "--anchor-high",
        type=str,
        default=None,
        help="Model name to anchor at --theta-high (default: highest pass-rate model).",
    )
    parser.add_argument(
        "--theta-low",
        type=float,
        default=-1.0,
        help="Ability value for anchor-low (default: -1).",
    )
    parser.add_argument(
        "--theta-high",
        type=float,
        default=1.0,
        help="Ability value for anchor-high (default: +1).",
    )
    parser.add_argument(
        "--sigma-theta",
        type=float,
        default=2.0,
        help="Gaussian prior stddev on free thetas (default: 2).",
    )
    parser.add_argument(
        "--sigma-b",
        type=float,
        default=2.0,
        help="Gaussian prior stddev on difficulties b (default: 2).",
    )
    parser.add_argument(
        "--sigma-loga",
        type=float,
        default=1.0,
        help="Gaussian prior stddev on log discriminations log(a) (default: 1).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=2000,
        help="Maximum optimizer iterations (default: 2000).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/out"),
        help="Directory for CSV outputs (default: analysis/out).",
    )
    return parser.parse_args()


def prepare_data(
    infile: Path,
    item_col: str,
    exclude_fatal: list[str],
    min_item_success: int,
    min_item_fail: int,
) -> PreparedData:
    suffix = infile.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(infile)
    elif suffix == ".parquet":
        df = pd.read_parquet(infile)
    else:
        df = pd.read_json(infile, lines=True)
    df = df[["model", item_col, "score_binarized", "fatal_error_from"]].copy()
    df = df.dropna(subset=["model", item_col, "score_binarized"])
    df["score_binarized"] = df["score_binarized"].astype(int)

    if exclude_fatal:
        df = df[~df["fatal_error_from"].isin(exclude_fatal)]

    agg = (
        df.groupby(["model", item_col], as_index=False)["score_binarized"]
        .agg(n="count", s="sum")
        .astype({"n": int, "s": int})
    )

    item_totals = agg.groupby(item_col, as_index=True).agg(n=("n", "sum"), s=("s", "sum"))
    keep_items = item_totals[(item_totals["s"] >= min_item_success) & ((item_totals["n"] - item_totals["s"]) >= min_item_fail)].index
    agg = agg[agg[item_col].isin(keep_items)].reset_index(drop=True)

    model_names = sorted(agg["model"].unique().tolist())
    item_names = sorted(agg[item_col].unique().tolist())
    model_index = {m: idx for idx, m in enumerate(model_names)}
    item_index = {t: idx for idx, t in enumerate(item_names)}

    i_idx = agg["model"].map(model_index).to_numpy(dtype=int)
    j_idx = agg[item_col].map(item_index).to_numpy(dtype=int)
    n = agg["n"].to_numpy(dtype=float)
    s = agg["s"].to_numpy(dtype=float)

    return PreparedData(
        agg=agg,
        model_names=model_names,
        item_names=item_names,
        model_index=model_index,
        item_index=item_index,
        i_idx=i_idx,
        j_idx=j_idx,
        n=n,
        s=s,
    )


def pick_anchors(prepped: PreparedData, anchor_low: str | None, anchor_high: str | None) -> tuple[int, int]:
    totals = prepped.agg.groupby("model", as_index=True).agg(n=("n", "sum"), s=("s", "sum"))
    model_rates = (totals["s"] / totals["n"]).sort_values()
    if anchor_low is None:
        anchor_low = str(model_rates.index[0])
    if anchor_high is None:
        anchor_high = str(model_rates.index[-1])
    if anchor_low not in prepped.model_index:
        raise SystemExit(f"--anchor-low {anchor_low!r} not found in data")
    if anchor_high not in prepped.model_index:
        raise SystemExit(f"--anchor-high {anchor_high!r} not found in data")
    low_idx = prepped.model_index[anchor_low]
    high_idx = prepped.model_index[anchor_high]
    if low_idx == high_idx:
        raise SystemExit("anchor-low and anchor-high must be different models")
    return low_idx, high_idx


def fit_2pl(
    prepped: PreparedData,
    anchor_low_idx: int,
    anchor_high_idx: int,
    theta_low: float,
    theta_high: float,
    sigma_theta: float,
    sigma_b: float,
    sigma_loga: float,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    m = len(prepped.model_names)
    j = len(prepped.item_names)

    free_model_mask = np.ones(m, dtype=bool)
    free_model_mask[anchor_low_idx] = False
    free_model_mask[anchor_high_idx] = False
    free_model_indices = np.flatnonzero(free_model_mask)

    item_totals = prepped.agg.groupby(prepped.agg.columns[1], as_index=True).agg(n=("n", "sum"), s=("s", "sum"))
    item_rate = (item_totals["s"] + 0.5) / (item_totals["n"] + 1.0)
    item_rate = item_rate.reindex(prepped.item_names)
    b0 = -np.log(item_rate.to_numpy() / (1.0 - item_rate.to_numpy()))

    theta0 = np.zeros(len(free_model_indices), dtype=float)
    loga0 = np.zeros(j, dtype=float)

    x0 = np.concatenate([theta0, b0, loga0])

    i_idx = prepped.i_idx
    j_idx = prepped.j_idx
    n_obs = prepped.n
    s_obs = prepped.s

    inv_sigma_theta2 = 1.0 / (sigma_theta * sigma_theta)
    inv_sigma_b2 = 1.0 / (sigma_b * sigma_b)
    inv_sigma_loga2 = 1.0 / (sigma_loga * sigma_loga)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta_free = x[: len(free_model_indices)]
        b = x[len(free_model_indices) : len(free_model_indices) + j]
        loga = x[len(free_model_indices) + j :]
        theta = np.zeros(m, dtype=float)
        theta[anchor_low_idx] = theta_low
        theta[anchor_high_idx] = theta_high
        theta[free_model_indices] = theta_free
        a = np.exp(loga)
        return theta, b, a

    def objective_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        theta, b, a = unpack(x)

        x_ij = a[j_idx] * (theta[i_idx] - b[j_idx])

        softplus = np.logaddexp(0.0, x_ij)
        softplus_neg = np.logaddexp(0.0, -x_ij)
        nll = float(np.sum(s_obs * softplus_neg + (n_obs - s_obs) * softplus))

        theta_free = x[: len(free_model_indices)]
        b_vec = x[len(free_model_indices) : len(free_model_indices) + j]
        loga_vec = x[len(free_model_indices) + j :]
        penalty = 0.5 * (
            inv_sigma_theta2 * float(np.dot(theta_free, theta_free))
            + inv_sigma_b2 * float(np.dot(b_vec, b_vec))
            + inv_sigma_loga2 * float(np.dot(loga_vec, loga_vec))
        )

        p = expit(x_ij)
        resid = n_obs * p - s_obs

        g_theta_full = np.zeros(m, dtype=float)
        np.add.at(g_theta_full, i_idx, resid * a[j_idx])

        g_b = np.zeros(j, dtype=float)
        np.add.at(g_b, j_idx, -resid * a[j_idx])

        g_loga = np.zeros(j, dtype=float)
        np.add.at(g_loga, j_idx, resid * a[j_idx] * (theta[i_idx] - b[j_idx]))

        g_theta_free = g_theta_full[free_model_indices] + inv_sigma_theta2 * theta_free
        g_b = g_b + inv_sigma_b2 * b_vec
        g_loga = g_loga + inv_sigma_loga2 * loga_vec

        grad = np.concatenate([g_theta_free, g_b, g_loga])
        return nll + penalty, grad

    res = minimize(
        fun=lambda x: objective_and_grad(x)[0],
        x0=x0,
        jac=lambda x: objective_and_grad(x)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )

    theta_hat, b_hat, a_hat = unpack(res.x)
    meta: dict[str, object] = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "niter": int(res.nit),
        "fun": float(res.fun),
    }
    return theta_hat, b_hat, a_hat, meta


def main() -> None:
    args = parse_args()
    prepped = prepare_data(
        infile=args.infile,
        item_col=args.item_col,
        exclude_fatal=args.exclude_fatal,
        min_item_success=args.min_item_success,
        min_item_fail=args.min_item_fail,
    )
    low_idx, high_idx = pick_anchors(prepped, args.anchor_low, args.anchor_high)

    theta, b, a, meta = fit_2pl(
        prepped=prepped,
        anchor_low_idx=low_idx,
        anchor_high_idx=high_idx,
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        sigma_theta=args.sigma_theta,
        sigma_b=args.sigma_b,
        sigma_loga=args.sigma_loga,
        maxiter=args.maxiter,
    )

    model_df = pd.DataFrame(
        {
            "model": prepped.model_names,
            "theta": theta,
        }
    ).sort_values("theta", ascending=False)
    item_df = pd.DataFrame(
        {
            args.item_col: prepped.item_names,
            "a": a,
            "b": b,
        }
    )

    totals = prepped.agg.groupby("model", as_index=True).agg(n=("n", "sum"), s=("s", "sum"))
    totals["pass_rate"] = totals["s"] / totals["n"]
    model_df = model_df.merge(totals[["pass_rate"]], left_on="model", right_index=True, how="left")

    item_totals = prepped.agg.groupby(prepped.agg.columns[1], as_index=True).agg(n=("n", "sum"), s=("s", "sum"))
    item_totals["pass_rate"] = item_totals["s"] / item_totals["n"]
    item_df = item_df.merge(item_totals[["pass_rate"]], left_on=args.item_col, right_index=True, how="left")
    item_df = item_df.sort_values("b", ascending=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    model_path = args.outdir / "2pl_models.csv"
    item_path = args.outdir / "2pl_items.csv"
    meta_path = args.outdir / "2pl_meta.json"

    model_df.to_csv(model_path, index=False)
    item_df.to_csv(item_path, index=False)
    pd.Series(meta).to_json(meta_path, indent=2)

    low_name = prepped.model_names[low_idx]
    high_name = prepped.model_names[high_idx]
    print(
        "fit:",
        f"success={meta['success']}",
        f"niter={meta['niter']}",
        f"anchors=({low_name} -> {args.theta_low}, {high_name} -> {args.theta_high})",
    )
    print(f"wrote: {model_path}")
    print(f"wrote: {item_path}")
    print(f"wrote: {meta_path}")


if __name__ == "__main__":
    main()
