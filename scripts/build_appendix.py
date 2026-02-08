#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare Quarto appendix inputs: run figures+diagnostics (if desired), copy plots into appendix/assets/, "
            "and write markdown fragments under appendix/_generated/ that reference the latest runs."
        )
    )
    p.add_argument(
        "--specs",
        type=str,
        default="time_irt__theta_linear,time_irt__theta_quadratic",
        help="Comma-separated specs to include (default: linear+quadratic).",
    )
    p.add_argument(
        "--run-scripts",
        action="store_true",
        help="If set, run make_figures.py and diagnostics.py for each spec before copying.",
    )
    return p.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_latest_id(runs_root: Path, spec: str) -> str:
    latest = runs_root / spec / "LATEST"
    if not latest.exists():
        raise SystemExit(f"Missing {latest}")
    return latest.read_text(encoding="utf-8").strip()


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def md_escape(code: str) -> str:
    return code.replace("`", "\\`")


def build_spec_fragment(
    *,
    spec: str,
    run_id: str,
    run_dir: Path,
    assets_dir: Path,
    gen_dir: Path,
) -> None:
    """
    Write `appendix/_generated/<spec>.md` and copy key plots into `appendix/assets/<spec>/`.
    """
    spec_assets = assets_dir / spec
    spec_assets.mkdir(parents=True, exist_ok=True)

    # Plots we want in the appendix.
    plot_sources = {
        "horizon_p50_typical_vs_marginal.png": run_dir / "figures" / "horizon_p50_typical_vs_marginal.png",
        "horizon_p80_typical_vs_marginal.png": run_dir / "figures" / "horizon_p80_typical_vs_marginal.png",
        "theta_vs_date.png": run_dir / "figures" / "theta_vs_date.png",
        "doubling_time_posterior.png": run_dir / "figures" / "doubling_time_posterior.png",
        "calibration.png": run_dir / "diagnostics" / "calibration.png",
        "ppc_time_bins.png": run_dir / "diagnostics" / "ppc_time_bins.png",
        "difficulty_residuals.png": run_dir / "diagnostics" / "difficulty_residuals.png",
    }
    for name, src in plot_sources.items():
        copy_if_exists(src, spec_assets / name)

    # Small CSVs (useful for linking / sanity).
    csv_sources = {
        "loo.csv": run_dir / "diagnostics" / "loo.csv",
        "calibration_metrics.csv": run_dir / "diagnostics" / "calibration_metrics.csv",
        "ppc_time_bins.csv": run_dir / "diagnostics" / "ppc_time_bins.csv",
    }
    for name, src in csv_sources.items():
        copy_if_exists(src, spec_assets / name)

    lines: list[str] = []
    lines.append(f"## `{md_escape(spec)}`")
    lines.append("")
    lines.append(f"**Run:** `outputs/runs/{md_escape(spec)}/{md_escape(run_id)}/`")
    lines.append("")
    for title, fname in [
        ("Horizon (p50, typical vs marginal)", "horizon_p50_typical_vs_marginal.png"),
        ("Horizon (p80, typical vs marginal)", "horizon_p80_typical_vs_marginal.png"),
        ("Ability vs release date", "theta_vs_date.png"),
        ("Doubling time posterior", "doubling_time_posterior.png"),
        ("Calibration (marginal)", "calibration.png"),
        ("PPC by task length (pooled)", "ppc_time_bins.png"),
        ("Difficulty residuals", "difficulty_residuals.png"),
    ]:
        path = spec_assets / fname
        if path.exists():
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![](assets/{spec}/{fname})")
            lines.append("")

    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / f"{spec}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_summary_fragment(*, specs: list[str], runs_root: Path, appendix_dir: Path) -> None:
    rows = []
    for spec in specs:
        run_id = run_latest_id(runs_root, spec)
        diag_dir = runs_root / spec / run_id / "diagnostics"
        loo_path = diag_dir / "loo.csv"
        cal_path = diag_dir / "calibration_metrics.csv"
        if not loo_path.exists() or not cal_path.exists():
            continue
        loo = pd.read_csv(loo_path).iloc[0].to_dict()
        cal = pd.read_csv(cal_path).iloc[0].to_dict()
        rows.append(
            {
                "spec": spec,
                "run_id": run_id,
                "elpd_loo": float(loo["elpd_loo"]),
                "p_loo": float(loo["p_loo"]),
                "loo_se": float(loo["se"]),
                "brier_weighted": float(cal["brier_weighted"]),
                "logscore_per_trial": float(cal["logscore_per_trial"]),
            }
        )

    df = pd.DataFrame(rows).sort_values("spec")
    gen_dir = appendix_dir / "_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        (gen_dir / "summary.md").write_text("_No diagnostics found under `outputs/`._\n", encoding="utf-8")
        return

    md = []
    md.append("## Summary")
    md.append("")
    md.append("This table is built from each run’s `diagnostics/loo.csv` and `diagnostics/calibration_metrics.csv`.")
    md.append("")
    md.append(df.to_markdown(index=False))
    md.append("")
    (gen_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = repo_root()
    runs_root = root / "outputs" / "runs"
    appendix_dir = root / "appendix"
    assets_dir = appendix_dir / "assets"
    gen_dir = appendix_dir / "_generated"

    specs = [s.strip() for s in args.specs.split(",") if s.strip()]

    if args.run_scripts:
        for spec in specs:
            subprocess.check_call(["python", "scripts/make_figures.py", "--spec", spec], cwd=root)
            subprocess.check_call(["python", "scripts/diagnostics.py", "--spec", spec], cwd=root)

    build_summary_fragment(specs=specs, runs_root=runs_root, appendix_dir=appendix_dir)

    for spec in specs:
        run_id = run_latest_id(runs_root, spec)
        run_dir = runs_root / spec / run_id
        build_spec_fragment(spec=spec, run_id=run_id, run_dir=run_dir, assets_dir=assets_dir, gen_dir=gen_dir)

    print(f"wrote: {gen_dir / 'summary.md'}")
    for spec in specs:
        print(f"wrote: {gen_dir / f'{spec}.md'}")
    print(f"copied assets into: {assets_dir}")


if __name__ == "__main__":
    main()

