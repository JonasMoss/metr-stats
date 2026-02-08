#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare multiple runs using diagnostics outputs (LOO + calibration metrics). "
            "Input can be --specs (uses each spec's LATEST) or explicit --fitdirs."
        )
    )
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--specs", type=str, default=None, help="Comma-separated specs to compare (uses LATEST).")
    p.add_argument("--fitdirs", type=str, default=None, help="Comma-separated fit directories.")
    p.add_argument("--outdir", type=Path, default=Path("outputs/compare"), help="Output directory (default: outputs/compare).")
    return p.parse_args()


def resolve_fitdirs(runs_root: Path, specs: list[str]) -> list[Path]:
    out: list[Path] = []
    for spec in specs:
        run_id = (runs_root / spec / "LATEST").read_text(encoding="utf-8").strip()
        out.append(runs_root / spec / run_id / "fit")
    return out


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fitdirs: list[Path] = []
    if args.fitdirs:
        fitdirs = [Path(s.strip()) for s in args.fitdirs.split(",") if s.strip()]
    elif args.specs:
        specs = [s.strip() for s in args.specs.split(",") if s.strip()]
        fitdirs = resolve_fitdirs(args.runs_root, specs)
    else:
        raise SystemExit("Provide either --specs or --fitdirs.")

    rows = []
    for fitdir in fitdirs:
        diagdir = fitdir.parent / "diagnostics"
        loo = pd.read_csv(diagdir / "loo.csv").iloc[0].to_dict()
        cal = pd.read_csv(diagdir / "calibration_metrics.csv").iloc[0].to_dict()
        import json

        meta = json.loads((diagdir / "meta.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "fitdir": str(fitdir),
                "spec": cal.get("spec", ""),
                "theta_trend": cal.get("theta_trend", ""),
                "elpd_loo": loo.get("elpd_loo"),
                "p_loo": loo.get("p_loo"),
                "loo_se": loo.get("se"),
                "brier_weighted": cal.get("brier_weighted"),
                "logscore_per_trial": cal.get("logscore_per_trial"),
                "draws_used": meta.get("draws_used"),
                "mc": meta.get("mc"),
            }
        )

    df = pd.DataFrame(rows).sort_values(["theta_trend", "spec"])
    df.to_csv(args.outdir / "compare.csv", index=False)
    print(f"wrote: {args.outdir / 'compare.csv'}")


if __name__ == "__main__":
    main()
