#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stan fits using PSIS-LOO (arviz).")
    parser.add_argument(
        "--fitdirs",
        type=str,
        required=True,
        help="Comma-separated list of fit directories (each produced by a fit_*.py script).",
    )
    parser.add_argument(
        "--names",
        type=str,
        default=None,
        help="Comma-separated names corresponding to --fitdirs (default: directory basenames).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/out_loo"),
        help="Output directory (default: analysis/out_loo).",
    )
    return parser.parse_args()


def stan_chain_csvs(fitdir: Path) -> list[Path]:
    csvs: list[Path] = []
    for p in sorted(fitdir.glob("*.csv")):
        name = p.name
        if name in {"summary.csv", "theta.csv", "a.csv", "b.csv", "items.csv"}:
            continue
        if name.endswith("-stdout.txt"):
            continue
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                first = f.read(1)
            if first == "#":
                csvs.append(p)
        except OSError:
            continue
    return csvs


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # arviz tries to create ~/.cache/arviz; force cache into a writable location.
    cache_dir = args.outdir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import arviz as az
    import pandas as pd
    from cmdstanpy import from_csv

    fitdirs = [Path(s.strip()) for s in args.fitdirs.split(",") if s.strip()]
    if not fitdirs:
        raise SystemExit("--fitdirs must contain at least one directory")

    if args.names is None:
        names = [d.name for d in fitdirs]
    else:
        names = [s.strip() for s in args.names.split(",") if s.strip()]
        if len(names) != len(fitdirs):
            raise SystemExit("--names must have the same number of entries as --fitdirs")

    idatas: dict[str, object] = {}
    loos: dict[str, object] = {}
    captured_warnings: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

        for name, fitdir in zip(names, fitdirs):
            csvs = stan_chain_csvs(fitdir)
            if not csvs:
                raise SystemExit(f"No CmdStan chain CSVs found in {fitdir}")
            fit = from_csv(path=[str(p) for p in csvs])
            try:
                log_lik = fit.draws_xr(vars=["log_lik"])["log_lik"].to_numpy()  # (chain, draw, N)
            except Exception as e:
                raise SystemExit(f"{fitdir} does not contain generated quantity log_lik") from e

            # arviz.loo expects a posterior group; provide a dummy scalar posterior.
            dummy = log_lik[..., :1]
            idata = az.from_dict(posterior={"_dummy": dummy}, log_likelihood={"y": log_lik})
            idatas[name] = idata
            loos[name] = az.loo(idata, pointwise=False)

        cmp = az.compare(idatas, ic="loo")
        captured_warnings = list(w)

    cmp.to_csv(args.outdir / "compare.csv")

    loo_table = pd.DataFrame(
        [
            {
                "name": k,
                "elpd_loo": float(v["elpd_loo"]),
                "se": float(v["se"]),
                "p_loo": float(v["p_loo"]),
                "looic": float(-2.0 * float(v["elpd_loo"])),
            }
            for k, v in loos.items()
        ]
    ).sort_values("elpd_loo", ascending=False)
    loo_table.to_csv(args.outdir / "loo.csv", index=False)

    # Print an abbreviated warning summary (keep the raw logs clean).
    pareto_msgs = []
    for msg in captured_warnings:
        text = str(msg.message)
        if "Estimated shape parameter of Pareto distribution is greater than" in text:
            pareto_msgs.append(text)
    if pareto_msgs:
        print("warning:", pareto_msgs[0])

    print(loo_table.to_string(index=False))
    print(f"wrote: {args.outdir / 'loo.csv'}")
    print(f"wrote: {args.outdir / 'compare.csv'}")


if __name__ == "__main__":
    main()
