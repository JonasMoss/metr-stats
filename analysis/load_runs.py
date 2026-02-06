#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load METR runs.jsonl into a pandas DataFrame.")
    parser.add_argument(
        "--infile",
        type=Path,
        default=Path("data_raw/runs.jsonl"),
        help="Path to runs.jsonl (default: data_raw/runs.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path (.csv, .pkl, or .parquet).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print a quick summary to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = args.infile.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(args.infile)
    elif suffix == ".parquet":
        df = pd.read_parquet(args.infile)
    else:
        df = pd.read_json(args.infile, lines=True)

    df["score_binarized"] = df["score_binarized"].astype("Int8")

    if args.show:
        print(f"rows={len(df):,} cols={len(df.columns)}")
        print("models:", df["model"].nunique())
        print("tasks:", df["task_id"].nunique())
        print("task_families:", df["task_family"].nunique())
        print("score_binarized value counts:")
        print(df["score_binarized"].value_counts(dropna=False).sort_index())
        print("fatal_error_from value counts (top 10):")
        print(df["fatal_error_from"].value_counts(dropna=False).head(10))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        suffix = args.out.suffix.lower()
        if suffix == ".csv":
            df.to_csv(args.out, index=False)
        elif suffix in {".pkl", ".pickle"}:
            df.to_pickle(args.out)
        elif suffix == ".parquet":
            try:
                df.to_parquet(args.out, index=False)
            except ImportError as e:
                raise SystemExit(
                    "Parquet support requires `pyarrow` or `fastparquet`. "
                    "Either install one of those, or use `--out analysis/runs.csv` or `--out analysis/runs.pkl`."
                ) from e
        else:
            raise SystemExit(f"--out must end with .csv, .pkl, or .parquet (got {args.out})")


if __name__ == "__main__":
    main()
