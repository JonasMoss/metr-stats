#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build easy-to-load artifacts from data_raw/*.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data_raw"),
        help="Directory containing raw inputs (default: data_raw).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write processed outputs (default: data).",
    )
    parser.add_argument(
        "--runs-in",
        type=Path,
        default=None,
        help="Override input runs.jsonl path (default: RAW_DIR/runs.jsonl).",
    )
    parser.add_argument(
        "--yaml-in",
        type=Path,
        default=None,
        help="Override input benchmark_results_1_1.yaml path (default: RAW_DIR/benchmark_results_1_1.yaml).",
    )
    parser.add_argument(
        "--runs-out",
        type=Path,
        default=None,
        help="Override output runs pickle path (default: OUT_DIR/runs.pkl).",
    )
    parser.add_argument(
        "--yaml-out",
        type=Path,
        default=None,
        help="Override output YAML pickle path (default: OUT_DIR/benchmark_results_1_1.pkl).",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSVs (runs.csv and IRT count tables).",
    )
    parser.add_argument(
        "--write-yaml-json",
        action="store_true",
        help="Also write a pretty JSON copy of the parsed YAML.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print a short summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir

    runs_in = args.runs_in or (raw_dir / "runs.jsonl")
    yaml_in = args.yaml_in or (raw_dir / "benchmark_results_1_1.yaml")
    runs_out = args.runs_out or (out_dir / "runs.pkl")
    yaml_out = args.yaml_out or (out_dir / "benchmark_results_1_1.pkl")
    runs_csv_out = out_dir / "runs.csv"
    counts_task_out = out_dir / "irt_counts_task_id.csv"
    counts_family_out = out_dir / "irt_counts_task_family.csv"
    yaml_json_out = out_dir / "benchmark_results_1_1.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.read_json(runs_in, lines=True)
    if "score_binarized" in runs_df.columns:
        runs_df["score_binarized"] = runs_df["score_binarized"].astype("Int8")

    with open(runs_out, "wb") as f:
        pickle.dump(runs_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.write_csv:
        runs_df.to_csv(runs_csv_out, index=False)
        counts_task = (
            runs_df.dropna(subset=["model", "task_id", "score_binarized"])
            .groupby(["model", "task_id"], as_index=False)["score_binarized"]
            .agg(n="count", s="sum")
        )
        counts_task["f"] = counts_task["n"] - counts_task["s"]
        counts_task.to_csv(counts_task_out, index=False)

        counts_family = (
            runs_df.dropna(subset=["model", "task_family", "score_binarized"])
            .groupby(["model", "task_family"], as_index=False)["score_binarized"]
            .agg(n="count", s="sum")
        )
        counts_family["f"] = counts_family["n"] - counts_family["s"]
        counts_family.to_csv(counts_family_out, index=False)

    with open(yaml_in, "r", encoding="utf-8") as f:
        yaml_obj = yaml.safe_load(f)
    with open(yaml_out, "wb") as f:
        pickle.dump(yaml_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    if args.write_yaml_json:
        with open(yaml_json_out, "w", encoding="utf-8") as f:
            json.dump(yaml_obj, f, indent=2, sort_keys=True, default=str)

    if args.show:
        print(f"runs: {runs_in} -> {runs_out} (rows={len(runs_df):,} cols={len(runs_df.columns)})")
        print(f"yaml: {yaml_in} -> {yaml_out} (top_keys={list(yaml_obj.keys())[:10]})")
        if args.write_csv:
            print(f"csv: {runs_csv_out}")
            print(f"irt counts: {counts_task_out}")
            print(f"irt counts: {counts_family_out}")
        if args.write_yaml_json:
            print(f"yaml json: {yaml_json_out}")


if __name__ == "__main__":
    main()
