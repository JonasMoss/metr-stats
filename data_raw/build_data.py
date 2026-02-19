#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml


def merge_runs(
    v11_path: Path, v10_path: Path, show: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge METR runs: all of v1.1 plus v1.0 models that have no v1.1 equivalent.

    v1.1 uses the Inspect scaffold (model names end in ``_inspect``).
    v1.0 uses modular-public/flock scaffolds (no ``_inspect`` suffix).
    For models present in both, we keep v1.1.  For models only in v1.0
    (DeepSeek, Grok, Qwen, etc.) we pull those runs in.

    Returns (merged_df, v11_df).
    """
    v11 = pd.read_json(v11_path, lines=True)
    v10 = pd.read_json(v10_path, lines=True)

    # Build set of base model names covered by v1.1
    v11_models = set(v11["model"].dropna().astype(str))
    v11_base = {m.replace("_inspect", "") for m in v11_models}

    def has_v11_equivalent(m: str) -> bool:
        return m in v11_models or m in v11_base or (m + "_inspect") in v11_models

    v10_has_model = v10["model"].notna() & (v10["model"].astype(str) != "None")
    v10_only_mask = v10["model"].apply(lambda m: not has_v11_equivalent(str(m)))
    v10_extra = v10.loc[v10_has_model & v10_only_mask]

    merged = pd.concat([v11, v10_extra], ignore_index=True)

    if show:
        v10_only_models = sorted(v10_extra["model"].dropna().unique())
        print(f"merge: {len(v11):,} v1.1 rows + {len(v10_extra):,} v1.0-only rows "
              f"= {len(merged):,} total")
        print(f"  v1.0-only models ({len(v10_only_models)}): {v10_only_models}")

    return merged, v11


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_raw = here.parent
    default_out = default_raw.parent / "data"
    parser = argparse.ArgumentParser(description="Build easy-to-load artifacts from data_raw/*.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw,
        help="Directory containing raw inputs (default: this script's directory).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Directory to write processed outputs (default: repo_root/data).",
    )
    parser.add_argument(
        "--runs-in",
        type=Path,
        default=None,
        help="Override input runs.jsonl path (default: merge runsv1.1 + runsv1.0).",
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

    yaml_in = args.yaml_in or (raw_dir / "benchmark_results_1_1.yaml")
    runs_out = args.runs_out or (out_dir / "runs.pkl")
    yaml_out = args.yaml_out or (out_dir / "benchmark_results_1_1.pkl")
    runs_csv_out = out_dir / "runs.csv"
    counts_task_out = out_dir / "irt_counts_task_id.csv"
    counts_family_out = out_dir / "irt_counts_task_family.csv"
    counts_task_v11_out = out_dir / "irt_counts_task_id_v11.csv"
    yaml_json_out = out_dir / "benchmark_results_1_1.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    v11_df = None
    if args.runs_in:
        runs_df = pd.read_json(args.runs_in, lines=True)
    else:
        runs_df, v11_df = merge_runs(
            raw_dir / "runsv1.1.jsonl",
            raw_dir / "runsv1.0.jsonl",
            show=args.show,
        )
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

        if v11_df is not None:
            counts_v11 = (
                v11_df.dropna(subset=["model", "task_id", "score_binarized"])
                .groupby(["model", "task_id"], as_index=False)["score_binarized"]
                .agg(n="count", s="sum")
            )
            counts_v11["f"] = counts_v11["n"] - counts_v11["s"]
            counts_v11.to_csv(counts_task_v11_out, index=False)

    with open(yaml_in, "r", encoding="utf-8") as f:
        yaml_obj = yaml.safe_load(f)
    with open(yaml_out, "wb") as f:
        pickle.dump(yaml_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    if args.write_yaml_json:
        with open(yaml_json_out, "w", encoding="utf-8") as f:
            json.dump(yaml_obj, f, indent=2, sort_keys=True, default=str)

    # Build release_dates.json: model_id -> date string, from release_dates.yaml + alias column.
    release_yaml_path = raw_dir / "release_dates.yaml"
    release_json_out = out_dir / "release_dates.json"
    if release_yaml_path.exists():
        with open(release_yaml_path, "r", encoding="utf-8") as f:
            rd_obj = yaml.safe_load(f)
        display_dates = rd_obj.get("date", {})
        # Build alias -> model_id map from runs
        alias_to_id: dict[str, str] = {}
        for _, row in runs_df[["model", "alias"]].drop_duplicates().iterrows():
            if pd.notna(row["alias"]) and pd.notna(row["model"]):
                alias_to_id[str(row["alias"])] = str(row["model"])
        release_map: dict[str, str] = {}
        for display_name, date_val in display_dates.items():
            model_id = alias_to_id.get(display_name)
            if model_id is not None:
                release_map[model_id] = str(date_val)
        with open(release_json_out, "w", encoding="utf-8") as f:
            json.dump(release_map, f, indent=2, sort_keys=True)
        if args.show:
            print(f"release dates: {release_yaml_path} -> {release_json_out} ({len(release_map)} models)")

    if args.show:
        src = args.runs_in or "runsv1.1 + runsv1.0 (merged)"
        print(f"runs: {src} -> {runs_out} (rows={len(runs_df):,} cols={len(runs_df.columns)})")
        print(f"yaml: {yaml_in} -> {yaml_out} (top_keys={list(yaml_obj.keys())[:10]})")
        if args.write_csv:
            print(f"csv: {runs_csv_out}")
            print(f"irt counts: {counts_task_out}")
            print(f"irt counts: {counts_family_out}")
            if v11_df is not None:
                print(f"irt counts (v1.1): {counts_task_v11_out}")
        if args.write_yaml_json:
            print(f"yaml json: {yaml_json_out}")


if __name__ == "__main__":
    main()
