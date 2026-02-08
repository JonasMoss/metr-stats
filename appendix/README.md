# Diagnostics appendix

This folder contains a Quarto document (`appendix/diagnostics.qmd`) that summarizes diagnostics for the **trend-only** time-IRT fits (linear and quadratic).

It reads figures/CSVs from `outputs/` and does **not** refit Stan during render.

## Quickstart

1) Build processed data:

`just build`

2) Run everything (figures + diagnostics + render):

`just appendix`

## Manual render (no `just`)

You can also run the steps manually:

- Build figures+diagnostics and copy assets into `appendix/`:
  - `python scripts/build_appendix.py --run-scripts`
- Render (requires Quarto):
  - `XDG_CACHE_HOME=$PWD/outputs/.xdg_cache XDG_DATA_HOME=$PWD/outputs/.xdg_data DENO_DIR=$PWD/outputs/.deno quarto render appendix/diagnostics.qmd`
