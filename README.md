# metr-stats

## Layout (core)

- `data_raw/`: raw inputs (kept as-downloaded) + the build script
- `data/`: derived, easy-to-load artifacts built from `data_raw/`
- `stan/`: the core Stan model(s) used in the writeup
- `scripts/`: thin entrypoints (fit + figures)
- `outputs/`: fit artifacts + plots (ignored by git)

## Quick start

Build processed artifacts (runs pickle + counts table):

```bash
python data_raw/build_data.py --show --write-csv --write-yaml-json
```

Fit the core time-IRT model (2PL with log-linear difficulty vs log task length; constant discrimination):

```bash
python scripts/fit_time_model.py
```

Optional: fit a joint ability trend \(\theta(d)\) inside Stan (linear or quadratic in release date):

```bash
python scripts/fit_time_model.py --theta-trend linear
python scripts/fit_time_model.py --theta-trend quadratic
```

Make the core figures (writes into `outputs/runs/<run_id>/figures/`):

```bash
python scripts/make_figures.py
```

Diagnostics (LOO + marginal calibration/Brier + residuals) for a spec:

```bash
python scripts/diagnostics.py --spec time_irt__theta_linear
python scripts/diagnostics.py --spec time_irt__theta_quadratic
```

Compare specs (uses each spec's `LATEST`):

```bash
python scripts/compare_runs.py --specs time_irt__theta_linear,time_irt__theta_quadratic --outdir outputs/compare_trends
```

## Appendix (Quarto)

Diagnostics appendix (renders from `outputs/` without refitting):

```bash
just appendix-prep
just appendix-render
```

Source: `appendix/diagnostics.qmd`.

## Using `just`

If you have `just` installed:

```bash
just build
just fit
just figs
```

Or end-to-end:

```bash
just all
```

## Experiments

Older one-off scripts and alternative Stan models live under `experiments/` (not part of the main pipeline).
