# Analysis helpers

This folder contains small scripts to work with `runs.jsonl` (newline-delimited JSON).

## Load into pandas

Quick summary:

```bash
python analysis/load_runs.py --show
```

Convert to a more convenient format (ignored by git via `analysis/.gitignore`):

```bash
python analysis/load_runs.py --out analysis/runs.csv
python analysis/load_runs.py --out analysis/runs.pkl
```

Or build the “package-style” processed artifacts:

```bash
python scripts/build_data.py --show
python scripts/build_data.py --show --write-csv --write-yaml-json
```

Then you can load from `data/runs.pkl` instead of reparsing JSONL each time:

```bash
python analysis/load_runs.py --infile data/runs.pkl --show
```

## Fit a 2PL (IRT) model

This fits a basic 2-parameter logistic IRT model to the binary run outcomes.

- “People” = models (`model`)
- “Items” = tasks (`task_id`) or task families (`task_family`)
- Observations are aggregated counts per (model, item) pair: successes `s` out of `n` attempts.

Run (dropping runs that fatally error due to usage limits):

```bash
python analysis/fit_2pl.py --exclude-fatal usageLimits
python analysis/fit_2pl.py --infile data/runs.pkl --exclude-fatal usageLimits
```

Outputs:

- `analysis/out/2pl_models.csv` (model ability `theta`)
- `analysis/out/2pl_items.csv` (item discrimination `a` and difficulty `b`)
- `analysis/out/2pl_meta.json` (optimizer status)

## Optional: base-R 2PL fit

If you prefer R, `analysis/fit_2pl.R` fits the same binomial 2PL likelihood from an aggregated count table (no extra R packages needed):

```bash
python scripts/build_data.py --write-csv
Rscript analysis/fit_2pl.R --counts data/irt_counts_task_id.csv
```

## Stan 2PL (cmdstanpy)

If you have `cmdstanpy` + CmdStan installed, `analysis/fit_2pl_stan.py` fits the 2PL model in Stan (binomial likelihood on the aggregated counts table):

```bash
python scripts/build_data.py --write-csv
python analysis/fit_2pl_stan.py --counts data/irt_counts_task_id.csv
```

Outputs go to `analysis/out_stan/` by default. By default, the Stan script anchors `gpt_4` and `claude_3_7_sonnet_inspect` at `theta=-1` and `theta=+1` when both are present in the counts table; otherwise it falls back to the lowest/highest pass-rate models.

To pick anchors, you can list model pass rates in the counts table:

```bash
python analysis/fit_2pl_stan.py --counts data/irt_counts_task_id.csv --list-models
```

Then set anchors explicitly, e.g.:

```bash
python analysis/fit_2pl_stan.py --anchor-low gpt_4 --anchor-high claude_3_7_sonnet_inspect
```

## Stan 2PL with log-linear difficulty vs time

`analysis/fit_2pl_time_stan.py` fits a second Stan model where task difficulty follows a log-linear function of task length, with task-level random effects:

- \(b_j \sim \mathcal N(\alpha + \kappa(\log t_j - \overline{\log t}), \sigma_b)\), with \(\kappa>0\)
- \(\log a_j \sim \mathcal N(\mu_{\log a}, \sigma_{\log a})\) (no dependence on \(t\))

Run:

```bash
python scripts/build_data.py --write-csv
python analysis/fit_2pl_time_stan.py --counts data/irt_counts_task_id.csv
python analysis/postprocess_time_model.py --fitdir analysis/out_stan_time
```

This writes posterior p-curves and a METR-style horizon plot into `analysis/out_time_model/`.

To add a log-length dependence for discrimination as well, fit the alternate variant:

```bash
python analysis/fit_2pl_time_stan.py --variant b_logt__loga_logt --outdir analysis/out_stan_time_loga_logt
python analysis/postprocess_time_model.py --fitdir analysis/out_stan_time_loga_logt --outdir analysis/out_time_model_loga_logt
```

LOO comparison (requires `arviz` import to work in this environment; the script sets `XDG_CACHE_HOME` automatically):

```bash
python analysis/compare_loo.py --fitdirs analysis/out_stan_time,analysis/out_stan_time_loga_logt
```

Residual diagnostic for the log-linear difficulty fit (plots \(\hat u_j = \hat b_j - (\hat\alpha + \hat\kappa x_j)\) vs task length):

```bash
python analysis/residuals_time_model.py --fitdir analysis/out_stan_time_long
```

Calibration diagnostics (reliability curves + Brier/log scores comparing full plug-in vs time-only curves):

```bash
python analysis/calibration_diagnostics.py --fitdir analysis/out_stan_time_long
```

## Release-date trend + doubling time (date → θ → horizon)

`analysis/trend_horizons_from_release.py` fits a simple linear trend in ability (θ-space) vs release date,
then maps that trend through the time-IRT model to get a smoother horizon-vs-date curve and an implied
doubling time (from the slope of `log(horizon)` vs date).

It writes:

- `analysis/out_trend/theta_vs_date.png` (θ vs release date with trend band)
- `analysis/out_trend/horizon_p50.png`, `analysis/out_trend/horizon_p80.png` (horizon-vs-date plots)
- `analysis/out_trend/doubling_time.csv`

Run:

```bash
python analysis/trend_horizons_from_release.py --fitdir analysis/out_stan_time_long
```

Outputs also include:

- `analysis/out_trend/horizons_multi.png` (multiple p levels on one plot if you pass multiple `--p`)
- `analysis/out_trend/doubling_time_posterior.png` and `analysis/out_trend/doubling_time_draws.csv`

## Horizon regression (simple METR-style plot)

After fitting the 2PL, you can “calibrate” item difficulty to task length by regressing estimated `b_j` on `log(t_j)` (where `t_j` comes from `human_minutes` in `runs.jsonl`), then compute each model’s 50% horizon `t50` and plot it vs model release date:

```bash
python analysis/horizon_regression.py
```

By default this expects:

- `analysis/out_stan/theta.csv` and `analysis/out_stan/b.csv` from `analysis/fit_2pl_stan.py`
- `data/runs.pkl` from `scripts/build_data.py`
- `data_raw/benchmark_results_1_1.yaml` for release dates

## Plug-in p(success) vs task length

To visualize the raw implied probabilities \(p_{ij}=\text{logit}^{-1}(a_j(\theta_i-b_j))\) against task length (no regression/calibration), run:

```bash
python analysis/prob_vs_length.py
```

This expects a task-level Stan fit (items are `task_id`) so `analysis/out_stan/items.csv` contains both `mean_a` and `mean_b` for each task.

## Note

The isotonic / PAVA “monotone horizons” experiment was deleted in the main refactor; this folder is for legacy reference only.
