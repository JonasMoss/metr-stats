# CLAUDE.md — metr-stats

## What this project is

Bayesian Item Response Theory (IRT) analysis of the METR benchmark data (Kwa et al. 2025,
"Measuring AI Ability to Complete Long Tasks"). METR's headline claim: the 50% task-completion
time horizon doubles every ~7 months. We reanalyse their data using a unified Bayesian
hierarchical model instead of their two-stage approach (per-model logistic → OLS trend).

**Goal:** a LessWrong blog post showing that (a) standard 2PL IRT fits the METR data well
with no ad hoc extensions, (b) multiple plausible trajectory shapes are nearly
indistinguishable on current data but produce wildly different forecasts, and (c) honest
credible intervals are much wider than METR's confidence intervals once you account for
structural (model-choice) uncertainty.

## Repo layout

```
data_raw/           Raw METR data (runs.jsonl, benchmark YAML) + build_data.py
data/               Processed artifacts: runs.pkl, irt_counts_*.csv
stan/               Production Stan models (2PL variants with trend)
experiments/        Experimental/legacy Stan models (singularity, baseline binomial)
scripts/            Python entrypoints:
  fit_time_model.py   Compile & sample Stan model
  make_figures.py     Horizon plots, theta-vs-date, doubling time
  diagnostics.py      LOO, calibration, PPC, residuals
  compare_runs.py     Aggregate LOO/Brier across specs
  one_month_horizon.py  Solve for when horizon hits 1 month (or other target)
  compare_horizon_plot.py  Overlay horizon curves from multiple specs
  build_appendix.py   Prep Quarto appendix assets
blog/               Blog post generated fragments
appendix/           Quarto diagnostics appendix
outputs/            All fit artifacts, figures, diagnostics (gitignored)
metr-stats.md       Main writeup / project notes
metr-stats.qmd      Quarto wrapper that renders the blog post HTML
Justfile            Task runner (~40 commands)
```

## Key commands

```bash
# Data prep
just build                          # or: python data_raw/build_data.py --show --write-csv --write-yaml-json

# Fit models (each writes to outputs/runs/<spec>/<run_id>/)
just fit-trend-linear               # Main: linear θ(date)
just fit-trend-quadratic            # Quadratic θ(date)
just fit-trend-xpow                 # Power-law θ(date) with learned exponent
just fit-trend-t50-logistic         # Saturating horizon (logistic ceiling on t50)
just fit-trend-sqrt                 # √(date) sublinear
just fit-trend-log1p                # log(1+date) sublinear
just fit-trend-singularity          # Finite-time singularity (experimental)
just fit                            # Baseline (no trend, free θ per model)

# Figures & diagnostics
just figs-blog-linear               # Figures for blog with point labels
just diag-linear                    # LOO + calibration + PPC + residuals
just compare-linear-quadratic       # Comparison table

# Blog rendering
just blog-prep                      # Generate one_month_horizon.md table
just blog-render                    # Quarto render → metr-stats.html
just blog                           # Both
```

## Stan model structure

All models are **2PL IRT** (two-parameter logistic):

```
P(success | model i, task j) = logit⁻¹( a_j * (θ_i - b_j) )
```

- `θ_i`: model ability (latent). Anchored: GPT-4 = -1, Claude 3.7 Sonnet = +1.
- `b_j`: task difficulty. Mean function: `b_j ~ N(α + κ·log(t_j), σ_b)` where t_j = human time.
- `a_j`: task discrimination. `log(a_j) ~ N(μ_a, σ_a)`, bounded [-2, 2].
- Likelihood: binomial (aggregated by model×task, not individual attempts).

**Trend models** add a prior on θ as a function of release date:
- `theta_trend.stan`: `θ ~ N(γ₀ + X·γ, σ_θ)` — linear or quadratic via design matrix
- `theta_trend_xpow.stan`: `θ ~ N(γ₀ + γ₁·z^a, σ_θ)` — power-law, exponent learned
- `theta_trend_t50_logistic.stan`: saturating logistic on horizon (t50), mapped back to θ
- Singularity variants: `θ ~ γ₀ + γ₁·x + c/(t*-x)^α` (experimental)

Models without dates (e.g. "human") get a weak `N(0, 1.5)` fallback prior.

## Fitting conventions

- 4 chains, 1000 warmup, 1000 sampling (default). adapt_delta=0.95, max_treedepth=12.
- Each fit goes to `outputs/runs/<spec>/<run_id>/fit/` with chain CSVs, summary, meta.json.
- `LATEST` symlink tracks the most recent run per spec.
- meta.json stores scalar hyperparameters, date centering, model names, task lists.

## Key results so far

- **LOO**: Linear elpd = -2192, quadratic = -2200 (linear wins by ~8, ~1 SE).
- **Calibration**: Brier ≈ 0.066, logscore/trial ≈ -0.31 (both models similar).
- **1-month horizon crossing** (from one_month_horizon.py):
  - Linear: 100% by 2028-06-24 (CrI 2027-11 to 2029-03)
  - Quadratic: 91% by 2027-09 (CrI 2026-11 to 2029-07)
  - xpow: 100% by 2027-10 (CrI 2027-02 to 2029-01)
  - t50_logistic (saturating): 0% — never reaches 1 month
- **Singularity models**: posterior dominated by prior; data uninformative about singularity.
- **Task difficulty**: κ ≈ 0.93 (log-length is strong predictor), σ_b ≈ 1.44 (large residual).

## Dependencies

Python 3.9+, cmdstanpy, pandas, numpy, matplotlib, pyyaml, tabulate, pyarrow.
Quarto (for rendering .qmd → .html). just (task runner, optional).

## Coding conventions

- Scripts use argparse with sensible defaults; most work out of the box with `just`.
- Paths are relative to repo root.
- Output isolation: each spec/run combo gets its own directory under outputs/runs/.
- Stan models live in stan/ (production) or experiments/ (exploratory).
- Generated blog/appendix fragments go in `blog/_generated/` and `appendix/_generated/`.

## Current status & next steps

- Core pipeline works: build → fit → figures → diagnostics → blog render.
- Need: 1-month and 125-year horizon tables with CIs across all fitted trend models.
- Need: finalize LessWrong blog post draft.
- Need: review Stan priors and model specifications for any issues.
- Optional: model-averaged forecasts, expected value of information analysis.
