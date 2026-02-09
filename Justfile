# Minimal workflow for the blog-post pipeline.

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

build:
  python data_raw/build_data.py --show --write-csv --write-yaml-json

# Full fit (can take a while).
fit:
  python scripts/fit_time_model.py

# Joint ability trend fits (same likelihood, adds θ(d) prior).
fit-trend-linear:
  python scripts/fit_time_model.py --theta-trend linear

fit-trend-quadratic:
  python scripts/fit_time_model.py --theta-trend quadratic

fit-trend-quadratic-pos:
  python scripts/fit_time_model.py --theta-trend quadratic_pos

fit-trend-sqrt:
  python scripts/fit_time_model.py --theta-trend sqrt

fit-trend-log1p:
  python scripts/fit_time_model.py --theta-trend log1p

fit-trend-xpow:
  python scripts/fit_time_model.py --theta-trend xpow

fit-trend-t50-logistic:
  python scripts/fit_time_model.py --theta-trend t50_logistic

fit-trend-singularity:
  python scripts/fit_time_model.py --theta-trend singularity

fit-trend-singularity-nolinear:
  python scripts/fit_time_model.py --theta-trend singularity_nolinear

# Small fit for sanity checking (fast).
fit-short:
  python scripts/fit_time_model.py --chains 2 --iter-warmup 250 --iter-sampling 250 --refresh 50

figs:
  python scripts/make_figures.py

figs-linear:
  python scripts/make_figures.py --spec time_irt__theta_linear

figs-quadratic:
  python scripts/make_figures.py --spec time_irt__theta_quadratic

# Blog-friendly versions with a few point labels on horizon plots.
figs-blog-linear:
  python scripts/make_figures.py --spec time_irt__theta_linear --label-preset blog

figs-blog-quadratic:
  python scripts/make_figures.py --spec time_irt__theta_quadratic --label-preset blog

figs-blog-sqrt:
  python scripts/make_figures.py --spec time_irt__theta_sqrt --label-preset blog

figs-blog-log1p:
  python scripts/make_figures.py --spec time_irt__theta_log1p --label-preset blog

figs-blog-xpow:
  python scripts/make_figures.py --spec time_irt__theta_xpow --label-preset blog

figs-blog-quadratic-pos:
  python scripts/make_figures.py --spec time_irt__theta_quadratic_pos --label-preset blog

figs-blog-t50-logistic:
  python scripts/make_figures.py --spec time_irt__theta_t50_logistic --label-preset blog

figs-singularity:
  python scripts/make_figures.py --spec time_irt__theta_singularity

figs-singularity-nolinear:
  python scripts/make_figures.py --spec time_irt__theta_singularity_nolinear

diag-linear:
  python scripts/diagnostics.py --spec time_irt__theta_linear

diag-quadratic:
  python scripts/diagnostics.py --spec time_irt__theta_quadratic

diag-quadratic-pos:
  python scripts/diagnostics.py --spec time_irt__theta_quadratic_pos

diag-xpow:
  python scripts/diagnostics.py --spec time_irt__theta_xpow

diag-t50-logistic:
  python scripts/diagnostics.py --spec time_irt__theta_t50_logistic

diag-singularity:
  python scripts/diagnostics.py --spec time_irt__theta_singularity

diag-singularity-nolinear:
  python scripts/diagnostics.py --spec time_irt__theta_singularity_nolinear

# Regenerate all blog-relevant figure CSVs (including theta_trend_grid.csv).
figs-all-blog: figs-blog-linear figs-blog-quadratic figs-blog-xpow figs-blog-t50-logistic

compare-linear-quadratic:
  python scripts/compare_runs.py --specs time_irt__theta_linear,time_irt__theta_quadratic --outdir outputs/compare_trends

compare-horizon-p50:
  python scripts/compare_horizon_plot.py --p 0.5 --kind typical --outdir outputs/compare_trends

blog-prep:
  python scripts/one_month_horizon.py --skip-missing --max-year 2200

blog-prep-125y:
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon.csv --out-md blog/_generated/125y_horizon.md

blog-render:
  mkdir -p outputs/.xdg_cache outputs/.xdg_data outputs/.deno
  XDG_CACHE_HOME=$PWD/outputs/.xdg_cache XDG_DATA_HOME=$PWD/outputs/.xdg_data DENO_DIR=$PWD/outputs/.deno quarto render metr-stats.qmd

blog: blog-prep blog-render

appendix-prep:
  python scripts/build_appendix.py --run-scripts

appendix-render:
  mkdir -p outputs/.xdg_cache outputs/.xdg_data outputs/.deno
  XDG_CACHE_HOME=$PWD/outputs/.xdg_cache XDG_DATA_HOME=$PWD/outputs/.xdg_data DENO_DIR=$PWD/outputs/.deno quarto render appendix/diagnostics.qmd

appendix: appendix-prep appendix-render

all: build fit figs

clean:
  rm -rf outputs
