set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Default: fit + blog.
default: fit blog

# Build data, fit all four trend models, generate figures, diagnostics, and blog-prep CSVs.
fit:
  python data_raw/build_data.py --show --write-csv --write-yaml-json
  python scripts/fit_time_model.py --theta-trend linear
  python scripts/fit_time_model.py --theta-trend quadratic_pos
  python scripts/fit_time_model.py --theta-trend xpow
  python scripts/fit_time_model.py --theta-trend theta_logistic
  python scripts/make_figures.py --spec time_irt__theta_linear --label-preset blog
  python scripts/make_figures.py --spec time_irt__theta_quadratic_pos --label-preset blog
  python scripts/make_figures.py --spec time_irt__theta_xpow --label-preset blog
  python scripts/make_figures.py --spec time_irt__theta_theta_logistic --label-preset blog
  python scripts/diagnostics.py --spec time_irt__theta_linear
  python scripts/one_month_horizon.py --skip-missing --max-year 2200
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon.csv --out-md blog/_generated/125y_horizon.md
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind typical --out-csv blog/_generated/one_month_horizon_p80_typical.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind marginal --out-csv blog/_generated/one_month_horizon_p80_marginal.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind typical --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon_p80_typical.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind marginal --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon_p80_marginal.csv

# Render the blog post with Quarto.
blog:
  mkdir -p outputs/.xdg_cache outputs/.xdg_data outputs/.deno
  XDG_CACHE_HOME=$PWD/outputs/.xdg_cache XDG_DATA_HOME=$PWD/outputs/.xdg_data DENO_DIR=$PWD/outputs/.deno quarto render metr-stats.qmd
