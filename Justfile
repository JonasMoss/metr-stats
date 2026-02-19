set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Pre-Sonnet-3.5 models to drop for the all-data (v1.0+v1.1) pipeline.
DROP_MODELS := "gpt2,davinci_002,gpt_3_5_turbo_instruct,gpt_4,gpt_4_1106_inspect,claude_3_opus_inspect,gpt_4_turbo_inspect,gpt_4o_inspect"
V11_SUFFIX := "__v11"
V11_MODELS := "o1_inspect,claude_3_7_sonnet_inspect,o3_inspect,claude_opus_4_5_inspect,gpt_5_2"
V11_SPECS := "time_irt__theta_linear__v11,time_irt__theta_quadratic_pos__v11,time_irt__theta_xpow__v11,time_irt__theta_theta_logistic__v11"

# Default: fit + fit-v11 + blog.
default: fit fit-v11 blog

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
  python scripts/diagnostics.py --spec time_irt__theta_quadratic_pos
  python scripts/diagnostics.py --spec time_irt__theta_xpow
  python scripts/diagnostics.py --spec time_irt__theta_theta_logistic
  python scripts/one_month_horizon.py --skip-missing --max-year 2200
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon.csv --out-md blog/_generated/125y_horizon.md
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind typical --out-csv blog/_generated/one_month_horizon_p80_typical.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind marginal --out-csv blog/_generated/one_month_horizon_p80_marginal.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind typical --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon_p80_typical.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind marginal --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon_p80_marginal.csv

# Re-generate figures, diagnostics, and blog-prep CSVs from existing fits (no refitting).
postfit:
  python scripts/make_figures.py --spec time_irt__theta_linear --label-preset blog
  python scripts/make_figures.py --spec time_irt__theta_quadratic_pos --label-preset blog
  python scripts/make_figures.py --spec time_irt__theta_xpow --label-preset blog
  python scripts/make_figures.py --spec time_irt__theta_theta_logistic --label-preset blog
  python scripts/diagnostics.py --spec time_irt__theta_linear
  python scripts/diagnostics.py --spec time_irt__theta_quadratic_pos
  python scripts/diagnostics.py --spec time_irt__theta_xpow
  python scripts/diagnostics.py --spec time_irt__theta_theta_logistic
  python scripts/one_month_horizon.py --skip-missing --max-year 2200
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon.csv --out-md blog/_generated/125y_horizon.md
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind typical --out-csv blog/_generated/one_month_horizon_p80_typical.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind marginal --out-csv blog/_generated/one_month_horizon_p80_marginal.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind typical --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon_p80_typical.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --p 0.8 --kind marginal --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon_p80_marginal.csv

# Fit all v1.1 models, generate figures/diagnostics/blog CSVs.
fit-v11:
  python scripts/fit_time_model.py --theta-trend linear --counts data/irt_counts_task_id_v11.csv --spec-suffix {{V11_SUFFIX}}
  python scripts/fit_time_model.py --theta-trend quadratic_pos --counts data/irt_counts_task_id_v11.csv --spec-suffix {{V11_SUFFIX}}
  python scripts/fit_time_model.py --theta-trend xpow --counts data/irt_counts_task_id_v11.csv --spec-suffix {{V11_SUFFIX}}
  python scripts/fit_time_model.py --theta-trend theta_logistic --counts data/irt_counts_task_id_v11.csv --spec-suffix {{V11_SUFFIX}}
  python scripts/make_figures.py --spec time_irt__theta_linear__v11 --label-preset blog --models {{V11_MODELS}}
  python scripts/make_figures.py --spec time_irt__theta_quadratic_pos__v11 --label-preset blog --models {{V11_MODELS}}
  python scripts/make_figures.py --spec time_irt__theta_xpow__v11 --label-preset blog --models {{V11_MODELS}}
  python scripts/make_figures.py --spec time_irt__theta_theta_logistic__v11 --label-preset blog --models {{V11_MODELS}}
  python scripts/diagnostics.py --spec time_irt__theta_linear__v11 --counts data/irt_counts_task_id_v11.csv
  python scripts/diagnostics.py --spec time_irt__theta_quadratic_pos__v11 --counts data/irt_counts_task_id_v11.csv
  python scripts/diagnostics.py --spec time_irt__theta_xpow__v11 --counts data/irt_counts_task_id_v11.csv
  python scripts/diagnostics.py --spec time_irt__theta_theta_logistic__v11 --counts data/irt_counts_task_id_v11.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --specs {{V11_SPECS}} --out-csv blog/_generated/one_month_horizon_v11.csv --out-md blog/_generated/one_month_horizon_v11.md
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --specs {{V11_SPECS}} --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon_v11.csv --out-md blog/_generated/125y_horizon_v11.md
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --specs {{V11_SPECS}} --p 0.8 --kind typical --out-csv blog/_generated/one_month_horizon_p80_typical_v11.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --specs {{V11_SPECS}} --p 0.8 --kind marginal --out-csv blog/_generated/one_month_horizon_p80_marginal_v11.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --specs {{V11_SPECS}} --p 0.8 --kind typical --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon_p80_typical_v11.csv
  python scripts/one_month_horizon.py --skip-missing --max-year 2200 --specs {{V11_SPECS}} --p 0.8 --kind marginal --threshold-days 45656 --threshold-label "125 years" --out-csv blog/_generated/125y_horizon_p80_marginal_v11.csv

# Render the blog post with Quarto.
blog:
  mkdir -p outputs/.xdg_cache outputs/.xdg_data outputs/.deno
  XDG_CACHE_HOME=$PWD/outputs/.xdg_cache XDG_DATA_HOME=$PWD/outputs/.xdg_data DENO_DIR=$PWD/outputs/.deno quarto render metr-stats.qmd
  python scripts/clean_gfm.py metr-stats.md

# Generate IWSM 2026 paper figures and compile LaTeX.
iwsm:
  python iwsm/make_iwsm_figures.py
  cd iwsm && pdflatex Moss_marginal_typical_IRT.tex && pdflatex Moss_marginal_typical_IRT.tex
