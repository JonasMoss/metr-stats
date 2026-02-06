# metr-stats

## Data layout

- `data_raw/`: raw inputs (kept as-downloaded)
- `data/`: derived, easy-to-load artifacts built from `data_raw/`
- `analysis/`: one-off analysis scripts (e.g. 2PL / IRT fits)

## Quick start

Build processed artifacts:

```bash
python scripts/build_data.py --show
```

Fit a basic 2PL IRT model on run outcomes:

```bash
python analysis/fit_2pl.py --exclude-fatal usageLimits
```

Stan version (cmdstanpy):

```bash
python scripts/build_data.py --write-csv
python analysis/fit_2pl_stan.py --counts data/irt_counts_task_id.csv
```
