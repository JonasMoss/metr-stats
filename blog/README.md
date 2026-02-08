# Blog build

The blog post lives in `metr-stats.qmd` and includes:

- `metr-stats.md` (main text, currently authored outside Quarto)
- `blog/_generated/one_month_horizon.md` (generated table for “time to 1-month horizon”)

## Build + render

```bash
just blog
```

This writes `metr-stats.html` in the repo root.

