# Blog Post Outline: "The METR horizon doubling claim is underspecified — a Bayesian IRT reanalysis"

(Working title. Alternatives: "How uncertain are AI capability trajectories, really?",
"Psychometrics for AI: what standard IRT tells us about the METR benchmarks")

---

## 1. Hook / TL;DR (2-3 paragraphs)

METR says AI task-completion horizons are doubling every ~7 months. This is becoming
load-bearing for governance — it's cited in policy briefs, safety cases, lab timelines.
But their statistical approach (two-stage estimation) bakes in a specific functional form
and underestimates uncertainty.

We refit the same data using a unified Bayesian hierarchical model — specifically, standard
Item Response Theory (IRT), the same framework used in psychometrics for 50+ years. No custom
methodology needed. We then ask: given these data, can we distinguish between linear,
quadratic, decelerating, power-law, and saturating capability trajectories?

**Punchline:** We can't really distinguish them. LOO differences are small. But the forecasts
differ enormously — from "1-month horizon by mid-2028" to "never reaches 1 month." The
structural uncertainty (which trajectory shape?) dwarfs the parameter uncertainty within any
single model. METR's CIs are honest *conditional on the linear trend being correct*, but the
unconditional uncertainty is much wider.

---

## 2. Background: What METR did (keep short, most readers know this)

- Data: ~16,600 binary success/fail attempts, 15 frontier models, 161 tasks
- Each task has a human-time difficulty label (from human baselines)
- Stage 1: per-model logistic regression of P(success) on log(task-time) → extract t50 horizon
- Stage 2: OLS of log(t50) on release date → doubling time ≈ 7 months (CI: 3.5–23)
- Bootstrap for CIs (hierarchical: task families → tasks → runs)

**What's missing:**
- Two-stage approach discards Stage-1 uncertainty (unless you bootstrap *everything*)
- Only one trajectory shape tested (linear in log-horizon)
- Tasks at the same human duration vary widely in model-difficulty (their own "excess success
  rate" analysis shows correlation ~0.40)
- Only ~15 data points for the trend regression — very little power for curvature

---

## 3. The IRT connection: this is just psychometrics (KEY SECTION — the "aha" for LW readers)

METR's model is essentially a **2-Parameter Logistic (2PL) Item Response Theory** model,
the bread-and-butter of educational testing since the 1960s.

| METR concept | IRT concept |
|---|---|
| AI model | Test-taker ("person") |
| Task | Test item |
| Success/fail | Correct/incorrect |
| Horizon (t50) | Ability (θ) |
| Human time | Item difficulty (b) |
| Logistic slope | Item discrimination (a) |

The model:
```
P(success | model i, task j) = logit⁻¹( a_j · (θ_i - b_j) )
```

The only "extra" ingredient: task difficulty has a mean-function in log(human-time):
```
b_j ~ Normal(α + κ·log(t_j), σ_b)
```

This is just a **2PL IRT model with a covariate on difficulty**. It's a one-line
extension to any IRT textbook model. No custom methodology.

**Why this matters:** IRT is a mature field with 60 years of theory. We get:
- Principled uncertainty quantification (full posterior, not bootstrap)
- Model comparison tools (LOO-CV, WAIC, Bayes factors)
- The ability to add trajectory models as priors on θ, compared within one framework
- A blueprint for recurring updates: when METR publishes new model results, just refit

σ_b ≈ 1.44 — tasks vary substantially beyond what human-time captures. This inflates
uncertainty on any horizon prediction. METR's approach conditions on the point-estimated
difficulties; the Bayesian model integrates over them.

---

## 4. Multiple trajectory models, one framework

We fit the same IRT likelihood with different priors on how θ grows with release date:

| Model | θ(date) | Parameters | Intuition |
|---|---|---|---|
| Linear | γ₀ + γ₁·x | 2 | Constant exponential growth in horizon |
| Quadratic | γ₀ + γ₁·x + γ₂·x² | 3 | Accelerating or decelerating |
| Power-law | γ₀ + γ₁·z^a | 3 (a learned) | Flexible concavity |
| Saturating (t50-logistic) | Maps logistic ceiling on t50 back to θ | 4 | Growth plateaus |
| Singularity | γ₀ + γ₁·x + c/(t*-x)^α | 5 | Finite-time blowup |

All share the same task parameters (b, a, α, κ, σ_b). Only the θ-prior changes.

**Key point for LW:** This is not "picking the model that gives the answer we want."
We fit all of them, compare with LOO cross-validation, and report all forecasts. The
reader can weight them however they like.

---

## 5. Results: the models fit similarly but forecast wildly differently

### 5a. Model comparison (LOO)

| Model | elpd_loo | Δelpd | SE(Δ) |
|---|---|---|---|
| Linear | -2192 | 0 | — |
| Quadratic | -2200 | -8 | ~4 |
| (others TBD — need to run full diagnostics) | | | |

Linear wins by ~8 elpd ≈ ~2 SE. Not decisive. The data cannot strongly distinguish
these trajectory shapes.

### 5b. Calibration

Brier ≈ 0.066, log-score/trial ≈ -0.31. PPC by task-length quantile looks reasonable.
Difficulty residuals show no major systematic patterns (include residual plot).

### 5c. The forecasts

**1-month horizon (30 days): when does a new model's t50 reach it?**

| Trend | P(cross by 2200) | Mean date | 95% CrI |
|---|---|---|---|
| Linear | 100% | 2028-06-24 | 2027-11 to 2029-03 |
| Quadratic | 91% | 2027-09-17 | 2026-11 to 2029-07 |
| Power-law (xpow) | 100% | 2027-10-23 | 2027-02 to 2029-01 |
| Saturating (t50-logistic) | 0% | never | — |

**TODO: Add 125-year horizon table (for AI futures / Tom Davidson-style analysis)**

This is THE figure for the post. One chart showing all four (or five) trend lines with
credible bands, fanning out into the future, dramatically diverging.

### 5d. The singularity model

Tried it. The data are completely uninformative — the posterior on the singularity date
(t*) is essentially the prior. With 15 models spanning ~2 years of release dates, you
simply cannot constrain a 5-parameter blowup model. This is good to report: it shows
the limits of what these data can tell us. (It does NOT mean there's no singularity — it
means these data can't detect one either way.)

---

## 6. What would it take to distinguish the models? (brief, forward-looking)

- More models with later release dates (the next 2-3 METR data releases will help a lot)
- When a new model comes, you can compute LOO or posterior predictive scores *before seeing
  its benchmark results*, then compare — this is a genuine Bayesian forecast evaluation, not
  post-hoc curve fitting
- Quantitative: expected value of information analysis (how many more models at what spacing
  to get 3:1 Bayes factor between linear and quadratic?)

---

## 7. Policy implications (keep measured, don't oversell)

- METR's "7 months ± something" is an honest *conditional* estimate. The conditional on
  "linear trend" doing the heavy lifting.
- If you weight saturating models at all (and there's no strong reason not to), the
  unconditional CI on "when do we hit 1-month horizon" goes from [2028, 2029] to
  [2027, never].
- This doesn't mean we should be more or less alarmed. It means the uncertainty is real and
  we should build governance frameworks that are robust to it.
- The Bayesian framework gives a principled update rule: new METR data → refit → updated
  posteriors → updated trajectory weights. No hand-waving needed.

---

## 8. Limitations / caveats

- We use the same data as METR; garbage-in, garbage-out if the benchmarks are unrepresentative
- Release date is a crude proxy for "capability investment" — conflates compute, algorithms,
  architecture, RLHF, tool use
- Anchoring (GPT-4 = -1, Claude 3.7 Sonnet = +1) is arbitrary; results are
  scale-equivariant but worth noting
- We haven't tried change-point models or Gaussian process trajectories (future work)
- Small N (15 models) means priors matter — we use weakly informative priors throughout but
  sensitivity analysis would strengthen things

---

## 9. Conclusion

Standard psychometric methods (2PL IRT) fit the METR benchmark data well. The trajectory
question — is AI capability growth linear, accelerating, or saturating? — is not answerable
from current data alone. Different functional forms fit the data nearly equally well but
produce forecasts ranging from "1-month horizon by 2027" to "never." This structural
uncertainty should be front-and-centre in any policy discussion using METR-style horizons.

Code + data: [link to repo]

---

## Figures needed

1. **Theta vs. date** with credible bands and multiple trend lines overlaid (THE key figure)
2. **Horizon fan chart**: horizon curves for all models, fanning out into future
3. **1-month crossing table** (already generated)
4. **125-year crossing table** (TODO)
5. **Difficulty residuals** (shows the random effects are well-behaved)
6. **Calibration plot** (shows the model actually predicts well)
7. Optional: doubling-time posterior (shows uncertainty in the "7 months" claim)

---

## My editorial thoughts (Claude's notes to Jmoss)

**What would impress a top-tier LW reader:**
- The IRT framing is genuinely novel in this context and will land well with the "use
  existing tools from mature fields" crowd. Emphasise this.
- The structural uncertainty result is the real contribution. Everyone debates "are we on
  exponential or S-curve?" — you can show quantitatively that the data can't answer this.
- The singularity result is a nice negative finding. LW loves "I tried the dramatic thing
  and here's what happened."
- The Bayesian update angle ("when the next model drops, we can score our predictions") is
  very alignment-forum-friendly. Makes this a living analysis, not a one-off.

**What might concern a careful reader:**
- Only 15 models → limited trend data. Be upfront about this. It's not a weakness of your
  analysis, it's a feature — you're correctly quantifying how little info 15 points give you
  about trajectory shape.
- Anchoring choices. Worth a footnote. Show it doesn't change conclusions.
- The "marginal" vs "typical" horizon distinction is subtle — might need a clear box/sidebar.
- Make sure the LOO comparison includes ALL fitted models, not just linear vs quadratic.

**What would make this METR-hiring-relevant:**
- Shows you can identify the right statistical framework (IRT) for a novel problem
- Shows you can implement it properly in Stan (not just call a package)
- Shows you think carefully about model comparison, not just model fitting
- Shows you can communicate technical results to a policy-adjacent audience
- The "principled Bayesian updating" angle aligns with METR's stated desire for
  "more sophisticated statistical methods"

**Missing pieces to fill in:**
- [ ] Run diagnostics on ALL fitted specs (not just linear + quadratic)
- [ ] Generate the compare_horizon_plot overlay (the "fan chart")
- [ ] Compute 1-month AND 125-year horizon tables for all specs
- [ ] Write the actual prose (this outline → paragraphs)
- [ ] Review Stan priors (quick sanity check)
