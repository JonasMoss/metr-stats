# Bayesian Model Comparison for AI Capability Trajectories

## Project notes — Feb 2025

### The METR paper

Kwa, West, Becker et al. (2025), "Measuring AI Ability to Complete Long Tasks" ([arXiv:2503.14499](https://arxiv.org/abs/2503.14499)).

Headline claim: the **50% task-completion time horizon** — the duration of tasks that frontier AI models can complete with 50% success rate — has been doubling every ~7 months since 2019. Extrapolation: 1-month-horizon AI (167 working hours) between late 2028 and early 2031.

This is becoming load-bearing for AI governance and lab safety policies.

---

### What METR does statistically

**Data structure.** Let $i$ index models (~15–20 now), $j$ index tasks (170), $k$ index attempts (~8 per model-task pair). $Y_{ijk} \in \{0,1\}$ is binary success. Each task $j$ has a human difficulty rating $\log t_j$ (geometric mean time of successful human baselines).

**Stage 1: Per-model logistic regression.**

$$P(Y_{ijk} = 1) = \sigma\bigl(\beta_i (\log h_i - \log t_j)\bigr)$$

Two parameters per model: $h_i$ (50% time horizon) and $\beta_i$ (slope). $h_i$ is the task duration at which the fitted logistic crosses 0.5.

**Stage 2: OLS trend.**

$$\log h_i = \alpha + \gamma \cdot \text{date}_i + \varepsilon_i$$

Doubling time $= \log 2 / \gamma$. Confidence intervals from three-level hierarchical bootstrap (task families → tasks → runs), 10,000 iterations.

**Problems with this approach:**

- Two-stage estimation discards uncertainty from Stage 1 unless you bootstrap the whole pipeline
- $\log t_j$ treated as known difficulty, but tasks at the same human duration vary widely in model-difficulty
- Only ~15–20 points for the trend regression — very little power for detecting curvature or change points
- Release date conflates compute, algorithms, architecture, agency training, lab competition

---

### Improvement 1: Random task difficulty effects (GLMM)

Human time is a noisy proxy for model-relevant difficulty. Their own "excess success rate" analysis (correlation ~0.40 across models) documents this. Natural fix:

$$d_j = \log t_j + u_j, \quad u_j \sim \mathcal{N}(0, \sigma_u^2)$$

$$P(Y_{ijk} = 1) = \sigma\bigl(\beta(\theta_i - \log t_j - u_j)\bigr)$$

This is a standard GLMM. $\sigma_u^2$ quantifies how much task difficulty varies beyond what human time captures. Can be extended with a model-task interaction $v_{ij}$ if needed.

If $\sigma_u^2$ is large, CIs on everything downstream widen. Worth checking.

---

### Improvement 2: Separate capability from time trend

The deeper issue: release date is a proxy, not a causal variable. Better to:

1. **Estimate model capabilities $\theta_i$ from the task data** using IRT / the GLMM above. No date variable. This step is well-identified (170 × 8 observations per model).

2. **Model the trajectory $\theta_i$ over time (or compute, etc.) separately.** This is where the forecasting question lives, and where you have only ~15–20 data points.

This decomposition keeps capability estimation clean and makes the forecasting assumptions explicit.

---

### The actual project: Bayesian model comparison for the trajectory

The big question: **is the trend exponential, sub-exponential, or super-exponential?** Everything policy-relevant depends on this. Nobody is quantifying the structural uncertainty.

**Proposed approach:** Bayesian model averaging over a small set of trajectory models.

Given estimated capabilities $\hat{\theta}_i$ (with uncertainty) from the GLMM, consider:

- $M_1$: **Exponential.** $\theta = \alpha + \gamma t$ (linear in $\theta$, exponential in horizon)
- $M_2$: **Accelerating/decelerating.** $\theta = \alpha + \gamma t + \delta t^2$
- $M_3$: **Change-point.** Piecewise linear with breakpoint $\tau$ (kink at reasoning-model transition?)
- $M_4$: **Saturating.** $\theta = L \cdot \sigma(a + bt)$ (logistic ceiling)

Put priors on all parameters and on model probabilities. Compute posterior model weights and model-averaged predictive distribution for "when does $h$ reach 167 hours?"

**What this produces:**

- Posterior probability that the trend is accelerating vs. constant vs. decelerating
- Model-averaged forecast with uncertainty reflecting both parameter and structural uncertainty
- Quantification of how many more data points would be needed to discriminate between trajectory shapes (expected value of information)
- A mechanical update procedure: when METR publishes new results, rerun and report updated posteriors

**Honest expectation:** with ~15–20 models, model selection will not be decisive. That's the point — the forecast uncertainty is much wider than METR's CI suggests, and the extra width comes from structural uncertainty that their approach ignores.

---

### Prior considerations

The priors on trajectory shape will matter with this little data. Possible anchors:

- Historical technology S-curves (but which technology is the right reference class?)
- Compute scaling trends from Epoch AI (training compute doubling every 6–10 months)
- Theoretical arguments: feedback loops from AI R&D automation (superexponential), data/compute exhaustion (subexponential)
- Could calibrate against the Roodman (2020) superexponential growth literature

This is where domain expertise from METR/LW-adjacent people would complement the statistical framework.

---

### Venues and strategy

**Blog post first (LessWrong):**

- Pull METR data from `github.com/METR/eval-analysis-public`
- Reproduce their results
- Fit the GLMM, check if $\sigma_u^2$ matters
- Fit alternative trajectory shapes, report Bayes factors
- Fast, gets visibility with the right audience, functions as proof of concept
- Timeline: ~1 week

**Paper (if results are interesting):**

- JRSS-A: frame as "statistical methodology for policy-relevant AI forecasting"
- ML workshop (ICML/NeurIPS benchmarking): frame as "better evaluation methodology"
- Possible METR collaboration — they explicitly call for "more sophisticated statistical methods"

**Research program (if the space is worth entering):**

- "Psychometrics for AI" — IRT, test equating, adaptive testing, DIF, applied to AI evaluation
- Recurring updates as new models are evaluated
- Policy-relevant niche, methodologically deep, currently vacant

---

### Immediate next steps

1. Clone `github.com/METR/eval-analysis-public`, understand data structure
2. Reproduce Figure 1
3. Fit basic GLMM with random task effects in Stan (or brms)
4. Estimate $\theta_i$ per model, plot against date
5. Fit $M_1$–$M_4$, compute Bayes factors
6. Write up on LessWrong
7. Email Ben West if results are interesting

---

### Key references

- Kwa et al. (2025). Measuring AI Ability to Complete Long Tasks. arXiv:2503.14499
- Roodman (2020). On the probability distribution of long-term changes in the growth rate of the global economy
- Cotra (2020). Draft report on AI timelines (bio-anchors)
- Ngo (2023). Clarifying and predicting AGI
- Baker (2001). The basics of item response theory
- Epoch AI. Data on notable AI models