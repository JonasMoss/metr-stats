# (Updated) METR’s data can’t distinguish between trajectories (and 80%
horizons are an order of magnitude off)
Jonas Moss
2026-02-19

*Update:* Added GPT-5.2 to the main part of the text, this uses all data
from v1.1. Added appendix using all METR models, by joining v1.0 and
v1.1. Added appendix with marginal vs typical P(success) curves. Thanks
to Thomas Kwa for telling me about this.

## TLDR

I reanalyzed the METR task data using a Bayesian item response theory
model.

- **The METR data cannot distinguish exponential from superexponential
  growth.** Four trajectory shapes (linear, quadratic, power-law,
  saturating) fit the existing data equally well but diverge on
  forecasts. For instance, the 95% credible interval for the 125-year
  crossing is 2031-06 – 2033-10 for linear and 2028-07 – 2032-03 for
  quadratic.
- **METR’s headline horizon numbers overstate current capability by
  roughly an order of magnitude at 80% success.** METR doesn’t model
  variation in task difficulty, so their horizons reflect a task of
  typical difficulty for its length. But tasks of the same length vary a
  lot in how hard they are, and difficult tasks pull the horizon down
  more than the easy tasks push it up. Curiously, this doesn’t affect
  timelines by more than ~1 year, as it’s just a level-shift.
- **We need data about the human times to quantify uncertainty.**
  Credible intervals throughout are too narrow because I treat human
  times as known rather than estimating using latent variables. I’m
  doing this because I don’t have access to all the raw data. This could
  be a big deal, and could also affect the $80\%$ horizons.
- Doubling time under the standard linear (exponential growth) model is
  ~4.3 months, which is similar to METR’s estimate (95% credible
  interval: 3.7–5.1, but see caveat above).

## METR data

Let’s start with a plot that shouldn’t be too surprising. Four
reasonable models fit the [METR data (v
1.1)](https://github.com/METR/eval-analysis-public/tree/main/reports/time-horizon-1-1/data/raw)
equally well. They agree about the past but disagree strongly about the
future.

![](https://raw.githubusercontent.com/JonasMoss/metr-stats/main/metr-stats_files/figure-commonmark/fig-horizon-fan-output-1.png)

The model selection scores known as ELPD-LOO differ by at most ~6
points.[^1] Calibration is nearly identical, with Brier $\approx$ 0.067
across the board. Your prior matters a lot here and has clear-cut
consequences, as the models agree about the past but disagree strongly
about the future. The latest data point is GPT-5.2 (December 2025).

These curves are fitted using a Bayesian item response theory model
described below. Before describing it, let’s recall METR’s analysis of
the time horizon. They proceed in two stages:

1.  *Per-model logistic regression.* For each model $i$, fit
    $P(\text{success}) = \sigma(\beta_i(\log h_i - \log t_j))$ where
    $t_j$ is human time for task $j$. Here $h_i$ is the task duration
    where the curve crosses 50%. When $t_j = h_i$, we get
    $\sigma(0) = 0.5$, a $50\%$ horizon. This gives a “horizon score”
    $h_i$ per model.

2.  *An OLS trend.* Regress $\log h_i$ on release date. The slope gives
    a doubling time of ~4 months.

This is good modeling and gets the main story right, but there are some
non-standard choices here. For instance, the slope $\beta_i$ varies with
model rather than task (which is unusual in item response theory) and
Stage 1 uncertainty is not accounted for in Stage 2 (METR uses the
bootstrap). It also treats every task of the same length as equally
difficult and only considers one trajectory shape.

In this post I make a joint model, adjust some things to be more in line
with standard practice, and ask what happens when you try different
trajectory shapes. The post is somewhat technical, but not so god-awful
that Claude won’t be able to answer any question you have about the
methodology. Models are fitted with Stan, 4 chains $\times$ 1000
post-warmup draws, with code available
[here](https://github.com/JonasMoss/metr-stats). I intentionally won’t
go into details about technicalities, e.g. prior choices – the code
contains everything you’ll want to know and your favorite LLM will
figure it out for you. (All priors were chosen by Codex / Claude Code
and appear reasonable enough.)

## The basic model

The first stage of METR’s model is *almost* a 2-parameter logistic model
(2PL), the workhorse of educational testing since the 1960s.

So, what kind of problems was the 2PL model designed for? Say you give
200 students a math exam with 50 questions and record their answers as
correct / incorrect. You want to estimate the students’ math ability,
but raw percent correct scores aren’t necessarily very good, as they
depend on which questions (easy or hard? relative to which students?)
happened to be on the exam.

The 2PL model solves this by giving each student a single ability score
($\theta_i$) and each question two parameters: a *difficulty* ($b_j$,
how hard it is) and a *discrimination* ($a_j$, how cleanly it separates
strong from weak students). “What is 3×2?” has low discrimination as
everyone gets it right regardless of ability. A simple proof-writing
question has high discrimination as sufficiently strong students can
solve it, but weak students have no chance.

The model estimates all parameters simultaneously via a logistic
regression:

$$
P(\text{success} \mid \text{model } i, \text{task } j) = \text{logit}^{-1}\bigl(a_j (\theta_i - b_j)\bigr)
$$

This matters here because METR tasks are like exam questions. They vary
in both difficulty and how well they separate strong from weak models,
and we want to put all the models on a common ability scale.

## Modeling difficulty

Ability and difficulty parameters $\theta_i, b_j$ in the 2PL are hard to
interpret. The scale is arbitrary, and it’s not clear what, for
instance, a 0.1 increase in ability actually means. Or whether it would
be better to take a log-transform of the parameter, etc. The METR data
is cool and famous because each task comes with a human time, which
gives us a natural and interpretable scale for difficulty. So let’s
connect human time to difficulty first.

$$
b_j \sim \mathcal{N}(\alpha + \kappa \log t_j, \;\sigma_b)
$$

Each task’s difficulty has a mean that depends on log human time, plus a
random component to account for the fact that same-length tasks are not
born equal. (METR treats all tasks of identical length as equally hard.)

Since difficulty increases with log human time at rate $\kappa$, we can
convert any difficulty value back into a time, an *equivalent difficulty
time*. If a task takes humans 10 minutes but is unusually hard for AI,
its equivalent difficulty time might be 50 minutes. A task with human
time $t$ and difficulty residual $u$ has equivalent difficulty time
$t \cdot \exp(u / \kappa)$.[^2]

I estimate $\sigma_b \approx$ 1.51 (posterior median), which is quite
large once we interpret it. One standard deviation of unexplained
difficulty corresponds to a ~4.7x multiplier in equivalent difficulty
time.[^3] A task that’s $1\sigma$ harder than the average for its length
is as hard as a task 4.7x longer. And a task that’s $2\sigma$ harder is
as hard as a task roughly 23x longer. So tasks of identical human time
can span a huge range in difficulty for the AI models.

Of course, this is a modeling choice that can be wrong. There’s no
guarantee that difficulty is linear in $\log t_j$, so we need
diagnostics to check. The plot below does double duty as model
diagnostic and explanation of what the random effect means in practice.

![](https://raw.githubusercontent.com/JonasMoss/metr-stats/main/metr-stats_files/figure-commonmark/fig-difficulty-variation-output-1.png)

A plotted dot at 5x means the task’s equivalent difficulty time is 5x
its actual human time. Even within the $\pm 1\sigma$ band, tasks of
identical human time can differ multiplicatively by a factor of 23x in
equivalent difficulty time, so the practical spread is enormous.

There’s not too much curvature in the relationship between log human
time and difficulty, so I think the log-linear form is decent, but it’s
much more spread out than we’d like. There is a cluster of easy outliers
on the far left, which I think can be explained by very short tasks
containing virtually no information about difficulty. Overall this looks
reasonable for modeling purposes.

## Modeling ability over time

By directly modeling ability over time, we can try out shapes like
exponential, subexponential, superexponential, saturating, and
singularity. Forecasts depend a lot on which shape you pick, and the
data doesn’t really tell you much, so it’s not easy to choose between
them. Your priors rule here.

The abilities are modeled as $$
\theta_i \sim \mathcal{N}\bigl(f(x_i;\, \gamma),\; \sigma_\theta\bigr)
$$

where $x_i$ is the model release date in years, centered at the mean
(September 2024). I’m still using a random effect for model ability
here, since nobody seriously thinks every model released on the same
date must be equally capable. I’m looking at four shapes for $f$:[^4]

| Model | $f(x)$ | Params | Intuition |
|:---|:---|:--:|:---|
| Linear | $\gamma_0 + \gamma_1 x$ | 2 | Linear $\theta$ = exponential horizon growth (constant doubling time) |
| Quadratic | $\gamma_0 + \gamma_1 x + \gamma_2 x^2$, $\gamma_2 \geq 0$ | 3 | Superexponential, accelerating growth |
| Power-law | $\gamma_0 + \gamma_1 \tilde{x}^{\alpha}$, $\alpha \in [0.1, 2]$ | 3 | Flexible: sub- or super-exponential. $\tilde{x}$ is a shifted/scaled version of $x$. |
| Saturating | $\theta_{\min} + \Delta\theta \cdot \text{logit}^{-1}(a + bx)$ | 4 | S-curve ceiling on ability. |

If METR’s GitHub repo contained all the historical data, I would also
have tried a piecewise linear with a breakpoint around the time of o1,
which visually fits the original METR graphs better than a plain linear
fit. But since the available data doesn’t go that far back, I don’t need
to, and the value of including those early points in a forecasting
exercise is questionable anyway. Getting hold of the latest data points
is more important. (*Added:* I use all the data in the appendix below,
but I do not attempt a piecewise linear since running the STAN programs
take a lot of time.)

All models share the same 2PL likelihood and task parameters ($b_j$,
$a_j$, $\alpha$, $\kappa$, $\sigma_b$). Only the model for $\theta$
changes.

Each model except the saturating model will cross any threshold given
enough time. Here are posteriors for the 50% crossing across our models.
The saturating model almost never crosses the 1-month and 125-year
thresholds since it saturates too fast.

| Trend     | 1mo Mean | 1mo 95% CrI       | 125y Mean | 125y 95% CrI      |
|:----------|:---------|:------------------|:----------|:------------------|
| Linear    | 2028-09  | 2028-03 – 2029-05 | 2032-07   | 2031-06 – 2033-10 |
| Quadratic | 2027-11  | 2027-03 – 2028-10 | 2029-12   | 2028-07 – 2032-03 |
| Power-law | 2028-01  | 2027-05 – 2029-01 | 2030-06   | 2028-12 – 2033-01 |

## Problems with 80% success

Everything above uses 50% success, but METR also cares about 80% success
and fits a separate model for that. We don’t need to do that here since
the model estimation doesn’t really depend on success rates at all.
We’ll just calculate the 80%-success horizon using posterior draws
instead.

But there are actually two reasonable ways to define “80% success,” and
they give different answers.

1.  *Typical:* Pick a task of average difficulty for its length. Can the
    model solve it 80% of the time? This is roughly what METR computes.

2.  *Marginal:* Pick a random task of that length. What’s the expected
    success rate? Because some tasks are much harder than average, the
    hard ones drag down the average more than easy ones push it up.

At 50%, the two definitions agree exactly. But at 80%, the gap is
roughly an order of magnitude!

![](https://raw.githubusercontent.com/JonasMoss/metr-stats/main/metr-stats_files/figure-commonmark/fig-marginal-typical-output-1.png)

So, on the one hand, it’s the variance ($\sigma_b \approx 1.44$) alone
that causes these two plots to be necessary under our model. But on the
other hand, the difference is not really a consequence of modeling. Some
tasks of the same human time vary a lot in how hard they are for our
models, and a phenomenon like this would happen for *any* model that’s
actually honest about this.

The marginal horizon is the one that matters for practical purposes.
“Typical” is optimistic since it only considers tasks of average
difficulty for their length. The marginal accounts for the full spread
of tasks, so it’s what you actually care about when predicting success
on a random task of some length. That said, from the plot we see
frontier performance of roughly 6 minutes, which does sound sort of
short to me. I’m used to LLMs roughly one-shotting longer tasks than
that, but it usually takes some iterations to get it just right. Getting
the context and subtle intentions right on the first try is hard, so I’m
willing to believe this estimate is reasonable.

Anyway, the predicted crossing dates at 80% success are below. First,
the 1-month threshold (saturating model omitted since it almost never
crosses):

| Trend     | Typical Mean | Typical 95% CrI   | Marginal Mean | Marginal 95% CrI  |
|:----------|:-------------|:------------------|:--------------|:------------------|
| Linear    | 2029-02      | 2028-07 – 2029-10 | 2030-10       | 2029-12 – 2031-10 |
| Quadratic | 2028-02      | 2027-05 – 2029-02 | 2029-01       | 2027-12 – 2030-08 |
| Power-law | 2028-04      | 2027-08 – 2029-06 | 2029-06       | 2028-05 – 2031-02 |

And the 125-year threshold:

| Trend     | Typical Mean | Typical 95% CrI   | Marginal Mean | Marginal 95% CrI  |
|:----------|:-------------|:------------------|:--------------|:------------------|
| Linear    | 2032-12      | 2031-10 – 2034-03 | 2034-08       | 2033-04 – 2036-02 |
| Quadratic | 2030-02      | 2028-08 – 2032-07 | 2030-12       | 2029-02 – 2033-12 |
| Power-law | 2030-09      | 2029-02 – 2033-06 | 2031-09       | 2029-09 – 2035-03 |

Make of this what you will, but let’s go through one scenario. Let’s say
I’m a believer in superexponential models with no preference between
quadratic and power-law, so I have 50-50 weighting on those. Suppose
also I believe that 125 years is the magic number for the auto-coder of
[AI Futures](https://www.aifuturesmodel.com/), but I prefer $80\%$ to
$50\%$ as the latter is too brittle. Then, using the arguably correct
marginal formulation, my timeline has mean roughly April 2031, but the
typical framework yields roughly June 2030 instead. And this isn’t too
bad, just a difference of ~0.9 years! The linear model is similar, with
timelines pushed out roughly 1.7 years. So, the wide marginal-typical
gap doesn’t translate into *that* big of a timeline gap, as both
trajectories have the same “slope”, just at a different level.

Let’s also have a look at METR’s actual numbers. They report an 80%
horizon of around 15 minutes for Claude 3.7 Sonnet (in the original
paper). Our typical 80% horizon for that model under the linear model is
about 21.1 min, and the marginal is about 0.8 min, roughly 15x shorter
than METR’s.

## Modeling $t_j$

The available METR data contains the geometric mean of (typically 2-3
for HCAST) successful human baselines per task, but not the individual
times. Both METR’s analysis and mine treat this reported mean as a known
quantity, discarding uncertainty. But we can model $t_j$ as a latent
variable informed by the reported baselines. This is easy enough to do
in Stan, and would give a more honest picture of what the data actually
supports, as all credible intervals will be widened.

I’d expect smaller differences between the typical and marginal plots at
$80\%$ horizon if the $t_j$ values were modeled properly, as more of the
variance in the random effect would be absorbed by the uncertainty in
$t_j$. I’m not sure how big the effect would be, but getting hold of the
data or doing a short simulation would help.

A technical point: When modeling $t_j$, I would also try a Weibull
distribution instead of log-normal, since the log-normal is typically
heavier-tailed and the Weibull is easier to justify on theoretical
grounds using its failure-rate interpretation.

## Notes and remarks

- I also tried a finite-time singularity model of the form
  $\theta \sim \gamma_0 + \gamma_1 x + c / (t^* - x)^\alpha$. The
  posterior on the singularity date $t^*$ didn’t really move from the
  prior at all. This is no surprise. It just means the data is
  uninformative about $t^\star$.
- There are loads of other knobs you could turn. Perhaps you could
  introduce a discrimination parameter that varies by model and task,
  together with a hierarchical prior. Perhaps you could make
  discrimination a function of time, etc. I doubt any of these would
  change the picture much, if at all. The model fit is good enough as it
  is, even if the uncertainty is likely too small. That said, I don’t
  want to dissuade anyone from trying!
- The power-law model does in principle support both sub- and
  superexponential trajectories ($\alpha < 1$ and $\alpha > 1$,
  respectively, where $\alpha = 1$ is the linear model). The posterior
  puts $P(\alpha < 1) \approx 4\%$, so the data does not support
  subexponential growth. At least when using this model.
- There’s plenty of best-practice stuff I haven’t done, such as prior
  sensitivity analysis. (But we have a lot of data, and I wouldn’t
  expect it to matter too much.)
- The doubling time posterior median is 4.3 months (95% credible
  interval: 3.7–5.1), which is close to METR’s v1.1 estimate. Of course,
  doubling time only makes sense for the linear model above, as the
  doubling time of the other models varies with time.

![](https://raw.githubusercontent.com/JonasMoss/metr-stats/main/metr-stats_files/figure-commonmark/fig-doubling-time-output-1.png)

## Appendix: Results for all models

Recall that the main text uses only METR v1.1 data. In this appendix I
use all available data (v1.0 + v1.1 merged). The overall story is
similar, but the pre-Sonnet-3.5 models introduce a visible kink in the
trajectory that a single smooth trend struggles with. (This is
well-known.)

The ELPD-LOO scores differ by at most ~3 points, with Brier $\approx$
0.064.[^5]

![](https://raw.githubusercontent.com/JonasMoss/metr-stats/main/metr-stats_files/figure-commonmark/fig-horizon-fan-all-output-1.png)

![](https://raw.githubusercontent.com/JonasMoss/metr-stats/main/metr-stats_files/figure-commonmark/fig-marginal-typical-all-output-1.png)

![](https://raw.githubusercontent.com/JonasMoss/metr-stats/main/metr-stats_files/figure-commonmark/fig-difficulty-variation-all-output-1.png)

![](https://raw.githubusercontent.com/JonasMoss/metr-stats/main/metr-stats_files/figure-commonmark/fig-doubling-time-all-output-1.png)

## Appendix: Marginal vs typical success curves

These are fitted on v1.1 data only.

![](https://raw.githubusercontent.com/JonasMoss/metr-stats/main/metr-stats_files/figure-commonmark/fig-p-curves-appendix-output-1.png)

[^1]: The ELPD-LOO estimates are: power-law $\-2598\.2$ (SE $79\.2$),
    saturating $\-2601\.0$ (SE $79\.1$), quadratic $\-2601\.0$ (SE
    $79\.1$), linear $\-2604\.0$ (SE $79\.3$).

[^2]: Define $t^*$ as the human time whose mean difficulty equals $b_j$.
    Then $\alpha + \kappa \log t^* = \alpha + \kappa \log t_j + u_j$, so
    $\log t^* = \log t_j + u_j / \kappa$ and
    $t^* = t_j \exp(u_j / \kappa)$.

[^3]: The multiplier is $\exp(\sigma_b / \kappa)$ where
    $\kappa \approx 0.93$ is the posterior median

[^4]: Quadratic is the simplest choice of superexponential function. You
    could spin a story in its favor, but using it is somewhat arbitrary.
    The power-law is the simplest function that can be both super- and
    subexponential (in practice turns out to be superexponential here
    though), and I included the saturating model because, well, why not?

[^5]: The ELPD-LOO estimates are: power-law $\-5344\.1$ (SE $135\.7$),
    quadratic $\-5345\.4$ (SE $135\.9$), saturating $\-5346\.2$ (SE
    $136\.0$), linear $\-5347\.3$ (SE $135\.7$).
