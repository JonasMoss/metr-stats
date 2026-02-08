data {
  int<lower=1> N;                 // number of observed (model,item) pairs
  int<lower=1> I;                 // number of models ("people")
  int<lower=1> J;                 // number of items (tasks)
  array[N] int<lower=1, upper=I> ii;
  array[N] int<lower=1, upper=J> jj;
  array[N] int<lower=0> n;        // attempts
  array[N] int<lower=0> s;        // successes
  vector[J] x;                    // centered log task length: log(t_j) - mean(log t)

  int<lower=1, upper=I> anchor_low;
  int<lower=1, upper=I> anchor_high;
  real theta_low;
  real theta_high;

  // Release-date covariate for abilities (years since some reference date).
  vector[I] x_date;
  real x_date_max;                // max x_date among dated models
  array[I] int<lower=0, upper=1> has_date;  // whether model i participates in the θ(d) trend (exclude e.g. human)
}

parameters {
  // Abilities with two fixed anchors for identification
  vector[I - 2] theta_free;

  // Difficulty mean function b(t) = alpha + kappa * x
  real alpha;
  real<lower=0> kappa;
  real<lower=1e-6> sigma_b;
  vector[J] b;                    // task difficulties

  // Discrimination is constant in t, but heterogeneous across tasks
  real mu_loga;
  real<lower=1e-6> sigma_loga;
  vector<lower=-2, upper=2>[J] log_a;

  // Finite-time singularity trend for ability:
  //   θ(d) = gamma0 + gamma1*x + c / (t_star - x)^alpha_sing
  real gamma0;
  real gamma1;
  real<lower=0> c_sing;
  real<lower=0> alpha_sing;
  real eta_tstar;                 // t_star = x_date_max + exp(eta_tstar) + 0.25
  real<lower=1e-6> sigma_theta;
}

transformed parameters {
  vector[I] theta;
  real t_star;

  // Keep t_star strictly in the future, and not too close to the last observed date.
  t_star = x_date_max + exp(eta_tstar) + 0.25;

  {
    int pos;
    pos = 1;
    for (i in 1:I) {
      if (i == anchor_low) {
        theta[i] = theta_low;
      } else if (i == anchor_high) {
        theta[i] = theta_high;
      } else {
        theta[i] = theta_free[pos];
        pos += 1;
      }
    }
  }
}

model {
  // Trend priors: intentionally informative enough to keep the sampler away from pathologies.
  gamma0 ~ normal(0, 1.5);
  gamma1 ~ normal(0, 1.0);
  c_sing ~ normal(0, 1.0);                  // half-normal (c_sing >= 0)
  alpha_sing ~ lognormal(log(1.0), 0.35);   // concentrated around 1
  eta_tstar ~ normal(log(8.0), 0.7);        // t_star - x_max typically ~ 8y, but wide
  sigma_theta ~ normal(0, 1.0);

  // Ability "prior" as a regression likelihood for dated models.
  // This includes the anchors (since theta[anchor_*] are fixed values), which helps identify the trend.
  for (i in 1:I) {
    if (has_date[i] == 1) {
      real denom = t_star - x_date[i];
      real mu = gamma0 + gamma1 * x_date[i] + c_sing / pow(denom, alpha_sing);
      target += normal_lpdf(theta[i] | mu, sigma_theta);
    } else if (i != anchor_low && i != anchor_high) {
      theta[i] ~ normal(0, 1.5);
    }
  }

  alpha ~ normal(0, 1.5);
  kappa ~ normal(0, 1);
  sigma_b ~ normal(0, 1);
  b ~ normal(alpha + kappa * x, sigma_b);

  mu_loga ~ normal(0, 0.5);
  sigma_loga ~ normal(0, 0.5);
  log_a ~ normal(mu_loga, sigma_loga);

  for (k in 1:N) {
    real a = exp(log_a[jj[k]]);
    real eta = a * (theta[ii[k]] - b[jj[k]]);
    s[k] ~ binomial_logit(n[k], eta);
  }
}

generated quantities {
  vector[J] a = exp(log_a);
  vector[I] mu_theta;
  vector[N] log_lik;

  for (i in 1:I) {
    if (has_date[i] == 1) {
      real denom = t_star - x_date[i];
      mu_theta[i] = gamma0 + gamma1 * x_date[i] + c_sing / pow(denom, alpha_sing);
    } else {
      mu_theta[i] = 0;
    }
  }

  for (k in 1:N) {
    real a_k = exp(log_a[jj[k]]);
    real eta_k = a_k * (theta[ii[k]] - b[jj[k]]);
    log_lik[k] = binomial_logit_lpmf(s[k] | n[k], eta_k);
  }
}

