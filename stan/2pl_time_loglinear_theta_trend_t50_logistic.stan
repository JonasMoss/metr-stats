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

  // Centering constant used to define x: x_j = log(t_j_hours) - mean_log_t_hours
  real mean_log_t_hours;

  // Release dates for trajectory model (years since mean dated model release).
  vector[I] x_date;
  array[I] int<lower=0, upper=1> has_date;
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

  // Saturating trajectory for the *t50 horizon in hours*:
  //   t50(x) = t_low + (t_high - t_low) * inv_logit(a_t + b_t * x)
  real log_t_low;                 // log hours
  real log_delta_t;               // log(t_high - t_low), hours
  real a_t;
  real<lower=0> b_t;

  real<lower=1e-6> sigma_theta;
}

transformed parameters {
  vector[I] theta;
  real<lower=0> t_low = exp(log_t_low);
  real<lower=0> delta_t = exp(log_delta_t);
  real<lower=0> t_high = t_low + delta_t;

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
  // Abilities: dated models are shrunk toward a horizon-based saturating curve.
  // For p=0.5, the implied horizon satisfies:
  //   log t50 = mean_log_t_hours + (theta - alpha)/kappa
  // => theta = alpha + kappa * (log t50 - mean_log_t_hours)
  //
  // So we model t50(x_date) as a logistic curve in *hours*, then map to a mean theta.
  log_t_low ~ normal(log(1.0 / 3600.0), 2.0);     // ~ 1 sec in hours, broad
  log_delta_t ~ normal(log(720.0), 3.0);          // ~ 1 month in hours, very broad
  a_t ~ normal(0, 2.0);
  b_t ~ normal(0, 1.0);
  sigma_theta ~ normal(0, 1.0);

  alpha ~ normal(0, 1.5);
  kappa ~ normal(0, 1);
  sigma_b ~ normal(0, 1);
  b ~ normal(alpha + kappa * x, sigma_b);

  mu_loga ~ normal(0, 0.5);
  sigma_loga ~ normal(0, 0.5);
  log_a ~ normal(mu_loga, sigma_loga);

  for (i in 1:I) {
    if (i == anchor_low || i == anchor_high) continue;
    if (has_date[i] == 1) {
      real s_curve = inv_logit(a_t + b_t * x_date[i]);
      real t50 = t_low + (t_high - t_low) * s_curve;
      real mu_theta = alpha + kappa * (log(t50) - mean_log_t_hours);
      theta[i] ~ normal(mu_theta, sigma_theta);
    } else {
      theta[i] ~ normal(0, 1.5);
    }
  }

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
      real s_curve = inv_logit(a_t + b_t * x_date[i]);
      real t50 = t_low + (t_high - t_low) * s_curve;
      mu_theta[i] = alpha + kappa * (log(t50) - mean_log_t_hours);
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

