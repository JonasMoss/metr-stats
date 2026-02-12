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

  // Release dates (years since mean dated model release).
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

  // Saturating ability trend (sigmoid in theta-space):
  //   theta(x) = theta_min + theta_range * inv_logit(a_logis + b_logis * x)
  real theta_min;
  real<lower=0> theta_range;
  real a_logis;
  real<lower=0> b_logis;
  real<lower=1e-6> sigma_theta;
}

transformed parameters {
  vector[I] theta;
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
  // Priors for logistic trend parameters
  theta_min ~ normal(-2, 2.0);
  theta_range ~ normal(5, 3.0);
  a_logis ~ normal(0, 2.0);
  b_logis ~ normal(0, 1.0);
  sigma_theta ~ normal(0, 1.0);

  alpha ~ normal(0, 1.5);
  kappa ~ normal(0, 1);
  sigma_b ~ normal(0, 1);
  b ~ normal(alpha + kappa * x, sigma_b);

  mu_loga ~ normal(0, 0.5);
  sigma_loga ~ normal(0, 0.5);
  log_a ~ normal(mu_loga, sigma_loga);

  // Ability shrinkage toward saturating trend
  for (i in 1:I) {
    if (i == anchor_low || i == anchor_high) continue;
    if (has_date[i] == 1) {
      real s_curve = inv_logit(a_logis + b_logis * x_date[i]);
      theta[i] ~ normal(theta_min + theta_range * s_curve, sigma_theta);
    } else {
      theta[i] ~ normal(0, 1.5);
    }
  }

  // Likelihood
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
      real s_curve = inv_logit(a_logis + b_logis * x_date[i]);
      mu_theta[i] = theta_min + theta_range * s_curve;
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
