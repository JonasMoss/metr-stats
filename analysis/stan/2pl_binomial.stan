data {
  int<lower=1> N;                 // number of observed (model,item) pairs
  int<lower=1> I;                 // number of models ("people")
  int<lower=1> J;                 // number of items (tasks or task families)
  array[N] int<lower=1, upper=I> ii;
  array[N] int<lower=1, upper=J> jj;
  array[N] int<lower=0> n;        // attempts
  array[N] int<lower=0> s;        // successes
  int<lower=1, upper=I> anchor_low;
  int<lower=1, upper=I> anchor_high;
  real theta_low;
  real theta_high;
}

parameters {
  vector[J] b;                    // item difficulty
  vector<lower=-2, upper=2>[J] log_a; // item log discrimination (bounded for stability)
  vector[I - 2] theta_free;
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
  theta_free ~ normal(0, 1.5);
  b ~ normal(0, 1.5);
  log_a ~ normal(0, 0.5);

  for (k in 1:N) {
    real a = exp(log_a[jj[k]]);
    real eta = a * (theta[ii[k]] - b[jj[k]]);
    s[k] ~ binomial_logit(n[k], eta);
  }
}

generated quantities {
  vector[J] a = exp(log_a);
}
