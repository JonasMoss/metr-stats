#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_flag_value <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx)) return(default)
  if (idx == length(args)) stop(paste0("Missing value after ", flag))
  args[[idx + 1]]
}

has_flag <- function(flag) flag %in% args

counts_in <- get_flag_value("--counts", "data/irt_counts_task_id.csv")
outdir <- get_flag_value("--outdir", "analysis/out_r")
exclude_fatal <- get_flag_value("--exclude-fatal", NA_character_)

theta_low <- as.numeric(get_flag_value("--theta-low", "-1.0"))
theta_high <- as.numeric(get_flag_value("--theta-high", "1.0"))
sigma_theta <- as.numeric(get_flag_value("--sigma-theta", "2.0"))
sigma_b <- as.numeric(get_flag_value("--sigma-b", "2.0"))
sigma_loga <- as.numeric(get_flag_value("--sigma-loga", "1.0"))
maxit <- as.integer(get_flag_value("--maxit", "2000"))

anchor_low <- get_flag_value("--anchor-low", NA_character_)
anchor_high <- get_flag_value("--anchor-high", NA_character_)

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(counts_in, stringsAsFactors = FALSE)

if (!all(c("model", "n", "s") %in% names(df))) {
  stop("--counts must have columns: model, <item>, n, s")
}

item_col <- setdiff(names(df), c("model", "n", "s", "f"))[1]
if (is.na(item_col)) stop("Could not infer item column in --counts")

if (!is.na(exclude_fatal)) {
  warning("--exclude-fatal is ignored for --counts input; filter at build time from runs if needed.")
}

df$n <- as.numeric(df$n)
df$s <- as.numeric(df$s)

models <- sort(unique(df$model))
items <- sort(unique(df[[item_col]]))
m <- length(models)
j <- length(items)

model_index <- setNames(seq_len(m), models)
item_index <- setNames(seq_len(j), items)

i_idx <- model_index[df$model]
j_idx <- item_index[df[[item_col]]]
n_obs <- df$n
s_obs <- df$s

totals <- aggregate(cbind(n, s) ~ model, data = df, FUN = sum)
totals$pass_rate <- totals$s / totals$n
totals <- totals[order(totals$pass_rate), ]

if (is.na(anchor_low)) anchor_low <- totals$model[[1]]
if (is.na(anchor_high)) anchor_high <- totals$model[[nrow(totals)]]

if (!(anchor_low %in% models)) stop(paste0("anchor-low not found: ", anchor_low))
if (!(anchor_high %in% models)) stop(paste0("anchor-high not found: ", anchor_high))
if (anchor_low == anchor_high) stop("anchor-low and anchor-high must differ")

low_idx <- model_index[[anchor_low]]
high_idx <- model_index[[anchor_high]]

free_mask <- rep(TRUE, m)
free_mask[low_idx] <- FALSE
free_mask[high_idx] <- FALSE
free_indices <- which(free_mask)

item_totals <- aggregate(cbind(n, s) ~ df[[item_col]], data = df, FUN = sum)
names(item_totals)[1] <- item_col
item_totals$rate <- (item_totals$s + 0.5) / (item_totals$n + 1.0)
item_totals <- item_totals[match(items, item_totals[[item_col]]), ]
b0 <- -qlogis(item_totals$rate)
theta0 <- rep(0.0, length(free_indices))
loga0 <- rep(0.0, j)

x0 <- c(theta0, b0, loga0)

unpack <- function(x) {
  theta_free <- x[seq_len(length(free_indices))]
  b <- x[length(free_indices) + seq_len(j)]
  loga <- x[length(free_indices) + j + seq_len(j)]

  theta <- rep(0.0, m)
  theta[low_idx] <- theta_low
  theta[high_idx] <- theta_high
  theta[free_indices] <- theta_free

  a <- exp(loga)
  list(theta = theta, b = b, a = a, theta_free = theta_free, loga = loga)
}

objective <- function(x) {
  u <- unpack(x)
  eta <- u$a[j_idx] * (u$theta[i_idx] - u$b[j_idx])
  p <- plogis(eta)

  nll <- -sum(dbinom(s_obs, size = n_obs, prob = p, log = TRUE))

  pen <- 0.5 * (
    sum(u$theta_free^2) / (sigma_theta^2) +
      sum(u$b^2) / (sigma_b^2) +
      sum(u$loga^2) / (sigma_loga^2)
  )
  nll + pen
}

fit <- optim(
  par = x0,
  fn = objective,
  method = "L-BFGS-B",
  control = list(maxit = maxit)
)

u_hat <- unpack(fit$par)

model_df <- data.frame(model = models, theta = u_hat$theta, stringsAsFactors = FALSE)
model_df <- merge(model_df, totals[, c("model", "pass_rate")], by = "model", all.x = TRUE)
model_df <- model_df[order(-model_df$theta), ]

item_df <- data.frame(item = items, a = u_hat$a, b = u_hat$b, stringsAsFactors = FALSE)
names(item_df)[1] <- item_col

item_totals2 <- aggregate(cbind(n, s) ~ df[[item_col]], data = df, FUN = sum)
names(item_totals2)[1] <- item_col
item_totals2$pass_rate <- item_totals2$s / item_totals2$n
item_df <- merge(item_df, item_totals2[, c(item_col, "pass_rate")], by = item_col, all.x = TRUE)
item_df <- item_df[order(item_df$b), ]

write.csv(model_df, file.path(outdir, "2pl_models.csv"), row.names = FALSE)
write.csv(item_df, file.path(outdir, "2pl_items.csv"), row.names = FALSE)

meta_lines <- c(
  paste0("convergence=", fit$convergence),
  paste0("message=", fit$message),
  paste0("value=", fit$value),
  paste0("counts_in=", counts_in),
  paste0("item_col=", item_col),
  paste0("anchor_low=", anchor_low, " theta_low=", theta_low),
  paste0("anchor_high=", anchor_high, " theta_high=", theta_high),
  paste0("maxit=", maxit)
)
writeLines(meta_lines, file.path(outdir, "2pl_meta.txt"))

cat(
  "fit:",
  paste0("convergence=", fit$convergence),
  paste0("anchors=(", anchor_low, " -> ", theta_low, ", ", anchor_high, " -> ", theta_high, ")"),
  "\n"
)
cat("wrote:", file.path(outdir, "2pl_models.csv"), "\n")
cat("wrote:", file.path(outdir, "2pl_items.csv"), "\n")
cat("wrote:", file.path(outdir, "2pl_meta.txt"), "\n")

