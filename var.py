import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 1. LOAD / GENERATE RETURNS (simulate daily returns for demo)
# ================================================================
np.random.seed(42)  # reproducible results

days = 252            # trading days in a year
mu = 0.07             # annual mean return (S&P 500)
sigma = 0.15          # annual volatility

daily_mu = mu / days
daily_sigma = sigma / np.sqrt(days)

# Generate 1 year of daily returns
returns = np.random.normal(daily_mu, daily_sigma, days)

# Convert returns to losses
losses = -returns


# ================================================================
# 2. HISTORICAL VALUE AT RISK (95%)
# ================================================================
confidence = 0.95
historical_var = np.percentile(losses, 100 * confidence)


# ================================================================
# 3. VARIANCE–COVARIANCE (PARAMETRIC) VAR
#    Assumes returns are normally distributed
# ================================================================
z_score_95 = 1.65  # one-sided 95% z-score

varcov_var = daily_mu + z_score_95 * daily_sigma
varcov_var = -varcov_var  # convert to loss


# ================================================================
# 4. MONTE CARLO VALUE AT RISK
# ================================================================
num_paths = 10_000
Z = np.random.normal(0, 1, num_paths)

mc_returns = daily_mu + daily_sigma * Z
mc_losses = -mc_returns

montecarlo_var = np.percentile(mc_losses, 100 * confidence)


# ================================================================
# 5. PRINT RESULTS
# ================================================================
print("=========== VALUE AT RISK (95%) ===========")
print(f"Historical VaR        : {historical_var:.5f}")
print(f"Variance-Covariance   : {varcov_var:.5f}")
print(f"Monte Carlo VaR       : {montecarlo_var:.5f}")
print("===========================================")


# ================================================================
# 6. VISUALISE LOSS DISTRIBUTION + VAR LINES
# ================================================================
plt.figure(figsize=(10, 6))

plt.hist(losses, bins=40, alpha=0.6, label="Historical Losses")

plt.axvline(
    historical_var,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Historical VaR = {historical_var:.3f}"
)

plt.axvline(
    varcov_var,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Var-Cov VaR = {varcov_var:.3f}"
)

plt.axvline(
    montecarlo_var,
    color="purple",
    linestyle="--",
    linewidth=2,
    label=f"Monte Carlo VaR = {montecarlo_var:.3f}"
)

plt.title("Loss Distribution with 95% Value at Risk Estimates")
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()
