# Value at Risk (VaR) Comparison

This project compares three common methods for estimating 95% Value at Risk (VaR) using Python:

* Historical VaR
* Variance-Covariance VaR
* Monte Carlo VaR

The script simulates one year of daily returns using typical long-term S&P 500 assumptions, converts returns into losses, calculates VaR using each method, and visualises the loss distribution with VaR thresholds.

## Project Overview

Value at Risk is a common financial risk measure used to estimate the potential loss of a portfolio over a given time horizon at a chosen confidence level.

In this project, I estimate daily 95% VaR using three different approaches:

1. Historical simulation
2. Variance-covariance method
3. Monte Carlo simulation

The aim is to compare how different VaR methods can produce different risk estimates depending on their assumptions.

## What the Code Does

The script:

* Simulates 252 daily returns
* Uses an annual expected return of 7%
* Uses an annual volatility of 15%
* Converts daily returns into daily losses
* Calculates 95% VaR using three methods
* Prints the VaR estimates
* Plots the loss distribution with VaR threshold lines

## Methods Used

### Historical VaR

Historical VaR is calculated as the 95th percentile of the loss distribution.

This method estimates risk directly from the observed or simulated loss data.

### Variance-Covariance VaR

The variance-covariance method assumes returns are normally distributed.

It estimates VaR using the mean return, standard deviation, and a one-sided 95% z-score.

This method is simple and fast, but it depends heavily on the normality assumption.

### Monte Carlo VaR

Monte Carlo VaR simulates many possible daily return outcomes.

The simulated returns are converted into losses, and the 95th percentile of those losses is used as the VaR estimate.

## Technologies Used

* Python
* NumPy
* Matplotlib

## Example Parameters

The current script uses the following assumptions:

* Trading days: 252
* Annual expected return: 7%
* Annual volatility: 15%
* Confidence level: 95%
* Monte Carlo simulations: 10,000

## Example Output

The script prints output similar to:

```text
=========== VALUE AT RISK (95%) ===========
Historical VaR        : ...
Variance-Covariance   : ...
Monte Carlo VaR       : ...
===========================================
```

The script also produces a histogram of simulated daily losses with vertical lines showing the VaR estimate from each method.

## Financial Interpretation

A 95% daily VaR estimates the loss level that is expected to be exceeded only 5% of the time.

For example, if the 95% daily VaR is 0.016, this means the model estimates a 5% chance of losing more than 1.6% in a single day.

Comparing the three methods helps show how different modelling assumptions affect risk estimates.

## Limitations

This is a simplified educational project and should not be treated as a real risk management system.

Important limitations include:

* Uses simulated returns rather than real market data
* Assumes constant expected return and volatility
* Assumes normally distributed returns
* Does not capture fat tails or extreme market events well
* Ignores portfolio composition and asset correlations
* Ignores liquidity, transaction costs, and market impact
* Models a simple return process rather than a full multi-asset portfolio

## Possible Improvements

Future extensions could include:

* Using real historical S&P 500 data
* Comparing VaR at different confidence levels
* Adding Expected Shortfall
* Extending the model to a multi-asset portfolio
* Including correlations between assets
* Backtesting the VaR estimates
* Testing heavier-tailed return distributions
* Building an interactive dashboard using Streamlit

## Purpose

This project was created as a beginner quantitative finance project to practise:

* Value at Risk methodology
* financial risk measurement
* Monte Carlo simulation
* Python programming
* data visualisation
* NumPy and Matplotlib
