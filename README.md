# Monte Carlo Option Pricing Engine

This project implements a Monte Carlo pricing engine for European options under geometric Brownian motion, with a focus on numerical stability, reproducibility, and validation.

The goal is not to build a trading strategy, but to model how pricing and risk calculations are performed in quantitative systems when closed-form solutions are unavailable or insufficient.

## Overview

The engine prices European call and put options using risk-neutral Monte Carlo simulation and reports both price estimates and statistical uncertainty. It also computes first-order risk sensitivities (Greeks) needed for hedging and validation.

## Design Highlights

- Monte Carlo pricing under the risk-neutral GBM model  
- Variance reduction using antithetic variates and a control variate based on the terminal stock price  
- Standard error and confidence interval reporting for all price estimates  
- Delta and Vega estimation via finite differences with common random numbers to reduce noise  
- Validation against Black–Scholes closed-form prices  
- Command-line interface to support reproducible experiments and convergence studies

## Notes

- Variance reduction techniques are used to improve estimator efficiency rather than increase simulation count.
- Greek estimation prioritizes stability over raw speed.
- Closed-form pricing is used only for validation; the Monte Carlo engine is designed to extend to products without analytic solutions.

## Potential Trading Extensions

This pricing engine is intended as a foundational component rather than a standalone trading system. The next steps that I am currently working on are:

- Compare model prices to observed market prices and trade only when discrepancies exceed statistical uncertainty.  
- Use Monte Carlo–derived Greeks for hedging, position sizing, and risk management in a trading framework.

## Usage with Given Parameters

The pricing engine is designed to be called with explicit option and market parameters, similar to how pricing components are used inside larger quantitative systems.

Example: pricing a European call option with  
S₀ = 102, K = 100, T = 45 days, r = 3%, σ = 25%

```bash
python option_pricer.py \
  --S0 102 \
  --K 100 \
  --T_days 45 \
  --r 0.03 \
  --sigma 0.25 \
  --type call
