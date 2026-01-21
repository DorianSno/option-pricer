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

## Potential Extensions and Use Cases

This pricing engine is intended as a foundational component rather than a standalone trading system. In a larger quantitative workflow, similar Monte Carlo engines are typically used as building blocks for:

- Pricing and risk evaluation of derivatives that lack closed-form solutions (e.g., path-dependent or exotic options)
- Computing Greeks used for hedging and risk management within a trading or portfolio system
- Scenario analysis and stress testing under different volatility or rate assumptions
- Integration into a broader trading framework where pricing, risk, and execution are handled by separate components

For example, in a future trading or research project, this engine will be extended to support path-dependent payoffs, calibrated model parameters, or portfolio-level aggregation, while remaining decoupled from signal generation and execution logic.
