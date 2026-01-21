from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional

import numpy as np


# ----------------------------
# Black-Scholes closed form
# ----------------------------
def norm_cdf(x: float) -> float:
    """Standard normal CDF without SciPy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S0: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Require S0, K, T, sigma > 0")
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == "put":
        return K * math.exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# ----------------------------
# Monte Carlo core
# ----------------------------
def simulate_terminal_prices(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    Z: np.ndarray,
) -> np.ndarray:
    """Simulate S_T given standard normals Z under risk-neutral GBM."""
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)


def discounted_payoff(ST: np.ndarray, K: float, T: float, r: float, option_type: str) -> np.ndarray:
    disc = math.exp(-r * T)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return disc * payoff


@dataclass(frozen=True)
class MCResult:
    price: float
    se: float
    ci_low: float
    ci_high: float
    method: str


def mean_se_ci(samples: np.ndarray, z: float = 1.96) -> Tuple[float, float, float, float]:
    """Return mean, standard error, and CI."""
    mean = float(samples.mean())
    se = float(samples.std(ddof=1) / math.sqrt(samples.size))
    return mean, se, mean - z * se, mean + z * se


def mc_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    n_sims: int,
    seed: int,
    antithetic: bool = True,
    control_variate: bool = True,
) -> MCResult:
    """
    Monte Carlo price with optional variance reduction.

    Control variate uses S_T with known expectation:
      E[S_T] = S0 * exp(rT) under risk-neutral measure.

    If both antithetic and control variate are enabled, both are applied.
    """
    if n_sims <= 1:
        raise ValueError("n_sims must be > 1")

    rng = np.random.default_rng(seed)

    # Antithetic: use Z and -Z, effectively doubling samples while preserving budget interpretation.
    if antithetic:
        half = (n_sims + 1) // 2
        Z_half = rng.standard_normal(half)
        Z = np.concatenate([Z_half, -Z_half])[:n_sims]
        method = "MC + Antithetic"
    else:
        Z = rng.standard_normal(n_sims)
        method = "MC"

    ST = simulate_terminal_prices(S0, T, r, sigma, Z)
    Y = discounted_payoff(ST, K, T, r, option_type)  # discounted payoff samples

    if control_variate:
        # Control variate variable X = discounted S_T, whose expectation is S0 (since E[S_T] = S0 e^{rT})
        X = math.exp(-r * T) * ST
        EX = S0

        # Optimal coefficient b* = Cov(Y, X) / Var(X)
        cov_yx = float(np.cov(Y, X, ddof=1)[0, 1])
        var_x = float(np.var(X, ddof=1))
        b = cov_yx / var_x if var_x > 0 else 0.0

        Y_cv = Y - b * (X - EX)
        price, se, lo, hi = mean_se_ci(Y_cv)
        method += " + ControlVariate(S_T)"
        return MCResult(price=price, se=se, ci_low=lo, ci_high=hi, method=method)

    price, se, lo, hi = mean_se_ci(Y)
    return MCResult(price=price, se=se, ci_low=lo, ci_high=hi, method=method)


# ----------------------------
# Greeks via CRN finite differences
# ----------------------------
@dataclass(frozen=True)
class Greeks:
    delta: float
    vega: float


def mc_delta_vega_crn(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    n_sims: int,
    seed: int,
    bump_S_frac: float = 0.01,
    bump_sigma_abs: float = 0.01,
    antithetic: bool = True,
    control_variate: bool = True,
) -> Greeks:
    """
    Central-difference Delta and Vega using common random numbers (CRN).
    We reuse the same Z across bumps to reduce estimator noise.
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        half = (n_sims + 1) // 2
        Z_half = rng.standard_normal(half)
        Z = np.concatenate([Z_half, -Z_half])[:n_sims]
    else:
        Z = rng.standard_normal(n_sims)

    def price_given(S0_local: float, sigma_local: float) -> float:
        ST = simulate_terminal_prices(S0_local, T, r, sigma_local, Z)
        Y = discounted_payoff(ST, K, T, r, option_type)

        if control_variate:
            X = math.exp(-r * T) * ST
            EX = S0_local
            cov_yx = float(np.cov(Y, X, ddof=1)[0, 1])
            var_x = float(np.var(X, ddof=1))
            b = cov_yx / var_x if var_x > 0 else 0.0
            Y = Y - b * (X - EX)

        return float(Y.mean())

    # Delta
    S_up = S0 * (1 + bump_S_frac)
    S_dn = S0 * (1 - bump_S_frac)
    p_up = price_given(S_up, sigma)
    p_dn = price_given(S_dn, sigma)
    delta = (p_up - p_dn) / (S_up - S_dn)

    # Vega
    sig_up = sigma + bump_sigma_abs
    sig_dn = sigma - bump_sigma_abs
    if sig_dn <= 0:
        raise ValueError("sigma bump makes sigma <= 0; reduce bump_sigma_abs")
    v_up = price_given(S0, sig_up)
    v_dn = price_given(S0, sig_dn)
    vega = (v_up - v_dn) / (sig_up - sig_dn)

    return Greeks(delta=float(delta), vega=float(vega))


# ----------------------------
# Convergence study
# ----------------------------
def convergence_study(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    n_list: Iterable[int],
    seed: int,
    antithetic: bool,
    control_variate: bool,
) -> Dict[int, MCResult]:
    out: Dict[int, MCResult] = {}
    for n in n_list:
        out[int(n)] = mc_price(
            S0, K, T, r, sigma, option_type,
            n_sims=int(n), seed=seed,
            antithetic=antithetic, control_variate=control_variate
        )
    return out


def print_table(results: Dict[int, MCResult], bs_price: float) -> None:
    print("N        Price        SE           95% CI Low    95% CI High   |Price-BS|")
    print("-" * 79)
    for n in sorted(results.keys()):
        res = results[n]
        diff = abs(res.price - bs_price)
        print(f"{n:<8d} {res.price:<12.6f} {res.se:<12.6f} {res.ci_low:<12.6f} {res.ci_high:<12.6f} {diff:.6f}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo European option pricer with variance reduction and Greeks.")
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--K", type=float, default=105.0)
    p.add_argument("--T_days", type=float, default=30.0, help="Time to maturity in days.")
    p.add_argument("--r", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--type", type=str, default="call", choices=["call", "put"])
    p.add_argument("--n", type=int, default=300_000, help="Number of Monte Carlo simulations.")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--no-antithetic", action="store_true", help="Disable antithetic variates.")
    p.add_argument("--no-cv", action="store_true", help="Disable control variate.")
    p.add_argument("--bumpS", type=float, default=0.01, help="Relative bump for S0 in Delta.")
    p.add_argument("--bumpSigma", type=float, default=0.01, help="Absolute bump for sigma in Vega.")

    p.add_argument("--study", action="store_true", help="Run a convergence study across multiple N.")
    p.add_argument("--studyN", type=str, default="10000,50000,100000,300000", help="Comma-separated N values.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    S0 = args.S0
    K = args.K
    T = args.T_days / 365.0
    r = args.r
    sigma = args.sigma
    option_type = args.type

    antithetic = not args.no_antithetic
    control_variate = not args.no_cv

    print("Parameters:")
    print(f"  S0={S0}, K={K}, T={T:.6f} yrs ({args.T_days} days), r={r}, sigma={sigma}, type={option_type}")
    print(f"  Variance reduction: antithetic={antithetic}, control_variate={control_variate}\n")

    bs = black_scholes_price(S0, K, T, r, sigma, option_type)
    print(f"Black-Scholes price: {bs:.6f}\n")

    if args.study:
        n_list = [int(x.strip()) for x in args.studyN.split(",") if x.strip()]
        results = convergence_study(S0, K, T, r, sigma, option_type, n_list, args.seed, antithetic, control_variate)
        print_table(results, bs)
        print("\n(Expect SE to shrink roughly like 1/sqrt(N). Control variate should materially reduce SE.)\n")

    res = mc_price(S0, K, T, r, sigma, option_type, args.n, args.seed, antithetic, control_variate)
    diff = abs(res.price - bs)
    print("Single-run estimate:")
    print(f"  Method: {res.method}")
    print(f"  MC price: {res.price:.6f}")
    print(f"  SE:       {res.se:.6f}")
    print(f"  95% CI:   [{res.ci_low:.6f}, {res.ci_high:.6f}]")
    print(f"  |MC-BS|:  {diff:.6f}\n")

    greeks = mc_delta_vega_crn(
        S0, K, T, r, sigma, option_type,
        n_sims=args.n, seed=args.seed,
        bump_S_frac=args.bumpS, bump_sigma_abs=args.bumpSigma,
        antithetic=antithetic, control_variate=control_variate
    )
    print("Greeks (finite differences with CRN):")
    print(f"  Delta: {greeks.delta:.6f}")
    print(f"  Vega:  {greeks.vega:.6f}")


if __name__ == "__main__":
    main()
