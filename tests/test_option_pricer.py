import math
import numpy as np
import pytest

import option_pricer as pricer


def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bs_delta_vega(S0: float, K: float, T: float, r: float, sigma: float, option_type: str):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if option_type == "call":
        delta = pricer.norm_cdf(d1)
    elif option_type == "put":
        delta = pricer.norm_cdf(d1) - 1.0
    else:
        raise ValueError("option_type must be call/put")
    vega = S0 * norm_pdf(d1) * math.sqrt(T)
    return delta, vega


def test_black_scholes_put_call_parity():
    S0, K, T, r, sigma = 100.0, 105.0, 30.0 / 365.0, 0.05, 0.2
    c = pricer.black_scholes_price(S0, K, T, r, sigma, "call")
    p = pricer.black_scholes_price(S0, K, T, r, sigma, "put")
    # Put-call parity: C - P = S0 - K e^{-rT}
    rhs = S0 - K * math.exp(-r * T)
    assert abs((c - p) - rhs) < 1e-10


def test_mc_price_ci_contains_bs_with_variance_reduction():
    S0, K, T, r, sigma, typ = 100.0, 105.0, 30.0 / 365.0, 0.05, 0.2, "call"
    bs = pricer.black_scholes_price(S0, K, T, r, sigma, typ)
    res = pricer.mc_price(S0, K, T, r, sigma, typ, n_sims=50_000, seed=123, antithetic=True, control_variate=True)
    assert res.ci_low <= bs <= res.ci_high


def test_control_variate_reduces_standard_error():
    S0, K, T, r, sigma, typ = 100.0, 105.0, 30.0 / 365.0, 0.05, 0.2, "call"
    base = pricer.mc_price(S0, K, T, r, sigma, typ, n_sims=60_000, seed=7, antithetic=True, control_variate=False)
    cv = pricer.mc_price(S0, K, T, r, sigma, typ, n_sims=60_000, seed=7, antithetic=True, control_variate=True)
    assert cv.se < base.se


def test_mc_greeks_close_to_black_scholes():
    # Loose tolerances to keep tests stable and fast.
    S0, K, T, r, sigma, typ = 100.0, 105.0, 30.0 / 365.0, 0.05, 0.2, "call"
    delta_bs, vega_bs = bs_delta_vega(S0, K, T, r, sigma, typ)

    greeks = pricer.mc_delta_vega_crn(
        S0, K, T, r, sigma, typ,
        n_sims=120_000, seed=123,
        bump_S_frac=0.01, bump_sigma_abs=0.01,
        antithetic=True, control_variate=True,
    )

    assert abs(greeks.delta - delta_bs) < 0.02
    assert abs(greeks.vega - vega_bs) < 0.8


def test_invalid_inputs():
    with pytest.raises(ValueError):
        pricer.black_scholes_price(0.0, 100.0, 1.0, 0.05, 0.2, "call")
    with pytest.raises(ValueError):
        pricer.mc_price(100.0, 100.0, 1.0, 0.05, 0.2, "call", n_sims=1, seed=0)
    with pytest.raises(ValueError):
        pricer.mc_delta_vega_crn(100.0, 100.0, 1.0, 0.05, 0.001, "call", n_sims=10_000, seed=0, bump_sigma_abs=0.01)
