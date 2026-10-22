import numpy as np
import pandas as pd

from analytics.options import max_pain


def _chain(strikes, oi):
    return pd.DataFrame({"strike": strikes, "openInterest": oi})


def test_max_pain_symmetric_chain():
    calls = _chain([90, 100, 110], [10, 20, 30])
    puts = _chain([90, 100, 110], [30, 20, 10])
    # K=90: call 0 + put (20*10 + 10*20) = 400
    # K=100: call 10*10 + put 10*10 = 200
    # K=110: call (10*20 + 20*10) + put 0 = 400
    assert max_pain(calls, puts) == 100.0


def test_max_pain_is_not_simply_the_middle_strike():
    calls = _chain([90], [100])
    puts = _chain([110], [1])
    # K=90: put 1*20 = 20 ; K=110: call 100*20 = 2000
    assert max_pain(calls, puts) == 90.0


def test_max_pain_ignores_missing_open_interest():
    calls = _chain([90, 100, 110], [np.nan, 20, 30])
    puts = _chain([90, 100, 110], [30, 20, np.nan])
    # K=90: put 20*10 = 200 ; K=100: 0 ; K=110: call 20*10 = 200
    assert max_pain(calls, puts) == 100.0


def test_max_pain_none_without_open_interest():
    assert max_pain(_chain([90, 100], [0, 0]), _chain([90, 100], [np.nan, 0])) is None
    assert max_pain(pd.DataFrame(columns=["strike", "openInterest"]), None) is None


# ---------------------------------------------------------------------------
# Lanțul de opțiuni: valori de referință calculate manual (vezi comentariile).
# Exemplu standard: S=100, K=100, T=0,25, r=5%, q=0, sigma=20%
#   d1 = (0 + (0,05 + 0,02) * 0,25) / (0,2 * 0,5) = 0,175 ; d2 = 0,075
#   N(d1) = 0,569460 ; N(d2) = 0,529893 ; phi(d1) = 0,392880
# ---------------------------------------------------------------------------
import math
from datetime import date

import pytest

from analytics.options import (
    atm_iv, bs_greeks, bs_price, choose_expiration, expected_move,
    gamma_exposure, iv_rank_percentile, put_call_ratio, realized_vol,
)


def test_choose_expiration_picks_closest_to_30_days_inside_window():
    today = date(2026, 10, 22)
    exps = ["2026-10-23", "2026-10-30", "2026-11-13", "2026-11-20", "2026-12-18"]
    # zile: 1, 8, 22, 29, 57 -> în [20, 45]: 22 și 29; 29 e cea mai apropiată de 30
    assert choose_expiration(exps, today) == ("2026-11-20", 29)


def test_choose_expiration_none_outside_window_and_tie_prefers_earlier():
    today = date(2026, 10, 22)
    assert choose_expiration(["2026-10-23", "2027-01-15"], today) is None
    assert choose_expiration([], today) is None
    # 26 și 34 de zile sunt la egală distanță de 30 -> cea mai apropiată în timp
    assert choose_expiration(["2026-11-25", "2026-11-17"], today) == ("2026-11-17", 26)


def test_bs_price_textbook_and_put_call_parity():
    c = bs_price(100, 100, 0.25, 0.05, 0.0, 0.2, "call")
    p = bs_price(100, 100, 0.25, 0.05, 0.0, 0.2, "put")
    assert c == pytest.approx(4.6150, abs=1e-4)
    # paritate: C - P = S - K e^{-rT} = 100 - 98,7578 = 1,2422
    assert c - p == pytest.approx(100 - 100 * math.exp(-0.0125), abs=1e-9)


def test_bs_greeks_reference_values():
    g = bs_greeks(100, 100, 0.25, 0.05, 0.0, 0.2, "call")
    assert g["delta"] == pytest.approx(0.569460, abs=1e-6)
    # gamma = phi(d1) / (S sigma sqrt T) = 0,392880 / 10
    assert g["gamma"] == pytest.approx(0.0392880, abs=1e-6)
    # vega la 1 pp = S phi(d1) sqrt T / 100 = 0,196440
    assert g["vega_pt"] == pytest.approx(0.196440, abs=1e-6)
    # theta = [-S phi sigma / (2 sqrt T) - r K e^{-rT} N(d2)] / 365 = (-7,8576 - 2,6166) / 365
    assert g["theta_day"] == pytest.approx(-0.028696, abs=1e-6)
    p = bs_greeks(100, 100, 0.25, 0.05, 0.0, 0.2, "put")
    assert p["delta"] == pytest.approx(0.569460 - 1, abs=1e-6)
    assert p["gamma"] == pytest.approx(g["gamma"])
    assert p["theta_day"] == pytest.approx(-0.015168, abs=1e-6)


def test_bs_greeks_none_on_missing_or_invalid_input():
    assert bs_greeks(None, 100, 0.25, 0.05, 0, 0.2, "call") is None
    assert bs_greeks(100, 100, 0.0, 0.05, 0, 0.2, "call") is None
    assert bs_greeks(100, 100, 0.25, 0.05, 0, float("nan"), "call") is None
    assert bs_greeks(100, 100, 0.25, 0.05, 0, 0.2, "straddle") is None


def _ivchain(strikes, ivs, bids=None, oi=None, vol=None):
    n = len(strikes)
    return pd.DataFrame({
        "strike": strikes, "impliedVolatility": ivs,
        "bid": bids if bids is not None else [1.0] * n,
        "openInterest": oi if oi is not None else [0] * n,
        "volume": vol if vol is not None else [0] * n,
    })


def test_atm_iv_interpolates_and_skips_unquoted_rows():
    # call: 95 -> 0,30 ; 100 fără bid (exclus) ; 105 -> 0,20  => la 101: 0,30 - 0,10*6/10 = 0,24
    calls = _ivchain([95, 100, 105], [0.30, 0.90, 0.20], bids=[1, 0, 1])
    # put: 100 -> 0,26 ; 105 -> 0,22 => la 101: 0,26 - 0,04*1/5 = 0,252
    puts = _ivchain([100, 105], [0.26, 0.22])
    iv, n = atm_iv(calls, puts, 101)
    assert n == 2
    assert iv == pytest.approx((0.24 + 0.252) / 2)


def test_atm_iv_ignores_placeholder_iv_and_one_side_only():
    calls = _ivchain([95, 105], [0.00001, 0.2])  # 95 are IV substituent -> spotul nu e încadrat
    puts = _ivchain([95, 105], [0.30, 0.20])
    iv, n = atm_iv(calls, puts, 100)
    assert n == 1 and iv == pytest.approx(0.25)
    assert atm_iv(calls, puts, None) == (None, 0)
    assert atm_iv(None, None, 100) == (None, 0)


def test_put_call_ratio_none_without_denominator():
    calls = _ivchain([100, 110], [0.2, 0.2], oi=[100, 100], vol=[0, np.nan])
    puts = _ivchain([90, 100], [0.2, 0.2], oi=[50, np.nan], vol=[30, 10])
    assert put_call_ratio(calls, puts, "openInterest") == pytest.approx(0.25)
    assert put_call_ratio(calls, puts, "volume") is None


def test_gamma_exposure_sign_and_reference():
    # 1 call ATM cu OI 10: gamma 0,039288 * 10 * 100 * 100^2 * 0,01 = 3.928,80
    calls = _ivchain([100], [0.2], oi=[10])
    gex, n = gamma_exposure(calls, None, 100, 0.25, 0.05, 0.0)
    assert n == 1 and gex == pytest.approx(3928.80, abs=0.01)
    # put identic cu OI 10 anulează exact (aceeași gamma, semn opus)
    puts = _ivchain([100], [0.2], oi=[10])
    gex2, n2 = gamma_exposure(calls, puts, 100, 0.25, 0.05, 0.0)
    assert n2 == 2 and gex2 == pytest.approx(0.0, abs=1e-9)
    assert gamma_exposure(_ivchain([100], [0.2], oi=[0]), None, 100, 0.25, 0.05, 0) == (None, 0)


def test_expected_move():
    # 100 * 0,20 * sqrt(30/365) = 5,7338
    assert expected_move(100, 0.20, 30) == pytest.approx(5.7338, abs=1e-4)
    assert expected_move(100, None, 30) is None


def test_realized_vol_alternating_returns():
    # randamente log alternând +a, -a: abaterea standard (ddof=1) a 20 de valori = a*sqrt(20/19)
    a = 0.01
    idx = pd.bdate_range("2026-09-01", periods=21)
    close = pd.Series(100 * np.exp(np.cumsum([0] + [a if i % 2 == 0 else -a for i in range(20)])), index=idx)
    hv = realized_vol(close, as_of=idx[-1], days=40)
    assert hv == pytest.approx(a * math.sqrt(20 / 19) * math.sqrt(252), rel=1e-9)
    # fereastra de 30 de zile calendaristice păstrează prea puține randamente -> None
    assert realized_vol(close.iloc[:10], days=30) is None


def test_iv_rank_percentile_reference_and_min_obs():
    idx = pd.date_range("2026-01-01", periods=25, freq="D")
    hist = pd.Series(np.arange(20, 45) / 100, index=idx)  # 0,20 ... 0,44 din 0,01 în 0,01
    r = iv_rank_percentile(hist, 0.30, as_of=idx[-1])
    # rank = (0,30 - 0,20) / (0,44 - 0,20) = 41,67% ; sub 0,30: 0,20..0,29 = 10 din 25 = 40%
    assert r["n"] == 25
    assert r["rank"] == pytest.approx(100 * 0.10 / 0.24)
    assert r["percentile"] == pytest.approx(40.0)
    short = iv_rank_percentile(hist.iloc[:10], 0.30, as_of=idx[9])
    assert short["n"] == 10 and short["rank"] is None and short["percentile"] is None


def test_iv_rank_uses_only_last_calendar_year():
    old = pd.Series([0.90], index=pd.to_datetime(["2024-12-01"]))
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    recent = pd.Series(np.arange(20, 40) / 100, index=idx)
    r = iv_rank_percentile(pd.concat([old, recent]), 0.39, as_of=idx[-1])
    # 0,90 din 2024-12 e în afara anului (fereastra începe la 2025-01-20) -> max = 0,39 -> rank 100
    assert r["n"] == 20 and r["rank"] == pytest.approx(100.0)
