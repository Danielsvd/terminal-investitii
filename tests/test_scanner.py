import math

from analytics.scanner import MIN_AVG_SHARES, RVOL_DEFAULT, liquidity_ok


def test_prag_bvb_in_lei_ca_in_rezumat():
    assert liquidity_ok("TBM.RO", 0.31, 178_264, 99_839)            # 55.262 lei ≥ 50.000
    assert not liquidity_ok("RMAH.RO", 0.81, 48_043, 23_903)        # 38.915 lei < 50.000
    assert not liquidity_ok("SPX.RO", 0.29, 26_936, 6_744)          # 7.811 lei
    assert liquidity_ok("EBS.RO", 611.0, 500, 300)                  # 305.500 lei, deși sub 5.000 de acțiuni
    assert liquidity_ok("X.RO", 1.0, 50_000, 0)                     # exact pragul intră
    assert not liquidity_ok("X.RO", None, 50_000, 0) and not liquidity_ok("X.RO", 1.0, math.nan, 0)


def test_sua_si_europa_pastreaza_regula_in_actiuni():
    assert liquidity_ok("AAPL", 200.0, 1, MIN_AVG_SHARES)
    assert not liquidity_ok("SAP.DE", 200.0, 10**6, MIN_AVG_SHARES - 1)
    assert not liquidity_ok("AAPL", 200.0, 1, None)
    assert RVOL_DEFAULT == 1.2
