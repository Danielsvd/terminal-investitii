"""Stresul pieței: structura VIX și spread-urile de credit."""
import numpy as np
import pandas as pd
import pytest

from analytics.stress import percentile_rank, spread_stats, vix_term


def test_structura_vix_pe_ziua_comuna():
    v = pd.Series([18.0, 30.0], index=pd.to_datetime(["2026-11-03", "2026-11-04"]))
    v3 = pd.Series([20.0, 27.0, 26.0], index=pd.to_datetime(["2026-11-03", "2026-11-04", "2026-11-05"]))
    t = vix_term(v, v3)
    assert t["date"] == pd.Timestamp("2026-11-04")                 # ultima zi comună, nu 05.11
    assert t["ratio"] == pytest.approx(30 / 27) and t["state"] == "stres"
    assert vix_term(v.iloc[:1], v3)["state"] == "tensiune"           # 18 / 20 = 0,90
    assert vix_term(pd.Series([15.0], index=[pd.Timestamp("2026-11-03")]), v3)["state"] == "normal"
    assert vix_term(v, pd.Series(dtype=float)) is None


def test_percentila():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert percentile_rank(s) == 100.0
    assert percentile_rank(s, 2.0) == 50.0
    assert percentile_rank(pd.Series(dtype=float)) is None


def test_spread_credit():
    idx = pd.bdate_range("2025-10-01", "2026-11-04")
    s = pd.Series(np.linspace(3.0, 4.0, len(idx)), index=idx)
    st = spread_stats(s)
    assert st["value"] == pytest.approx(4.0) and st["date"] == idx[-1] and st["n"] == len(idx)
    assert st["d1m"] == pytest.approx(4.0 - s[s.index <= pd.Timestamp("2026-10-04")].iloc[-1])
    assert st["d1y"] == pytest.approx(4.0 - s[s.index <= pd.Timestamp("2025-11-04")].iloc[-1])
    assert st["pctile"] == 100.0
    short = spread_stats(s.loc["2026-09-15":])
    assert short["d1m"] is not None and short["d1y"] is None
    assert spread_stats(None)["value"] is None
