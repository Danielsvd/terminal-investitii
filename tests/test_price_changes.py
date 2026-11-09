"""Cardurile Interval / Evoluție Azi: lipsa dă None (N/A), nu 0."""
import numpy as np
import pandas as pd
import pytest

from analytics.technical import price_changes


def test_valori_de_referinta():
    full = pd.Series([90.0, 100.0, 104.0, 110.0])
    r = price_changes(full.iloc[1:], full)          # intervalul începe la 100
    assert r["last"] == 110.0
    assert r["diff"] == pytest.approx(10.0) and r["diff_pct"] == pytest.approx(10.0)
    assert r["day"] == pytest.approx(6.0) and r["day_pct"] == pytest.approx(6 / 104 * 100)


def test_interval_gol_toate_none():
    r = price_changes(pd.Series(dtype="float64"), pd.Series([1.0, 2.0]))
    assert r == {"last": None, "diff": None, "diff_pct": None, "day": None, "day_pct": None}


def test_o_singura_sedinta():
    r = price_changes(pd.Series([50.0]), pd.Series([50.0]))
    assert r["last"] == 50.0
    assert r["diff"] is None and r["day"] is None


def test_nan_la_final_se_ignora():
    full = pd.Series([100.0, 105.0, np.nan])
    r = price_changes(full, full)
    assert r["last"] == 105.0 and r["day_pct"] == pytest.approx(5.0)
