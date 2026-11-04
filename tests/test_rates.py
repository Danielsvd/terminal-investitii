"""Dobânzi comparate: serii lunare/zilnice, spread pe aceeași lună, inflație an/an, variații în puncte procentuale."""
import numpy as np
import pandas as pd
import pytest

from analytics.rates import daily_diff, monthly_diff, period_label, rate_stats, to_monthly, yoy_monthly


def test_lunara_din_zilnica_ia_sfarsitul_lunii():
    d = pd.Series([4.0, 4.2, 4.5], index=pd.to_datetime(["2026-09-02", "2026-09-30", "2026-10-15"]))
    m = to_monthly(d)
    assert m.loc[pd.Period("2026-09", "M")] == 4.2 and m.loc[pd.Period("2026-10", "M")] == 4.5


def test_spread_ro_de_pe_aceeasi_luna():
    ro10 = pd.Series({"2026-07": 7.20, "2026-08": 7.10, "2026-09": 7.00})
    de10 = pd.Series({"2026-08": 2.70, "2026-09": 2.75, "2026-10": 2.80})
    sp = monthly_diff(ro10, de10)
    assert list(sp.index.astype(str)) == ["2026-08", "2026-09"]       # doar lunile comune
    assert sp.iloc[-1] == pytest.approx(4.25)                          # 7,00 − 2,75, nu 7,00 − 2,80


def test_inflatie_an_pe_an_din_indice():
    idx = pd.period_range("2025-01", "2026-03", freq="M").astype(str)
    lvl = pd.Series(np.linspace(100, 114, len(idx)), index=idx)
    y = yoy_monthly(lvl)
    assert str(y.index[0]) == "2026-01" and len(y) == 3
    assert y.iloc[-1] == pytest.approx((lvl["2026-03"] / lvl["2025-03"] - 1) * 100)


def test_inflatie_fara_luna_de_baza():
    lvl = pd.Series({"2025-01": 100.0, "2026-02": 105.0})
    assert yoy_monthly(lvl).empty


def test_variatii_lunare():
    s = to_monthly(pd.Series({"2025-09": 6.0, "2026-06": 6.5, "2026-09": 7.0}))
    st = rate_stats(s)
    assert st["value"] == 7.0 and st["monthly"] is True and period_label(st["period"]) == "09.2026"
    assert st["d3m"] == pytest.approx(0.5) and st["d1y"] == pytest.approx(1.0)
    st2 = rate_stats(to_monthly(pd.Series({"2026-07": 6.0, "2026-09": 7.0})))
    assert st2["d3m"] is None and st2["d1y"] is None                    # luna de comparat lipsește


def test_variatii_zilnice():
    idx = pd.bdate_range("2025-10-01", "2026-11-04")
    s = pd.Series(np.linspace(4.0, 4.6, len(idx)), index=idx)
    st = rate_stats(s)
    assert st["monthly"] is False and period_label(st["period"]) == "04.11.2026"
    ref = s[s.index <= pd.Timestamp("2026-08-04")].iloc[-1]
    assert st["d3m"] == pytest.approx(4.6 - ref)
    ref1 = s[s.index <= pd.Timestamp("2025-11-04")].iloc[-1]
    assert st["d1y"] == pytest.approx(4.6 - ref1)
    assert rate_stats(s.loc["2026-09-01":])["d1y"] is None


def test_spread_zilnic():
    a = pd.Series([4.1, 4.2], index=pd.to_datetime(["2026-11-03", "2026-11-04"]))
    b = pd.Series([3.6, 3.5], index=pd.to_datetime(["2026-11-02", "2026-11-04"]))
    d = daily_diff(a, b)
    assert len(d) == 1 and d.iloc[0] == pytest.approx(0.7)


def test_goale():
    assert rate_stats(pd.Series(dtype=float))["value"] is None
    assert rate_stats(None)["value"] is None
    assert monthly_diff(None, pd.Series({"2026-01": 1.0})).empty
    assert period_label(None) is None
