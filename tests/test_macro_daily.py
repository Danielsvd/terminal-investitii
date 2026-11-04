"""„Indicatori Macroeconomici” / „Ghidul macro”: variații pe ultimele ședințe valide, lipsă = None."""
import numpy as np
import pandas as pd
import pytest

from analytics import macro as m


def _s(vals, start="2026-10-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def test_ultima_variatie_sare_peste_zilele_fara_valoare():
    s = _s([4.30, 4.40, np.nan])
    c = m.last_change(s)
    assert c["last"] == 4.40 and c["prev"] == 4.30
    assert c["abs"] == pytest.approx(0.10) and c["pct"] == pytest.approx((4.40 / 4.30 - 1) * 100)
    assert m.last_change(_s([4.3])) is None and m.last_change(None) is None


def test_variatie_pe_o_luna():
    idx = pd.to_datetime(["2026-09-30", "2026-10-03", "2026-11-04"])
    s = pd.Series([4.0, 4.1, 4.5], index=idx)
    assert m.change_over(s, 1) == pytest.approx(4.5 - 4.1)        # ≤ 04.10 = 03.10
    assert m.change_over(s.iloc[1:], 2) is None


def test_trend_dobanzi():
    assert m.yield_trend(0.30) == "creste" and m.yield_trend(-0.26) == "scade"
    assert m.yield_trend(0.25) == "stabil" and m.yield_trend(None) is None


def test_semnale_zilnice_lipsa_nu_e_zero():
    assert m.daily_cross_asset_signals({"usd": -0.5, "gold": 0.6}) == ["aur_refugiu"]
    # dolar lipsă: regula aurului nu se aplică (înainte lipsa era 0)
    assert m.daily_cross_asset_signals({"usd": None, "gold": 0.6}) == []
    assert m.daily_cross_asset_signals({}, tnx_pp=0.12) == ["dobanzi_cresc"]
    assert m.daily_cross_asset_signals({}, tnx_pp=-0.11) == ["dobanzi_scad"]
    assert m.daily_cross_asset_signals({}, tnx_pp=0.05) == []
    full = m.daily_cross_asset_signals({"usd": 0.6, "gold": 1.2, "oil": -0.6, "copper": 1.5}, tnx_pp=0.0, vix_pct=12.0)
    assert full == ["petrol_dolar", "cupru_expansiune", "fuga_siguranta"]
    assert "fuga_siguranta" not in m.daily_cross_asset_signals({"usd": 0.6, "gold": 1.2}, vix_pct=None)
