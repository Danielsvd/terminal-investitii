"""Istoricul pantei 10 ani − 3 luni (graficul din „Dobânzi și risc de țară”)."""
import pandas as pd

from analytics.rates import curve_inversion_summary, daily_diff


def test_rezumat_inversare():
    idx = pd.to_datetime(["2025-06-02", "2025-11-03", "2025-11-04", "2026-05-04", "2026-11-03", "2026-11-04"])
    s = pd.Series([-0.5, -0.2, 0.1, -0.1, 0.3, 0.4], index=idx)
    r = curve_inversion_summary(s)
    assert r["value"] == 0.4 and r["date"] == pd.Timestamp("2026-11-04")
    # fereastra = după 04.11.2025: 04.05.2026, 03.11, 04.11 → 3 ședințe, una inversată (04.11.2025 e exclusă)
    assert r["n_window"] == 3 and r["inverted_window"] == 1
    assert r["last_inverted"] == pd.Timestamp("2026-05-04")
    assert r["start"] == pd.Timestamp("2025-06-02")


def test_fara_inversare_si_serie_goala():
    s = pd.Series([0.5, 0.6], index=pd.to_datetime(["2026-11-03", "2026-11-04"]))
    assert curve_inversion_summary(s)["last_inverted"] is None
    assert curve_inversion_summary(pd.Series(dtype=float)) is None
    assert curve_inversion_summary(None) is None


def test_panta_doar_pe_zilele_comune():
    t10 = pd.Series([4.4, 4.5, 4.6], index=pd.to_datetime(["2026-11-02", "2026-11-03", "2026-11-04"]))
    t3 = pd.Series([4.2, 4.7], index=pd.to_datetime(["2026-11-02", "2026-11-04"]))
    d = daily_diff(t10, t3)
    assert list(d.index) == list(pd.to_datetime(["2026-11-02", "2026-11-04"]))   # 03.11 lipsește la 3 luni: nu se umple
    assert round(d.iloc[0], 6) == 0.2 and round(d.iloc[-1], 6) == -0.1
