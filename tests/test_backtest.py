"""Teste pentru analytics/backtest.py. Valori sintetice, calculate de mână."""
import pandas as pd
import pytest

from analytics.backtest import (add_excess, bucket_summary, dividend_flags, forward_return,
                                month_end_dates, rank_ic, score_rows)
from analytics.scoring import technical_subscore
from analytics.technical import rsi_wilder


def test_ultima_sedinta_din_luna():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    got = month_end_dates(idx, "2024-01-15", "2024-03-31")
    assert [d.date().isoformat() for d in got] == ["2024-01-31", "2024-02-29", "2024-03-29"]


def _step_series():
    idx = pd.bdate_range("2024-01-01", "2024-06-28")
    s = pd.Series(100.0, index=idx)
    s[s.index > "2024-01-31"] = 110.0
    return s


def test_randament_viitor_pe_luni_calendaristice():
    s = _step_series()
    assert forward_return(s, "2024-01-31", 1) == pytest.approx(0.10)    # 100 → 110
    assert forward_return(s, "2024-02-29", 3) == pytest.approx(0.0)     # 110 → 110
    assert forward_return(s, "2024-06-28", 1) is None                   # orizont după ultima dată


def test_dividend_detectat_din_raportul_ajustat_neajustat():
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    raw = pd.Series(100.0, index=idx)
    adj = raw.copy()
    adj[adj.index < "2024-02-15"] = 95.0       # raport 0,95 → 1,00: salt de 5,26% pe 2024-02-15
    flags = dividend_flags(adj, raw, window=20)
    assert not flags.loc["2024-02-14"]
    assert flags.loc["2024-02-15"]
    # Fereastra de 20 de ședințe: 15–29 feb. (11 ședințe) + 1–13 mar. (9) → ultima zi marcată 13 martie.
    assert flags.loc["2024-03-13"]
    assert not flags.loc["2024-03-14"]
    assert not flags.loc["2024-04-30"]


def _panel():
    rows = []
    for d in ("2024-01-31", "2024-02-29"):
        for i, (score, ret) in enumerate(zip((10, 30, 50, 70, 90), (0.01, 0.02, 0.03, 0.04, 0.05))):
            rows.append({"symbol": f"S{i}", "date": d, "tech_score": score, "fwd_3m": ret})
    return pd.DataFrame(rows)


def test_exces_fata_de_media_universului():
    df = _panel()
    ex = add_excess(df, "fwd_3m")["fwd_3m_ex"]
    # Media pe dată = 0,03 → excese −0,02 … +0,02
    assert list(ex.round(4))[:5] == [-0.02, -0.01, 0.0, 0.01, 0.02]
    small = df[df["symbol"].isin(["S0", "S1", "S2"])]
    assert add_excess(small, "fwd_3m")["fwd_3m_ex"].isna().all()       # sub 5 simboluri pe dată


def test_tabel_pe_trepte():
    t = bucket_summary(_panel(), "tech_score", "fwd_3m")
    assert list(t.index.astype(str)) == ["0–20", "20–40", "40–60", "60–80", "80–100"]
    assert t.loc["0–20", "N"] == 2 and t.loc["0–20", "exces_mediu"] == pytest.approx(-0.02)
    assert t.loc["0–20", "pct_pozitiv"] == 0.0
    assert t.loc["80–100", "exces_mediu"] == pytest.approx(0.02)
    assert t.loc["80–100", "pct_pozitiv"] == 1.0 and t.loc["80–100", "nr_date"] == 2


def test_ic_de_rang():
    df = _panel()
    third = df[df["date"] == "2024-01-31"].copy()
    third["date"] = "2024-03-29"
    third["fwd_3m"] = third["fwd_3m"].values[::-1]        # ordine inversă → IC = −1
    r = rank_ic(pd.concat([df, third]), "tech_score", "fwd_3m")
    # IC pe date: 1, 1, −1 → medie 1/3; abatere (ddof=1) = √(4/3) = 1,1547; t = 0,3333 / 1,1547 × √3 = 0,5
    assert r["nr_date"] == 3
    assert r["ic_mediu"] == pytest.approx(1 / 3)
    assert r["ic_std"] == pytest.approx((4 / 3) ** 0.5)
    assert r["t"] == pytest.approx(0.5)
    assert r["pct_pozitiv"] == pytest.approx(2 / 3)


def test_randuri_pe_simbol_folosesc_doar_trecutul():
    idx = pd.bdate_range("2023-01-02", periods=400)
    close = pd.Series(range(100, 500), index=idx, dtype="float64")
    vol = pd.Series(1000.0, index=idx)
    vol.iloc[-10:] = 0.0
    dates = month_end_dates(idx, "2023-06-01", idx[-1])
    rows = score_rows("TEST", close, dates, close_raw=close, volume=vol)
    assert len(rows) == len(dates)
    first = rows[0]
    d = pd.Timestamp(first["date"])
    hist = close.loc[:d]
    assert first["tech_score"] == technical_subscore(hist, rsi_wilder(close, 14).loc[d])
    assert first["tech_score_unadj"] == first["tech_score"] and first["dividend_20d"] is False
    assert first["fwd_1m"] == pytest.approx(forward_return(close, d, 1))
    assert rows[-1]["fwd_1m"] is None                     # ultima lună nu are viitor
    assert rows[-1]["zero_vol_frac_3m"] == pytest.approx(10 / len(vol.loc[pd.Timestamp(rows[-1]["date"]) - pd.DateOffset(months=3):rows[-1]["date"]]))


def test_simbol_inactiv_e_sarit():
    idx = pd.bdate_range("2024-01-01", "2024-02-28")
    close = pd.Series(100.0, index=idx)
    rows = score_rows("X", close, [pd.Timestamp("2024-05-31")])
    assert rows == []
