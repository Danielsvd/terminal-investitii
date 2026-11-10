"""Sub-portofoliul pe monedă din registru și TWR-ul lui (analytics/ledger.py). Valori calculate de mână."""
import numpy as np
import pandas as pd
import pytest

from analytics.ledger import COLUMNS, currency_sleeve, twr_in_out
from analytics.risk import beta_corr_weekly

IDX = pd.bdate_range("2026-01-05", "2026-01-09")      # luni–vineri


def _ledger(rows):
    df = pd.DataFrame([{c: r.get(c, np.nan if c in ("Quantity", "Price", "Fees", "Amount") else "") for c in COLUMNS}
                       for r in rows])
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ("Quantity", "Price", "Fees", "Amount"):
        df[c] = df[c].astype(float)
    return df


LEDGER = _ledger([
    {"Date": "2026-01-05", "Symbol": "AAA", "Class": "Actiuni SUA", "Type": "BUY", "Quantity": 10, "Price": 100, "Currency": "USD", "Amount": -1000},
    {"Date": "2026-01-07", "Symbol": "AAA", "Class": "Actiuni SUA", "Type": "DIV", "Currency": "USD", "Amount": 20},
    {"Date": "2026-01-08", "Symbol": "AAA", "Class": "Actiuni SUA", "Type": "SELL", "Quantity": 10, "Price": 121, "Currency": "USD", "Amount": 1210},
    # alt sub-portofoliu și o obligațiune: nu intră în cel în USD
    {"Date": "2026-01-05", "Symbol": "EEE.DE", "Class": "ETF", "Type": "BUY", "Quantity": 1, "Price": 50, "Currency": "EUR", "Amount": -50},
    {"Date": "2026-01-05", "Symbol": "BOND", "Class": "Obligatiuni", "Type": "BUY", "Quantity": 1, "Price": 100, "Currency": "USD", "Amount": -100},
    # numerar: nu intră
    {"Date": "2026-01-05", "Symbol": "", "Class": "", "Type": "DEPOSIT", "Currency": "USD", "Amount": 5000},
])
CLOSES = pd.DataFrame({"AAA": [100.0, 110.0, 110.0, 121.0, 121.0], "EEE.DE": [50.0] * 5}, index=IDX)


def test_valori_si_fluxuri_usd():
    values, ins, outs, sources, blockers = currency_sleeve(LEDGER, "USD", IDX, CLOSES)
    assert list(values) == [1000.0, 1100.0, 1100.0, 0.0, 0.0]
    assert list(ins) == [1000.0, 0.0, 0.0, 0.0, 0.0]
    assert list(outs) == [0.0, 0.0, 20.0, 1210.0, 0.0]
    assert sources == {"AAA": "market"} and blockers == []


def test_twr_cu_dividend_si_vanzare_totala():
    values, ins, outs, _, _ = currency_sleeve(LEDGER, "USD", IDX, CLOSES)
    daily, cumulative, total = twr_in_out(values, ins, outs)
    # zi 1: 1000/1000−1 = 0; zi 2: 1100/1000−1 = 10%; zi 3: (1100+20)/1100−1 = 1,8182%;
    # zi 4 (vânzare totală): 1210/1100−1 = 10% (nu −100%); zi 5: nimic deținut → 0
    assert daily.round(6).tolist() == [0.0, 0.1, round(20 / 1100, 6), 0.1, 0.0]
    assert total == pytest.approx(1.1 * (1120 / 1100) * 1.1 - 1)        # 23,2%


def test_inainte_de_prima_detinere_lipseste_valoarea():
    ledger = _ledger([{"Date": "2026-01-07", "Symbol": "AAA", "Class": "Actiuni SUA", "Type": "BUY",
                       "Quantity": 1, "Price": 110, "Currency": "USD", "Amount": -110}])
    values, ins, outs, _, _ = currency_sleeve(ledger, "USD", IDX, CLOSES)
    assert values.iloc[:2].isna().all() and list(values.iloc[2:]) == [110.0, 121.0, 121.0]
    daily, _, total = twr_in_out(values, ins, outs)
    assert len(daily) == 3 and total == pytest.approx(0.1)


def test_sold_deschidere_fara_amount():
    ledger = _ledger([{"Date": "2026-01-05", "Symbol": "AAA", "Class": "Actiuni SUA", "Type": "OPEN",
                       "Quantity": 2, "Price": 100, "Currency": "USD"}])
    _, ins, _, _, _ = currency_sleeve(ledger, "USD", IDX, CLOSES)
    assert ins.iloc[0] == 200.0


def test_moneda_fara_pozitii():
    values, ins, outs, sources, blockers = currency_sleeve(LEDGER, "RON", IDX, CLOSES)
    assert values.isna().all() and ins.sum() == 0 and outs.sum() == 0
    assert twr_in_out(values, ins, outs)[2] is None


def test_beta_corr_weekly():
    days = pd.bdate_range("2024-01-01", periods=400)
    rng = np.random.default_rng(1)
    bench = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(days))), index=days)
    asset = bench ** 2 / 100          # randamente ≈ dublul benchmarkului
    res = beta_corr_weekly(asset, bench)
    assert res["beta"] == pytest.approx(2.0, abs=0.1) and res["corr"] > 0.99 and res["n"] >= 52
    assert beta_corr_weekly(asset.iloc[:100], bench.iloc[:100]) is None       # sub 52 de săptămâni
