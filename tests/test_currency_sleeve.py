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


# --- Impozitul reținut după o vânzare totală (cazul XTB: 3% a doua zi) ---------------------------------

def test_impozit_dupa_vanzare_totala_se_scade_din_vanzare():
    # zi 4: vânzare totală 1210; zi 5: impozit 6,3 reținut când nu mai e nimic deținut
    ledger = pd.concat([LEDGER, _ledger([{"Date": "2026-01-09", "Symbol": "AAA", "Class": "Actiuni SUA",
                                          "Type": "TAX", "Currency": "USD", "Amount": -6.3}])], ignore_index=True)
    values, ins, outs, _, _ = currency_sleeve(ledger, "USD", IDX, CLOSES)
    assert ins.iloc[4] == pytest.approx(6.3)                       # intrarea brută rămâne în date
    daily, cumulative, total = twr_in_out(values, ins, outs)
    # zi 4: (1210 − 6,3)/1100 − 1 = 9,4273%; zi 5: 0 (nu −100%)
    assert daily.iloc[3] == pytest.approx(1203.7 / 1100 - 1)
    assert daily.iloc[4] == 0.0
    assert total == pytest.approx(1.1 * (1120 / 1100) * (1203.7 / 1100) - 1)   # 22,57%
    assert (1.0 + cumulative).min() > 0


def test_lantul_continua_dupa_impozit_si_o_cumparare_noua():
    idx = pd.bdate_range("2026-02-02", periods=6)
    values = pd.Series([100.0, 110.0, 0.0, 0.0, 50.0, 55.0], index=idx)
    ins = pd.Series([100.0, 0.0, 0.0, 2.0, 50.0, 0.0], index=idx)
    outs = pd.Series([0.0, 0.0, 121.0, 0.0, 0.0, 0.0], index=idx)
    daily, cumulative, total = twr_in_out(values, ins, outs)
    # zi 3: 119/110 − 1; zi 4: 0; zi 5: 50/50 − 1 = 0; zi 6: 55/50 − 1 = 10%
    assert daily.round(10).tolist() == [0.0, 0.1, round(119 / 110 - 1, 10), 0.0, 0.0, 0.1]
    assert total == pytest.approx(1.1 * (119 / 110) * 1.1 - 1)
    # seriile primite nu se modifică
    assert ins.iloc[3] == 2.0 and outs.iloc[2] == 121.0


def test_intrare_fara_detineri_si_fara_vanzare_anterioara_se_ignora():
    idx = pd.bdate_range("2026-02-02", periods=4)
    values = pd.Series([0.0, 0.0, 100.0, 110.0], index=idx)
    ins = pd.Series([0.0, 5.0, 100.0, 0.0], index=idx)
    outs = pd.Series(0.0, index=idx)
    daily, _, total = twr_in_out(values, ins, outs)
    assert daily.tolist() == [0.0, 0.0, 0.0, pytest.approx(0.1)]
    assert total == pytest.approx(0.1)


def test_cumparare_si_vanzare_in_aceeasi_zi_neschimbate():
    idx = pd.bdate_range("2026-02-02", periods=2)
    values = pd.Series([0.0, 0.0], index=idx)
    ins = pd.Series([0.0, 96.19], index=idx)
    outs = pd.Series([0.0, 95.91], index=idx)
    daily, _, _ = twr_in_out(values, ins, outs)
    assert daily.iloc[1] == pytest.approx(95.91 / 96.19 - 1)


def test_impozit_in_ziua_vanzarii_cu_detineri_neschimbat():
    idx = pd.bdate_range("2026-02-02", periods=2)
    values = pd.Series([100.0, 0.0], index=idx)
    ins = pd.Series([100.0, 1.0], index=idx)
    outs = pd.Series([0.0, 110.0], index=idx)
    daily, _, _ = twr_in_out(values, ins, outs)
    assert daily.iloc[1] == pytest.approx(110 / 101 - 1)
