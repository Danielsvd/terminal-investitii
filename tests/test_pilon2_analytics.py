"""analytics/pilon2.py: panoul, fluxurile și tabelele Pilon II. Valori sintetice, calculate de mână."""
import numpy as np
import pandas as pd
import pytest

from analytics import pilon2 as p2
from data.pilon2 import HOLDINGS_COLUMNS

NaN = np.nan


def _df():
    rows = [
        # Date, Fond, ISIN, Symbol, Valoare, Cantitate, Pret
        ("2026-05-31", "NN", "ROTLVAACNOR1", "TLV.RO", 24000, 1000, 24.0),
        ("2026-06-30", "NN", "ROTLVAACNOR1", "TLV.RO", 30000, 1200, 25.0),
        ("2026-07-31", "NN", "ROTLVAACNOR1", "TLV.RO", 31200, 1200, 26.0),
        ("2026-05-31", "NN", "ROSNPPACNOR9", "SNP.RO", 400, 500, 0.8),
        ("2026-05-31", "NN", "DE0007164600", "SAP.DE", 999, NaN, NaN),
        ("2026-06-30", "NN", "DE0007164600", "SAP.DE", 999, NaN, NaN),
        ("2026-07-31", "NN", "DE0007164600", "SAP.DE", 999, NaN, NaN),
        ("2026-06-30", "Vital", "ROTLVAACNOR1", "TLV.RO", 7500, 300, 25.0),
        ("2026-07-31", "Vital", "ROTLVAACNOR1", "TLV.RO", 5200, 200, 26.0),
        ("2026-05-31", "Vital", "ROSNPPACNOR9", "SNP.RO", 80, 100, 0.8),
        ("2026-06-30", "Vital", "ROSNPPACNOR9", "SNP.RO", 82, 100, 0.82),
        ("2026-07-31", "Vital", "ROSNPPACNOR9", "SNP.RO", 85, NaN, NaN),          # fără preț → cantitate necunoscută
    ]
    df = pd.DataFrame(rows, columns=["Date", "Fond", "ISIN", "Symbol", "Valoare_RON", "Cantitate_derivata", "Pret_folosit"])
    df["Date"] = pd.to_datetime(df["Date"])
    for c in HOLDINGS_COLUMNS:
        if c not in df:
            df[c] = NaN
    return df[HOLDINGS_COLUMNS]


def _row(panel, sym, fond, d):
    return panel[(panel.Symbol == sym) & (panel.Fond == fond) & (panel.Date == pd.Timestamp(d))].iloc[0]


def test_panel_zero_when_reported_without_symbol_and_flows():
    panel = p2.build_panel(_df())
    assert set(panel.Symbol) == {"TLV.RO", "SNP.RO"}                 # doar BVB
    v_may = _row(panel, "TLV.RO", "Vital", "2026-05-31")
    assert not v_may.Detinut and v_may.Cantitate == 0
    v_jun = _row(panel, "TLV.RO", "Vital", "2026-06-30")
    assert v_jun.dQ == 300 and v_jun.Flux_RON == 7500
    assert _row(panel, "TLV.RO", "Vital", "2026-07-31").Flux_RON == pytest.approx(-2600)
    assert _row(panel, "TLV.RO", "NN", "2026-07-31").Flux_RON == 0
    nn_snp = _row(panel, "SNP.RO", "NN", "2026-06-30")               # vândut complet: preț din alt fond
    assert nn_snp.Cantitate == 0 and nn_snp.dQ == -500 and nn_snp.Flux_RON == pytest.approx(-410)
    assert _row(panel, "SNP.RO", "NN", "2026-07-31").Flux_RON == 0   # 0 → 0, chiar fără preț
    v_jul = _row(panel, "SNP.RO", "Vital", "2026-07-31")
    assert np.isnan(v_jul.Cantitate) and np.isnan(v_jul.dQ) and np.isnan(v_jul.Flux_RON)   # lipsă ≠ 0


def test_fund_table_and_totals():
    cp = p2.company_panel(p2.build_panel(_df()), "tlv.ro")
    t = p2.fund_table(cp, shares_out=10000)
    assert list(t.Fond) == ["NN", "Vital"]                           # ordinea fixă a fondurilor
    nn, vi = t.iloc[0], t.iloc[1]
    assert (nn["Acțiuni"], nn["% din capital"], nn["Etichetă"]) == (1200, 12.0, "neschimbat")
    assert np.isnan(nn["Δ 3 luni"])                                  # nu există raport din aprilie
    assert (vi["Δ față de raportul anterior"], vi["Etichetă"], vi["% din capital"]) == (-100, "ieșire", 2.0)
    tot = p2.totals(cp).set_index("Date")
    assert tot.loc["2026-05-31", "Acțiuni"] == 1000 and np.isnan(tot.loc["2026-05-31", "dQ"])
    assert tot.loc["2026-06-30", ["Acțiuni", "dQ", "Flux_RON"]].tolist() == [1500, 500, 12500]
    assert tot.loc["2026-07-31", "Fonduri care dețin"] == 2 and tot.loc["2026-07-31", "Fonduri raportate"] == 2
    assert p2.fund_table(cp).iloc[0]["% din capital"] != p2.fund_table(cp).iloc[0]["% din capital"]   # fără nr. acțiuni → NaN


def test_labels():
    assert p2._label(100, 0) == "nou" and p2._label(0, 100) == "ieșit complet"
    assert p2._label(5, 5) == "neschimbat" and p2._label(NaN, 5) == "N/A"


def test_implied_cost_liquidity_and_market():
    panel = p2.build_panel(_df())
    cp = p2.company_panel(panel, "TLV.RO")
    assert p2.implied_cost(cp) == (25.0, 2)
    assert p2.implied_cost(cp, since=pd.Timestamp("2026-07-01")) == (None, 0)
    assert p2.days_to_liquidate(1400, 70) == 20.0
    assert p2.days_to_liquidate(1400, 0) is None and p2.days_to_liquidate(NaN, 70) is None
    assert p2.reference_date(panel) == pd.Timestamp("2026-07-31")
    m = p2.market_flows(panel)
    assert list(m.Symbol) == ["SNP.RO", "TLV.RO"]
    tlv = m.set_index("Symbol").loc["TLV.RO"]
    assert (tlv.Flux_RON, tlv.dQ, tlv["Cumpărători"], tlv["Vânzători"]) == (-2600, -100, 0, 1)
    hm = p2.flow_heatmap(panel, months=2, top=1)
    assert list(hm.index) == ["TLV.RO"] and hm.loc["TLV.RO", pd.Timestamp("2026-06-30")] == 12500


def test_empty_inputs():
    empty = pd.DataFrame(columns=HOLDINGS_COLUMNS)
    assert p2.build_panel(empty).empty and p2.build_panel(None).empty
    assert p2.fund_table(p2.company_panel(p2.build_panel(_df()), "H2O.RO")).empty
    assert p2.market_flows(p2.build_panel(empty)).empty and p2.flow_heatmap(p2.build_panel(empty)).empty
    assert p2.fund_color("NN") == "#3987e5" and p2.fund_color("Altul") == p2.OTHER_COLOR
    tot = p2.fund_equity_totals(_df()).set_index(["Fond", "Date"])
    assert tot.loc[("NN", pd.Timestamp("2026-05-31")), "Valoare"] == 24000 + 400 + 999


def test_reference_date_prefers_month_reported_by_most_funds():
    df = _df()
    extra = {**df.iloc[2].to_dict(), "Date": pd.Timestamp("2026-08-31")}     # doar NN a publicat august
    panel = p2.build_panel(pd.DataFrame(df.to_dict("records") + [extra]))
    assert p2.reference_date(panel) == pd.Timestamp("2026-07-31")
    assert p2.reference_date(p2.company_panel(panel, "TLV.RO")) == pd.Timestamp("2026-07-31")


def test_shares_outstanding_fallbacks():
    """TLV, 16.11.2026: Yahoo a trimis o dată `sharesOutstanding`, altă dată nu → „% din capital” N/A."""
    assert p2.shares_outstanding({"sharesOutstanding": 1_246_600_000}) == (1_246_600_000.0, "sharesOutstanding")
    assert p2.shares_outstanding({"sharesOutstanding": None, "impliedSharesOutstanding": 1.2e9}) == \
        (1.2e9, "impliedSharesOutstanding")
    v, src = p2.shares_outstanding({"marketCap": 46_000_000_000, "currentPrice": 36.9})
    assert round(v) == 1_246_612_466 and "proxy" in src
    v, _ = p2.shares_outstanding({"marketCap": 46e9, "currentPrice": None, "regularMarketPrice": 36.9})
    assert round(v) == 1_246_612_466
    for bad in (None, {}, {"sharesOutstanding": 0}, {"sharesOutstanding": "N/A"}, {"marketCap": 46e9},
                {"sharesOutstanding": float("nan")}):
        assert p2.shares_outstanding(bad) == (None, None)


def _hold(rows):
    return pd.DataFrame(rows, columns=["Date", "Fond", "Symbol", "Valoare_RON", "Cantitate_derivata", "Pret_folosit",
                                       "Abatere_intreg"]).assign(Date=lambda d: pd.to_datetime(d["Date"]), Pondere=1.0)


def test_rebase_quantities_bonus_shares_not_a_buy():
    """Emisiune de acțiuni gratuite 1:1 între iunie și iulie (sintetic). În foaie, iunie a fost scris înainte de
    emisiune (100 acț. la 40 lei), iulie după (200 acț. la 20 lei) → Δ fals +100. Cu seria de azi (iunie ajustat la 20),
    iunie devine 200 → Δ 0. Un simbol fără date Yahoo își păstrează valorile; o dată fără preț în 7 zile → NaN."""
    df = _hold([["2026-06-30", "Vital", "TLV.RO", 4000.0, 100.0, 40.0, 0.0],
                ["2026-07-31", "Vital", "TLV.RO", 4000.0, 200.0, 20.0, 0.0],
                ["2026-07-31", "Vital", "XYZ.RO", 990.0, 99.0, 10.0, 0.0],
                ["2026-07-31", "Vital", "SAP.DE", 500.0, np.nan, np.nan, np.nan],
                ["2026-05-31", "Vital", "TLV.RO", 3000.0, 75.0, 40.0, 0.0]])
    close = pd.DataFrame({"TLV.RO": [20.0000004, 20.0, np.nan]},
                         index=pd.to_datetime(["2026-06-30", "2026-07-31", "2026-05-29"]))
    out, kept = p2.rebase_quantities(df, close)
    tlv = out[out["Symbol"] == "TLV.RO"].set_index("Date")
    assert tlv.loc["2026-06-30", "Cantitate_derivata"] == 200 and tlv.loc["2026-06-30", "Pret_folosit"] == 20.0
    assert tlv.loc["2026-07-31", "Cantitate_derivata"] == 200
    assert np.isnan(tlv.loc["2026-05-31", "Cantitate_derivata"])              # fără preț în 7 zile → NaN, nu 0
    assert kept == ["XYZ.RO"] and out.loc[2, "Cantitate_derivata"] == 99.0
    assert np.isnan(out.loc[3, "Cantitate_derivata"])                          # străinele nu se ating
    panel = p2.build_panel(out[out["Date"] >= "2026-06-01"])
    assert panel[(panel["Symbol"] == "TLV.RO") & (panel["Date"] == "2026-07-31")]["dQ"].iloc[0] == 0
    same, kept_all = p2.rebase_quantities(df, None)
    assert same.equals(df) and kept_all == ["TLV.RO", "XYZ.RO"]
