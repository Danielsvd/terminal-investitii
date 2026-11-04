"""Expunerea pe sectoare din taburile Portofoliu: etichete, ponderi, verdict."""
import numpy as np
import pytest

from analytics.portfolio import (SECTOR_FUND, SECTOR_UNKNOWN, sector_label, sector_verdict,
                                 sector_weights)


def test_etichete():
    assert sector_label("Technology", "EQUITY") == "Tehnologie"
    assert sector_label("Financial Services") == "Servicii financiare"
    assert sector_label(None, "ETF") == SECTOR_FUND
    assert sector_label("Technology", "etf") == SECTOR_FUND        # ETF-ul nu e sector chiar dacă Yahoo trimite unul
    assert sector_label(None, "EQUITY") == SECTOR_UNKNOWN
    assert sector_label("", None) == SECTOR_UNKNOWN
    assert sector_label("Sector Nou") == "Sector Nou"              # necunoscut în hartă: rămâne cum vine


def test_ponderi_si_grupare():
    vals = {"AAPL": 400.0, "MSFT": 200.0, "JPM": 300.0, "VUSA": 100.0, "XXX": np.nan}
    labs = {"AAPL": "Tehnologie", "MSFT": "Tehnologie", "JPM": "Servicii financiare", "VUSA": SECTOR_FUND}
    df, unpriced = sector_weights(vals, labs)
    assert unpriced == ["XXX"]
    assert df["Sector"].tolist() == ["Tehnologie", "Servicii financiare", SECTOR_FUND]
    # numitor 400 + 200 + 300 + 100 = 1000
    assert df["Pondere %"].tolist() == pytest.approx([60.0, 30.0, 10.0])
    assert df.loc[0, "Simboluri"] == "AAPL, MSFT"
    assert df["Pondere %"].sum() == pytest.approx(100.0)


def test_fara_eticheta_e_nedefinit():
    df, _ = sector_weights({"A": 50.0, "B": 50.0}, {"A": "Energie"})
    assert set(df["Sector"]) == {"Energie", SECTOR_UNKNOWN}


def test_fara_pret_nu_apare_ca_sector_zero():
    df, unpriced = sector_weights({"A": None, "B": np.nan}, {"A": "Energie", "B": "Energie"})
    assert df.empty and unpriced == ["A", "B"]


def test_verdict_concentrat_pe_sectoarele_reale():
    df, _ = sector_weights({"A": 450.0, "B": 250.0, "E": 300.0},
                           {"A": "Tehnologie", "B": "Energie", "E": SECTOR_FUND})
    v = sector_verdict(df)
    assert v["status"] == "concentrat"          # 45% > 40%
    assert v["top"]["Sector"] == "Tehnologie"
    assert v["fund_pct"] == pytest.approx(30.0)


def test_verdict_etf_mare_nu_e_concentrare():
    # ETF-ul are 70%, dar nu e un sector; cel mai mare sector real are 20%
    df, _ = sector_weights({"E": 700.0, "A": 200.0, "B": 100.0},
                           {"E": SECTOR_FUND, "A": "Tehnologie", "B": "Energie"})
    v = sector_verdict(df)
    assert v["status"] == "ok" and v["top"]["Sector"] == "Tehnologie"


def test_verdict_prag_exact_nu_e_concentrare():
    df, _ = sector_weights({"A": 40.0, "B": 60.0}, {"A": "Tehnologie", "B": SECTOR_UNKNOWN})
    v = sector_verdict(df)
    assert v["status"] == "ok" and v["unknown_pct"] == pytest.approx(60.0)


def test_verdict_fara_sectoare_cunoscute():
    df, _ = sector_weights({"E": 1.0}, {"E": SECTOR_FUND})
    assert sector_verdict(df)["status"] == "na"
    assert sector_verdict(sector_weights({}, {})[0])["status"] == "na"
