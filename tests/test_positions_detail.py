"""Tabelul „Detaliu Poziții”: ponderi, ordine, coloane în română, N/A la lipsă de preț."""
import numpy as np
import pandas as pd
import pytest

from analytics.portfolio import DETAIL_COLUMNS, positions_detail


def _table():
    # AAA: 10 × 30 = 300; BBB: 5 × 140 = 700; CCC fără preț.
    return pd.DataFrame([
        {"Symbol": "AAA", "Quantity": 10.0, "AvgPrice": 25.0, "CurrentPrice": 30.0,
         "MarketValue": 300.0, "Profit": 50.0, "Profit %": 20.0},
        {"Symbol": "CCC", "Quantity": 2.0, "AvgPrice": 10.0, "CurrentPrice": np.nan,
         "MarketValue": np.nan, "Profit": np.nan, "Profit %": np.nan},
        {"Symbol": "BBB", "Quantity": 5.0, "AvgPrice": 160.0, "CurrentPrice": 140.0,
         "MarketValue": 700.0, "Profit": -100.0, "Profit %": -12.5},
    ])


def test_coloane_in_romana_in_ordine():
    out = positions_detail(_table())
    assert list(out.columns) == list(DETAIL_COLUMNS.values())
    assert list(out.columns) == ["Simbol", "Cantitate", "Cost mediu", "Preț curent",
                                 "Valoare", "Pondere %", "Profit", "Profit %"]


def test_ponderi_pe_pozitiile_cu_pret():
    out = positions_detail(_table()).set_index("Simbol")
    # numitor 300 + 700 = 1000 (CCC fără preț nu intră)
    assert out.loc["BBB", "Pondere %"] == pytest.approx(70.0)
    assert out.loc["AAA", "Pondere %"] == pytest.approx(30.0)
    assert np.isnan(out.loc["CCC", "Pondere %"])
    assert out["Pondere %"].sum() == pytest.approx(100.0)


def test_ordine_dupa_valoare_fara_pret_la_final():
    assert positions_detail(_table())["Simbol"].tolist() == ["BBB", "AAA", "CCC"]


def test_lipsa_ramane_nan_nu_zero():
    row = positions_detail(_table()).set_index("Simbol").loc["CCC"]
    assert np.isnan(row["Valoare"]) and np.isnan(row["Profit"]) and np.isnan(row["Profit %"])
    assert row["Cantitate"] == 2.0 and row["Cost mediu"] == 10.0


def test_niciun_pret_ponderi_nan():
    t = _table()
    t["MarketValue"] = np.nan
    assert positions_detail(t)["Pondere %"].isna().all()


def test_tabel_gol():
    out = positions_detail(pd.DataFrame())
    assert out.empty and list(out.columns) == list(DETAIL_COLUMNS.values())
