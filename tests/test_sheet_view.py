"""Tabul „Import Date”: foile BVB și GLOBAL citite ca atare (valori sintetice)."""
import pandas as pd

from data.sheet_view import (AVERAGE_ROW, MEDIAN_ROW, bvb_view, cell_currency, cell_number, column_kind,
                             currency_groups, global_view, mixed_currency_rows)


def test_celule_lipsa_sunt_none_nu_zero():
    for cell in ("", "-", "-%", "#DIV/0!", "#N/A", None, "  "):
        assert cell_number(cell) is None, cell


def test_celule_numerice_format_foaie():
    assert cell_number("1.003,40€") == 1003.4
    assert cell_number("-,34 $") == -0.34
    assert cell_number(",97 $") == 0.97
    assert cell_number("3,26%%") == 3.26
    assert cell_number("5.256.630.400.000 $") == 5256630400000.0
    assert cell_number("0,8699 lei") == 0.8699
    assert cell_number("15.040.000.000") == 15040000000.0
    assert cell_number(12.5) == 12.5


def test_text_cu_cifre_nu_e_numar():
    assert cell_number("Q2 26") is None and cell_number("T2 2025") is None and cell_number("Cumpără") is None


def test_moneda_din_celula():
    assert [cell_currency(c) for c in ("349,51 $", "51,12€", "2007,31 £", "0,05 lei", "24,49", "")] == \
        ["USD", "EUR", "GBP", "RON", None, None]


def test_tipul_coloanei():
    assert column_kind(["3,17%", "#DIV/0!", "-1,2%"]) == (True, "%")
    assert column_kind(["0,05 lei", "1,8 lei", ""]) == (True, " lei")
    assert column_kind(["Q2 26", "Q2 26"]) == (False, "")
    assert column_kind(["", "-"]) == (True, "")


BVB = [
    ["Multipli", "Multipli de preț", "TLV", "SNP", "H2O"],
    ["", "Multipli de preț"],                                   # rând-titlu, fără valori
    ["8,00", "P/E 2025", "6,00", "10,00", "200,00"],            # coloana nouă (lista fixă veche căuta P/E 2024)
    ["20,00%", "Rentabilitate capital (ROE)", "22,10%", "#DIV/0!", "15,00%"],
    ["", "EPS TTM", "1,8417 lei", "0,05 lei", ""],
    ["", "raportare", "Q2 26", "Q2 26", "Q1 26"],
    ["9,99", "P/E 2025", "1", "1", "1"],                        # duplicat
    ["", "", "x", "y", "z"],                                    # fără nume de indicator
]


def test_bvb_transpus_cu_media_si_mediana():
    df, units, dropped = bvb_view(BVB)
    assert list(df.index) == ["TLV", "SNP", "H2O", AVERAGE_ROW, MEDIAN_ROW]
    assert list(df.columns) == ["P/E 2025", "Rentabilitate capital (ROE)", "EPS TTM", "raportare"]
    assert df.loc["SNP", "P/E 2025"] == 10.0 and df.loc[AVERAGE_ROW, "P/E 2025"] == 8.0
    assert df.loc[MEDIAN_ROW, "P/E 2025"] == 10.0                       # mediana(6, 10, 200)
    assert df.loc["SNP", "Rentabilitate capital (ROE)"] is None or pd.isna(df.loc["SNP", "Rentabilitate capital (ROE)"])
    assert df.loc[MEDIAN_ROW, "Rentabilitate capital (ROE)"] == (22.10 + 15.0) / 2
    assert pd.isna(df.loc["H2O", "EPS TTM"]) and pd.isna(df.loc[AVERAGE_ROW, "EPS TTM"])
    assert units == {"P/E 2025": "", "Rentabilitate capital (ROE)": "%", "EPS TTM": " lei"}
    assert df.loc["H2O", "raportare"] == "Q1 26" and pd.isna(df.loc[MEDIAN_ROW, "raportare"])
    assert dropped == ["P/E 2025"]


def test_bvb_gol():
    assert bvb_view(None)[0].empty and bvb_view([["a", "b"]])[0].empty


GLOBAL = [
    ["Industrie", "Companii", "Capitalizare", "Preț actiune", "Abatere", "Preț țintă", "Dividend", "Nr Actiuni", "Recomandare"],
    ["Tehnologie", "Alfa ", "2.500.000.000.000 $", "180,50 $", "#DIV/0!", "200,00 $", "-", "15.040.000.000", "Cumpără"],
    ["", "Beta", "29.398.089.600€", "51,12€", "27,80%", "179,00 $", "0,28€", "575.080.000", "Ține"],
    ["Auto", "Gama", "59.818.861.728 £", "2007,31 £", "-%", "1.476,00 £", "0,33 £", "29.800.510", "Cumpără"],
    ["", "", "", "", "", "", "", "", ""],                                   # rând gol
]


def test_global_numere_monede_industrie():
    df, units, cur = global_view(GLOBAL)
    assert list(df["Companii"]) == ["Alfa", "Beta", "Gama"]
    assert list(df["Industrie"]) == ["Tehnologie", "Tehnologie", "Auto"]
    assert "Capitalizare (mld.)" in df.columns and df.at[0, "Capitalizare (mld.)"] == 2500.0
    assert df.at[1, "Preț actiune"] == 51.12 and pd.isna(df.at[0, "Abatere"]) and pd.isna(df.at[2, "Abatere"])
    assert pd.isna(df.at[0, "Dividend"]) and df.at[2, "Nr Actiuni"] == 29800510.0
    assert units["Abatere"] == "%" and units["Preț actiune"] == "" and "Recomandare" not in units
    assert list(cur["Preț actiune"]) == ["USD", "EUR", "GBP"]
    assert currency_groups(cur, "Preț țintă") == {"USD": [0, 1], "GBP": [2]}
    assert currency_groups(cur, "nu există") == {}


def test_global_monede_amestecate_pe_rand():
    df, _, cur = global_view(GLOBAL)
    assert mixed_currency_rows(df, cur) == [("Beta", "EUR", {"Preț țintă": "USD"})]


def test_global_gol():
    df, units, cur = global_view([])
    assert df.empty and units == {} and cur.empty
