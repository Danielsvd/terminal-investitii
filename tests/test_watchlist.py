"""Pagina Watchlist: ultimul preț cu data lui și citirea foii (valori sintetice)."""
import numpy as np
import pandas as pd

from analytics import watchlist as wl


def test_last_quotes_calendare_diferite_si_lipsa():
    idx = pd.to_datetime(["2026-11-04", "2026-11-05", "2026-11-06"])
    close = pd.DataFrame({
        "AAPL": [200.0, 201.5, 203.25],
        "TLV.RO": [30.1, 30.4, np.nan],      # BVB fără bara de vineri: ultimul preț e cel de joi
        "GOL": [np.nan, np.nan, np.nan],     # simbol fără date: lipsește, nu 0
        "ZERO": [10.0, 0.0, np.nan],         # 0 nu e preț: rămâne 10 de miercuri
    }, index=idx)
    q = wl.last_quotes(close)
    assert q["AAPL"] == {"price": 203.25, "date": pd.Timestamp("2026-11-06")}
    assert q["TLV.RO"] == {"price": 30.4, "date": pd.Timestamp("2026-11-05")}
    assert q["ZERO"] == {"price": 10.0, "date": pd.Timestamp("2026-11-04")}
    assert "GOL" not in q


def test_last_quotes_index_cu_fus_orar_si_gol():
    idx = pd.DatetimeIndex(["2026-11-06 00:00"], tz="America/New_York")
    q = wl.last_quotes(pd.DataFrame({"MSFT": [500.0]}, index=idx))
    assert q["MSFT"]["date"] == pd.Timestamp("2026-11-06")
    assert wl.last_quotes(pd.DataFrame()) == {}
    assert wl.last_quotes(None) == {}


def test_parse_watchlist_coloane_dupa_nume_si_valori():
    values = [
        ["Notes", "Symbol", "TargetPrice"],   # ordinea din foaie nu contează
        ["suport", " tsla ", 250],
        ["", "TLV.RO", "25,5"],               # text românesc
        ["fără țintă", "MSFT", ""],           # ținta goală: None, nu 0
        ["", "", 10],                         # rând fără simbol: ignorat
        ["scurt", "SAP.DE"],                  # rând mai scurt decât antetul
    ]
    df, problems = wl.parse_watchlist(values)
    assert problems == []
    assert df["Symbol"].tolist() == ["TSLA", "TLV.RO", "MSFT", "SAP.DE"]
    assert df["TargetPrice"].tolist()[:2] == [250.0, 25.5]
    assert pd.isna(df["TargetPrice"].iloc[2]) and pd.isna(df["TargetPrice"].iloc[3])   # lipsă, nu 0
    assert df["Notes"].tolist() == ["suport", "", "fără țintă", "scurt"]


def test_parse_watchlist_foaie_necitita_antet_gresit_goala():
    df, problems = wl.parse_watchlist(None)
    assert df.empty and problems == ["Foaia watchlist nu a putut fi citită."]
    df, problems = wl.parse_watchlist([["Simbol", "Tinta"], ["AAPL", 100]])
    assert df.empty and "Symbol, TargetPrice" in problems[0]
    df, problems = wl.parse_watchlist([["Symbol", "TargetPrice", "Notes"]])
    assert df.empty and problems == []
    df, problems = wl.parse_watchlist([])
    assert df.empty and problems == []
    # fără Notes: se citește, notele rămân goale
    df, problems = wl.parse_watchlist([["Symbol", "TargetPrice"], ["AAPL", 100]])
    assert problems == [] and df["Notes"].tolist() == [""]


def test_symbol_rows_doar_coloana_symbol_de_jos_in_sus():
    values = [
        ["Symbol", "TargetPrice", "Notes"],
        ["AAPL", 150, "TSLA"],        # nota conține exact „TSLA”: nu se șterge rândul ăsta
        ["TSLA", 250, ""],
        ["MSFT", 300, ""],
        [" tsla ", 240, "duplicat"],  # spații și litere mici: același simbol
    ]
    assert wl.symbol_rows(values, "TSLA") == [5, 3]
    assert wl.symbol_rows(values, "tsla ") == [5, 3]
    assert wl.symbol_rows(values, "MSFT") == [4]
    assert wl.symbol_rows(values, "NVDA") == []
    assert wl.symbol_rows(values, "") == []
    assert wl.symbol_rows([["Notes", "Symbol"], ["x", "AAPL"]], "AAPL") == [2]   # coloana după antet
    assert wl.symbol_rows([["Simbol"], ["AAPL"]], "AAPL") == []                   # fără coloana Symbol
    assert wl.symbol_rows([], "AAPL") == [] and wl.symbol_rows(None, "AAPL") == []


def test_normalize_symbol():
    assert wl.normalize_symbol(" tlv.ro ") == "TLV.RO"
    assert wl.normalize_symbol(None) == ""
