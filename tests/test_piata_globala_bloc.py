"""„4. Piață Globală” în bloc: indicii, mărfurile și topurile din închiderile zilnice (valori sintetice, calculate de mână)."""
import numpy as np
import pandas as pd
import pytest

from analytics.briefing import quote_table, session_table, top_frames

DAYS = pd.bdate_range("2026-11-02", periods=5)          # 02–06.11.2026 (luni–vineri)


def test_quote_table_variatie_pe_seria_proprie():
    # ^GSPC are toate zilele; TVBETETF.RO lipsește vineri (06.11): variația lui e joi față de miercuri, cu data joi
    closes = pd.DataFrame({"^GSPC": [100.0, 101.0, 102.0, 103.0, 98.88],
                           "TVBETETF.RO": [40.0, 41.0, 42.0, 43.05, np.nan]}, index=DAYS)
    t = quote_table(closes, {"S&P 500": "^GSPC", "BET": "TVBETETF.RO", "Lipsă": "NG=F"}).set_index("Simbol")
    sp = t.loc["^GSPC"]
    assert sp["Preț"] == pytest.approx(98.88)
    assert sp["Variație"] == pytest.approx(98.88 - 103.0)                # −4,12
    assert sp["Variație %"] == pytest.approx(-4.0)                       # −4,12 ÷ 103 × 100
    assert sp["Data"] == pd.Timestamp("2026-11-06")
    bet = t.loc["TVBETETF.RO"]
    assert bet["Variație %"] == pytest.approx(2.5)                       # 43,05 ÷ 42 − 1 = 2,5%
    assert bet["Data"] == pd.Timestamp("2026-11-05")
    # simbolul fără date rămâne în tabel, cu None (N/A), nu 0 și nu ascuns
    ng = t.loc["NG=F"]
    assert ng["Preț"] is None or pd.isna(ng["Preț"])
    assert ng["Variație %"] is None or pd.isna(ng["Variație %"])
    assert list(t["Instrument"]) == ["S&P 500", "BET", "Lipsă"]          # ordinea din dicționar


def test_quote_table_gol():
    t = quote_table(pd.DataFrame(), {"Aur": "GC=F"})
    assert len(t) == 1 and pd.isna(t.loc[0, "Preț"])


def test_top_frames_doar_tranzactionate_si_cu_semn():
    close = pd.DataFrame({"AAA": [10.0, 11.0], "BBB": [20.0, 19.0], "CCC": [5.0, 5.5], "DDD": [8.0, 8.8],
                          "SAP.DE": [100.0, 103.0]}, index=DAYS[-2:])
    volume = pd.DataFrame({"AAA": [1e5, 2e5], "BBB": [1e5, 1e5], "CCC": [1e5, 0.0], "DDD": [1e5, 3e5],
                           "SAP.DE": [1e5, 1e5]}, index=DAYS[-2:])
    table, session = session_table(close, volume)
    assert session == pd.Timestamp("2026-11-06")
    gain, lose = top_frames(table, n=2)
    # CCC (+10%) are volum 0 în ședință: nu intră; dintre AAA +10%, DDD +10%, SAP.DE +3% rămân primele două
    assert set(gain["Instrument"]) == {"AAA", "DDD"}
    assert gain["Variație %"].tolist() == pytest.approx([10.0, 10.0])
    assert list(lose["Instrument"]) == ["BBB"] and lose["Variație %"].iloc[0] == pytest.approx(-5.0)
    assert list(lose.columns) == ["Instrument", "Preț", "Monedă", "Variație %"]
    gain3, _ = top_frames(table, n=3)
    assert gain3.set_index("Instrument").loc["SAP.DE", "Monedă"] == "EUR"
    assert gain3.set_index("Instrument").loc["AAA", "Monedă"] == "USD"


def test_top_frames_tabel_gol():
    gain, lose = top_frames(pd.DataFrame(columns=["date", "close", "prev", "pct", "volume", "value", "traded", "sigma"]))
    assert gain.empty and lose.empty and list(gain.columns) == ["Instrument", "Preț", "Monedă", "Variație %"]
