"""Foaia Portofoliu ca sursă a pozițiilor deschise. Valori sintetice."""
import numpy as np
import pandas as pd
import pytest

from analytics import ledger as L
from data.portfolio_sheet import (cell_number, classes_with_diffs, infer_class, merge_positions,
                                  parse_portfolio_values)

HEADER = ["Symbol", "Date", "Quantity", "AvgPrice", "Currency"]


def test_cell_number_unformatted_and_text():
    assert cell_number(1250) == 1250.0          # celulă numerică citită cu UNFORMATTED_VALUE
    assert cell_number(0.3048) == 0.3048
    assert cell_number("7,0328") == pytest.approx(7.0328)
    assert cell_number("12.345,6") == pytest.approx(12345.6)
    assert cell_number("") is None and cell_number(None) is None
    assert cell_number("#DIV/0!") is None and cell_number("n/a") is None
    assert cell_number(float("nan")) is None


def test_parse_reads_thousands_as_numbers_when_unformatted():
    values = [HEADER,
              ["DN.RO", "2024-02-10", 1250, 1.7702, "RON"],      # fostul „1.250” citit ca 1,25
              ["ABC.DE", "2026-10-22", 6, "7,0328", "eur"],
              ["", "", "", "", ""]]
    df, problems = parse_portfolio_values(values)
    assert problems == []
    assert list(df["Symbol"]) == ["DN.RO", "ABC.DE"]
    assert df.loc[0, "Quantity"] == 1250.0
    assert df.loc[1, "Currency"] == "EUR" and df.loc[1, "AvgPrice"] == pytest.approx(7.0328)


def test_parse_skips_invalid_rows_and_keeps_missing_cost_as_nan():
    values = [HEADER,
              ["X", "", "", 10, "USD"],         # cantitate lipsă → sărit
              ["Y", "", 0, 10, "USD"],          # cantitate 0 → sărit
              ["Z", "", 2, "", "USD"],          # preț lipsă → NaN, nu 0
              ["W", "", 1, 5, ""]]              # monedă lipsă → sărit
    df, problems = parse_portfolio_values(values)
    assert list(df["Symbol"]) == ["Z"]
    assert np.isnan(df.loc[0, "AvgPrice"])
    assert len(problems) == 4


def test_parse_aggregates_duplicates_weighted():
    # 2 @ 10 + 3 @ 20 = 80 / 5 = 16
    df, problems = parse_portfolio_values([HEADER, ["Q", "", 2, 10, "USD"], ["Q", "", 3, 20, "USD"]])
    assert len(df) == 1 and df.loc[0, "Quantity"] == 5 and df.loc[0, "AvgPrice"] == pytest.approx(16.0)
    assert any("mai multe ori" in p for p in problems)


def test_parse_missing_columns():
    df, problems = parse_portfolio_values([["Symbol", "Quantity"], ["A", 1]])
    assert df.empty and "AvgPrice" in problems[0] and "Currency" in problems[0]


def _ledger_positions():
    rows = [
        ["Date", "Symbol", "Class", "Type", "Quantity", "Price", "Currency", "Fees", "Amount", "Broker", "ID", "Notes"],
        ["2025-01-02 10:00:00", "ETFX.DE", "ETF", "BUY", "5", "7", "EUR", "0", "-35", "B", "1", ""],
        ["2025-02-02 10:00:00", "ETFX.DE", "ETF", "DIV", "", "", "EUR", "0", "1,5", "B", "2", ""],
        ["2025-01-02 10:00:00", "OLD", "Actiuni SUA", "BUY", "2", "10", "USD", "0", "-20", "B", "3", ""],
        ["2025-03-02 10:00:00", "OLD", "Actiuni SUA", "SELL", "1", "15", "USD", "0", "15", "B", "4", ""],
        ["2025-01-05 10:00:00", "R1AE", "Obligatiuni", "BUY", "3", "100", "EUR", "0", "-300", "A", "5", ""],
    ]
    df, _ = L.parse_ledger(rows)
    pos, _ = L.build_positions(df)
    return pos


def test_merge_sheet_wins_for_quantity_and_cost_ledger_keeps_history():
    sheet, _ = parse_portfolio_values([HEADER,
                                       ["ETFX.DE", "", 6, 7.25, "EUR"],     # o cumpărare neimportată
                                       ["NEW", "", 4, 5, "USD"]])           # simbol care nu e în registru
    pos, diffs, inferred = merge_positions(sheet, _ledger_positions())
    p = pos.set_index("Symbol")
    # foaia: 6 × 7,25 = 43,5
    assert p.loc["ETFX.DE", "Quantity"] == 6 and p.loc["ETFX.DE", "CostBasis"] == pytest.approx(43.5)
    assert p.loc["ETFX.DE", "AvgCost"] == pytest.approx(7.25)
    assert p.loc["ETFX.DE", "Income"] == pytest.approx(1.5)          # dividendul rămâne din registru
    # OLD: 1 bucată în registru, lipsește din foaie → închis; realizatul 15 − 10 = 5 rămâne
    assert p.loc["OLD", "Quantity"] == 0 and p.loc["OLD", "Realized"] == pytest.approx(5.0)
    # titlul de stat nu se trece în foaie → rămâne din registru
    assert p.loc["R1AE", "Quantity"] == 3 and p.loc["R1AE", "CostBasis"] == pytest.approx(300.0)
    # simbol nou: clasa dedusă din monedă
    assert p.loc["NEW", "Class"] == "Actiuni SUA" and p.loc["NEW", "Quantity"] == 4
    assert inferred == ["NEW"]
    assert diffs == [("ETFX.DE", 6.0, 5.0), ("NEW", 4.0, 0.0), ("OLD", 0.0, 1.0)]
    assert classes_with_diffs(pos, diffs) == {("ETF", "EUR"), ("Actiuni SUA", "USD")}


def test_merge_no_diffs_when_sheet_matches():
    sheet, _ = parse_portfolio_values([HEADER, ["ETFX.DE", "", 5, 7, "EUR"], ["OLD", "", 1, 10, "USD"]])
    pos, diffs, inferred = merge_positions(sheet, _ledger_positions())
    assert diffs == [] and inferred == []


def test_merge_without_ledger():
    sheet, _ = parse_portfolio_values([HEADER, ["AAA.RO", "", 100, 2, "RON"], ["BBB.DE", "", 1, 50, "EUR"]])
    pos, diffs, inferred = merge_positions(sheet, None)
    p = pos.set_index("Symbol")
    assert p.loc["AAA.RO", "Class"] == "Actiuni RO" and p.loc["BBB.DE", "Class"] == "Actiuni UE"
    assert p.loc["AAA.RO", "CostBasis"] == pytest.approx(200.0)
    valued = L.value_open_positions(pos, {"AAA.RO": 2.5, "BBB.DE": 40.0})
    v = valued.set_index("Symbol")
    assert v.loc["AAA.RO", "Unrealized"] == pytest.approx(50.0) and v.loc["BBB.DE", "Unrealized"] == pytest.approx(-10.0)


def test_infer_class():
    assert infer_class("X", "USD", {"X": "ETF"}) == ("ETF", False)
    assert infer_class("Y.RO", "RON") == ("Actiuni RO", True)
