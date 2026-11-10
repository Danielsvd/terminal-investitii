"""Foaia Portofoliu ca sursă a pozițiilor deschise. Valori sintetice."""
import numpy as np
import pytest

from data.portfolio_sheet import cell_number, infer_class, parse_portfolio_values

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


def test_infer_class():
    assert infer_class("X", "USD", {"X": "ETF"}) == ("ETF", False)
    assert infer_class("Y.RO", "RON") == ("Actiuni RO", True)
