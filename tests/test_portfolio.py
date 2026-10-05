import numpy as np
import pandas as pd
import pytest

from analytics.portfolio import portfolio_curve, value_positions

IDX = pd.bdate_range("2024-01-01", periods=4)
CLOSES = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0], "B": [np.nan, np.nan, 20.0, 22.0]}, index=IDX)
POSITIONS = pd.DataFrame({
    "Symbol": ["A", "B", "C"],
    "Quantity": [2.0, 1.0, 5.0],
    "AvgPrice": [10.0, 25.0, 1.0],
})


def test_value_positions_live_price_fallback_and_missing():
    table, daily_abs, daily_pct, notes = value_positions(POSITIONS, {"A": 14.0, "B": None, "C": None}, CLOSES)
    t = table.set_index("Symbol")

    # A: preț live 14 -> valoare 28, investit 20, profit 8 (40%)
    assert t.loc["A", "MarketValue"] == 28.0
    assert t.loc["A", "Profit"] == 8.0
    assert t.loc["A", "Profit %"] == pytest.approx(40.0)

    # B: fără preț live -> ultima închidere 22; investit 25, profit -3 (-12%)
    assert t.loc["B", "CurrentPrice"] == 22.0
    assert t.loc["B", "Profit %"] == pytest.approx(-12.0)

    # C: fără preț și fără istoric -> NaN, NU zero (zero ar însemna -100%)
    assert np.isnan(t.loc["C", "CurrentPrice"])
    assert np.isnan(t.loc["C", "MarketValue"])
    assert np.isnan(t.loc["C", "Profit %"])

    assert notes == {"missing_price": ["C"], "price_from_close": ["B"]}

    # variație zilnică: A (14-12)*2 = 4 ; B (22-20)*1 = 2 ; total 6
    # valoare curentă cunoscută 28+22 = 50 -> 6 / (50-6) = 13.636%
    assert daily_abs == pytest.approx(6.0)
    assert daily_pct == pytest.approx(600 / 44)


def test_value_positions_rejects_zero_or_invalid_live_price():
    table, _, _, notes = value_positions(POSITIONS.iloc[:1], {"A": 0.0}, CLOSES)
    assert table.loc[0, "CurrentPrice"] == 13.0
    assert notes["price_from_close"] == ["A"]
    table, _, _, _ = value_positions(POSITIONS.iloc[:1], {"A": float("nan")}, CLOSES)
    assert table.loc[0, "CurrentPrice"] == 13.0


def test_curve_starts_where_all_held_symbols_have_prices():
    curve, notes = portfolio_curve(POSITIONS, CLOSES)
    # primele două zile lipsesc: B nu are preț și nu se inventează unul
    assert list(curve.index) == list(IDX[2:])
    assert curve.tolist() == [2 * 12 + 20, 2 * 13 + 22]
    assert notes["no_history"] == ["C"]
    assert notes["limiting_symbol"] == "B"
    assert notes["curve_start"] == IDX[2]


def test_curve_fills_inner_gaps_forward_only():
    closes = pd.DataFrame({"A": [10.0, np.nan, 12.0, 13.0], "B": [5.0, 6.0, 7.0, 8.0]}, index=IDX)
    curve, _ = portfolio_curve(POSITIONS.iloc[:2], closes)
    # ziua 2: A folosește ultimul preț cunoscut (10) -> 2*10 + 6 = 26
    assert curve.tolist() == [25.0, 26.0, 31.0, 34.0]


def test_curve_sums_duplicate_rows_for_the_same_symbol():
    positions = pd.DataFrame({"Symbol": ["A", "A"], "Quantity": [1.0, 2.0], "AvgPrice": [9.0, 11.0]})
    curve, _ = portfolio_curve(positions, CLOSES)
    assert curve.tolist() == [30.0, 33.0, 36.0, 39.0]


def test_curve_empty_when_nothing_has_history():
    curve, notes = portfolio_curve(POSITIONS.iloc[2:], CLOSES)
    assert curve.empty
    assert notes["no_history"] == ["C"] and notes["curve_start"] is None
