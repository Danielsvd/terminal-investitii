"""Taburile pe monedă și jobul IV citesc pozițiile din registrul `Tranzactii`. Date inventate."""
import pandas as pd
import pytest

from analytics import ledger as L

HEADER = L.COLUMNS


def sheet(*rows):
    return [HEADER] + [list(r) for r in rows]


VALUES = sheet(
    ["2024-01-15 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "100", "10", "RON", "5", "-1005", "TV", "t1", ""],
    ["2024-03-01 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "100", "12", "RON", "6", "-1206", "TV", "t2", ""],
    ["2024-06-01 09:00:00", "AAA.RO", "Actiuni RO", "BONUS", "50", "", "RON", "0", "0", "TV", "t3", ""],
    ["2024-09-02 11:00:00", "AAA.RO", "Actiuni RO", "SELL", "125", "11", "RON", "7", "1368", "TV", "t4", ""],
    ["2024-02-01 10:00:00", "BBB", "Actiuni SUA", "BUY", "2", "50,25", "USD", "0", "-100,5", "XTB", "x1", ""],
    ["2024-07-01 10:00:00", "BBB", "Actiuni SUA", "SELL", "2", "60", "USD", "0", "120", "XTB", "x2", ""],
    ["2024-08-01 10:00:00", "CCC", "Actiuni SUA", "BUY", "3", "100", "USD", "1", "-301", "XTB", "x3", ""],
    ["2024-08-02 10:00:00", "DDD.DE", "Actiuni UE", "BUY", "4", "25", "EUR", "0", "-100", "XTB", "x4", ""],
    ["2024-08-03 10:00:00", "R2901A", "Obligatiuni", "BUY", "3", "100", "RON", "0", "-300", "TV", "b1", ""],
)


def positions():
    df, problems = L.parse_ledger(VALUES)
    assert not problems
    pos, _ = L.build_positions(df)
    return pos


def test_tab_frame_open_positions_without_bonds():
    tf = L.tab_frame(positions())
    assert list(tf.columns) == ["Symbol", "Date", "Quantity", "AvgPrice", "Currency"]
    by = tf.set_index("Symbol")
    assert set(by.index) == {"AAA.RO", "CCC", "DDD.DE"}      # BBB închis, R2901A titlu de stat
    # AAA.RO: cost 1005 + 1206 = 2211 pe 250 (cu 50 gratuite); vânzarea a 125 scoate jumătate din cost → 1105,5 / 125
    assert by.loc["AAA.RO", "Quantity"] == pytest.approx(125)
    assert by.loc["AAA.RO", "AvgPrice"] == pytest.approx(1105.5 / 125)
    assert by.loc["CCC", "AvgPrice"] == pytest.approx(301 / 3)   # comisionul intră în cost
    assert by.loc["DDD.DE", "Currency"] == "EUR"
    assert (tf["Date"] == "").all()


def test_tab_frame_empty_or_missing():
    assert L.tab_frame(None).empty
    assert list(L.tab_frame(pd.DataFrame()).columns) == ["Symbol", "Date", "Quantity", "AvgPrice", "Currency"]
