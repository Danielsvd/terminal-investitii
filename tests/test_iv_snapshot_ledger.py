"""Jobul IV ia simbolurile SUA și din pozițiile deschise ale registrului `Tranzactii`. Date inventate."""
from analytics import ledger as L
from tools.iv_snapshot import ledger_us_symbols

VALUES = [L.COLUMNS,
          ["2024-02-01 10:00:00", "BBB", "Actiuni SUA", "BUY", "2", "50,25", "USD", "0", "-100,5", "XTB", "x1", ""],
          ["2024-07-01 10:00:00", "BBB", "Actiuni SUA", "SELL", "2", "60", "USD", "0", "120", "XTB", "x2", ""],
          ["2024-08-01 10:00:00", "CCC", "Actiuni SUA", "BUY", "3", "100", "USD", "1", "-301", "XTB", "x3", ""],
          ["2024-08-02 10:00:00", "DDD.DE", "Actiuni UE", "BUY", "4", "25", "EUR", "0", "-100", "XTB", "x4", ""],
          ["2024-08-03 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "10", "10", "RON", "0", "-100", "TV", "t1", ""]]


def test_ledger_us_symbols_only_open_usd():
    assert ledger_us_symbols(VALUES) == ["CCC"]     # BBB închis; .DE și .RO nu sunt SUA


def test_ledger_us_symbols_empty():
    assert ledger_us_symbols([]) == []
    assert ledger_us_symbols([L.COLUMNS]) == []
