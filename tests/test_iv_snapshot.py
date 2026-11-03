from datetime import date

from data.iv_history import parse_iv_history
from tools.iv_snapshot import rows_to_write, us_symbols


def test_us_symbols_filters_markets_currency_and_duplicates():
    portfolio = [["Symbol", "Date", "Quantity", "AvgPrice", "Currency"],
                 ["aapl", "2024-01-02", 10, 150, "USD"],
                 ["SNP.RO", "2024-01-02", 100, 0.6, "RON"],
                 ["SPYL.DE", "2024-01-02", 5, 12, "EUR"],
                 ["VWCE", "2024-01-02", 1, 100, "EUR"],      # fără sufix, dar în EUR → nu e SUA
                 ["BRK-B", "2024-01-02", 1, 400, "usd"],
                 ["", "", "", "", ""]]
    watchlist = [["Symbol", "TargetPrice", "Notes"],
                 ["MSFT", 300, ""], ["AAPL", 140, "duplicat"], ["^GSPC", "", ""], ["TLV.RO", 30, ""]]
    assert us_symbols(portfolio, watchlist) == ["AAPL", "BRK-B", "MSFT"]


def test_us_symbols_missing_sheets_or_columns():
    assert us_symbols([], []) == []
    assert us_symbols([["Simbol"], ["AAPL"]], [["Symbol"], ["NVDA"]]) == ["NVDA"]
    # fără coloana Currency, simbolurile valide trec
    assert us_symbols([["Symbol"], ["AMD"]], None) == ["AMD"]


def test_rows_to_write_skips_existing_invalid_and_missing():
    hist, _ = parse_iv_history([["Date", "Symbol", "IV_ATM"],
                                ["2026-11-02", "AAPL", 0.25],
                                ["2026-10-30", "MSFT", 0.22]])
    ivs = {"AAPL": 0.26,      # există deja pe 02.11 → sărit
           "MSFT": 0.2345678,  # există doar pe altă zi → scris, rotunjit la 6 zecimale
           "NVDA": None,       # IV lipsă → sărit, nu 0
           "AMD": 7.5,         # peste MAX_IV (procent greșit) → sărit
           "TSLA": 0.0}        # zero → sărit
    rows = rows_to_write(hist, ivs, date(2026, 11, 2))
    assert rows == [["2026-11-02", "MSFT", 0.234568]]


def test_rows_to_write_empty_history():
    hist, _ = parse_iv_history([])
    assert rows_to_write(hist, {"AAPL": 0.3}, date(2026, 11, 2)) == [["2026-11-02", "AAPL", 0.3]]
