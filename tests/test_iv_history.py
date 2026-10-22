from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from analytics.options import iv_rank_percentile
from data.iv_history import (history_for, make_row, parse_iv_history, should_save, us_session_date,
                             us_session_open)

UTC = ZoneInfo("UTC")


def test_parse_valid_invalid_and_duplicates():
    values = [["Date", "Symbol", "IV_ATM"],
              ["2026-10-20", "aapl", 0.25],
              ["2026-10-21", "AAPL", "0,26"],
              ["2026-10-21", "AAPL", 0.27],      # duplicat: rămâne ultimul
              ["2026-10-21", "MSFT", 27.5],      # procent trecut greșit → sărit
              ["21.10.2026", "MSFT", 0.3],       # dată în alt format → sărit
              ["", "", ""]]
    df, problems = parse_iv_history(values)
    s = history_for(df, "AAPL")
    assert list(s.round(2)) == [0.25, 0.27]
    assert history_for(df, "MSFT").empty
    assert problems and "2 rânduri" in problems[0]


def test_parse_bad_header_and_empty():
    df, problems = parse_iv_history([["Data", "Simbol", "IV"]])
    assert df.empty and "antetul" in problems[0]
    assert parse_iv_history(None)[0].empty


def test_session_open_and_date_ny():
    # 22.10.2026 (joi), 17:00 UTC = 13:00 New York (EDT, UTC−4) → deschis
    assert us_session_open(datetime(2026, 10, 22, 17, 0, tzinfo=UTC))
    # 13:29 UTC = 9:29 New York → închis
    assert not us_session_open(datetime(2026, 10, 22, 13, 29, tzinfo=UTC))
    # 20:00 UTC = 16:00 → închis
    assert not us_session_open(datetime(2026, 10, 22, 20, 0, tzinfo=UTC))
    # sâmbătă
    assert not us_session_open(datetime(2026, 10, 24, 17, 0, tzinfo=UTC))
    # iarna (EST, UTC−5): 14:45 UTC = 9:45 → deschis
    assert us_session_open(datetime(2026, 12, 2, 14, 45, tzinfo=UTC))
    # 02:00 UTC pe 23.10 = 22:00 pe 22.10 la New York
    assert us_session_date(datetime(2026, 10, 23, 2, 0, tzinfo=UTC)) == date(2026, 10, 22)


def test_should_save_and_row():
    df, _ = parse_iv_history([["Date", "Symbol", "IV_ATM"], ["2026-10-22", "AAPL", 0.27]])
    assert not should_save(df, "aapl", date(2026, 10, 22))
    assert should_save(df, "AAPL", date(2026, 10, 23))
    assert should_save(df, "MSFT", date(2026, 10, 22))
    assert should_save(None, "AAPL", date(2026, 10, 22))
    assert make_row(date(2026, 10, 22), "aapl", 0.27718345) == ["2026-10-22", "AAPL", 0.277183]


def test_history_feeds_iv_rank():
    rows = [["Date", "Symbol", "IV_ATM"]] + [[d.strftime("%Y-%m-%d"), "AAPL", (20 + i) / 100]
                                            for i, d in enumerate(pd.date_range("2026-09-01", periods=25))]
    df, _ = parse_iv_history(rows)
    r = iv_rank_percentile(history_for(df, "AAPL"), 0.30, as_of=pd.Timestamp("2026-09-25"))
    assert r["n"] == 25 and r["rank"] == pytest.approx(100 * 10 / 24) and r["percentile"] == pytest.approx(40.0)
