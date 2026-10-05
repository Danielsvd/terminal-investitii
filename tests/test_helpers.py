import time

import numpy as np
import pandas as pd

from data.helpers import close_frame, field_frame, now_ro, num, slice_window, struct_time_utc_to_ro


# --- num --------------------------------------------------------------------

def test_num_returns_none_for_missing_none_nan_and_text():
    info = {"a": None, "b": float("nan"), "c": "N/A", "d": True, "e": float("inf")}
    for key in ("a", "b", "c", "d", "e", "missing"):
        assert num(info, key) is None
    assert num(None, "a") is None
    assert num({}, "a") is None


def test_num_keeps_real_values_including_zero():
    info = {"zero": 0, "neg": -1.5, "txt": "12.5", "np": np.float64(3.0)}
    assert num(info, "zero") == 0.0
    assert num(info, "neg") == -1.5
    assert num(info, "txt") == 12.5
    assert num(info, "np") == 3.0


# --- field_frame / close_frame ----------------------------------------------

IDX = pd.bdate_range("2024-01-01", periods=3)


def _multi(order):
    """Construiește rezultatul yf.download pentru AAA și BBB în ambele forme de MultiIndex."""
    data = {}
    for t, base in (("AAA", 10.0), ("BBB", 20.0)):
        for f, add in (("Close", 0.0), ("Volume", 1000.0)):
            key = (f, t) if order == "field_first" else (t, f)
            data[key] = [base + add, base + add + 1, base + add + 2]
    df = pd.DataFrame(data, index=IDX)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_close_frame_field_first_multiindex():
    out = close_frame(_multi("field_first"), ["AAA", "BBB"])
    assert list(out.columns) == ["AAA", "BBB"]
    assert out["BBB"].tolist() == [20.0, 21.0, 22.0]


def test_close_frame_ticker_first_multiindex_group_by_ticker():
    out = close_frame(_multi("ticker_first"), ["AAA", "BBB"])
    assert sorted(out.columns) == ["AAA", "BBB"]
    assert out["AAA"].tolist() == [10.0, 11.0, 12.0]


def test_close_frame_single_ticker_all_shapes():
    flat = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Volume": [5, 5, 5]}, index=IDX)
    assert close_frame(flat, ["AAA"])["AAA"].tolist() == [1.0, 2.0, 3.0]
    assert close_frame(flat, "AAA")["AAA"].tolist() == [1.0, 2.0, 3.0]

    one_field_first = flat.copy()
    one_field_first.columns = pd.MultiIndex.from_tuples([("Close", "AAA"), ("Volume", "AAA")])
    assert close_frame(one_field_first, ["AAA"])["AAA"].tolist() == [1.0, 2.0, 3.0]

    one_ticker_first = flat.copy()
    one_ticker_first.columns = pd.MultiIndex.from_tuples([("AAA", "Close"), ("AAA", "Volume")])
    assert close_frame(one_ticker_first, ["AAA"])["AAA"].tolist() == [1.0, 2.0, 3.0]


def test_close_frame_drops_symbols_without_data_and_handles_empty():
    df = _multi("ticker_first")
    df[("BBB", "Close")] = np.nan
    assert list(close_frame(df, ["AAA", "BBB"]).columns) == ["AAA"]
    assert close_frame(None, ["AAA"]).empty
    assert close_frame(pd.DataFrame(), ["AAA"]).empty
    assert field_frame(_multi("field_first"), "Open", ["AAA"]).empty


def test_field_frame_volume():
    out = field_frame(_multi("ticker_first"), "Volume", ["AAA", "BBB"])
    assert out["AAA"].tolist() == [1010.0, 1011.0, 1012.0]


# --- slice_window -----------------------------------------------------------

def test_slice_window_uses_calendar_dates_not_row_counts():
    s = pd.Series(1.0, index=pd.bdate_range("2021-01-01", "2024-12-31"))
    one_year = slice_window(s, "1A")
    assert one_year.index[0] >= pd.Timestamp("2023-12-31")
    assert 255 <= len(one_year) <= 265          # ~1 an de ședințe, nu 365 de rânduri
    assert len(slice_window(s, "1 An")) == len(one_year)
    assert 20 <= len(slice_window(s, "1L")) <= 24
    assert len(slice_window(s, "1S")) in (5, 6)


def test_slice_window_one_day_default_and_empty():
    s = pd.Series(range(10), index=pd.bdate_range("2024-01-01", periods=10), dtype="float64")
    assert slice_window(s, "1Z").tolist() == [8.0, 9.0]
    assert len(slice_window(s, "necunoscut")) == 10   # implicit 1 an, seria e mai scurtă
    empty = pd.Series(dtype="float64")
    assert slice_window(empty, "1A").empty
    assert slice_window(None, "1A") is None


def test_slice_window_works_with_timezone_aware_index_and_dataframes():
    idx = pd.bdate_range("2023-01-01", "2024-06-28", tz="America/New_York")
    df = pd.DataFrame({"Close": 1.0}, index=idx)
    out = slice_window(df, "6 Luni")
    assert out.index[0] >= idx[-1] - pd.DateOffset(months=6)
    assert 120 <= len(out) <= 132


# --- timp -------------------------------------------------------------------

def test_now_ro_is_timezone_aware():
    assert str(now_ro().tzinfo) == "Europe/Bucharest"


def test_struct_time_utc_to_ro_summer_and_winter():
    summer = struct_time_utc_to_ro(time.strptime("2026-07-15 10:00", "%Y-%m-%d %H:%M"))
    assert (summer.hour, summer.utcoffset().total_seconds()) == (13, 3 * 3600)
    winter = struct_time_utc_to_ro(time.strptime("2026-01-15 10:00", "%Y-%m-%d %H:%M"))
    assert (winter.hour, winter.utcoffset().total_seconds()) == (12, 2 * 3600)
