import pandas as pd
import pytest

from analytics.macro import real_rate, yoy_pct


def _monthly(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype="float64")


def test_yoy_pct_from_index_level():
    # 13 luni: ianuarie 2024 = 300, ianuarie 2025 = 309 -> +3.0%
    s = _monthly([300, 301, 302, 303, 304, 305, 306, 307, 308, 308.5, 308.7, 308.9, 309])
    assert yoy_pct(s) == pytest.approx(3.0)


def test_yoy_pct_none_when_history_is_too_short():
    assert yoy_pct(_monthly([300 + i for i in range(12)])) is None  # doar 11 luni înapoi
    assert yoy_pct(_monthly([300])) is None
    assert yoy_pct(None) is None


def test_yoy_pct_none_when_reference_point_is_missing():
    s = _monthly([300 + i for i in range(20)])
    s = s.drop(s.index[2:9])  # gol exact în zona de acum 12 luni
    assert yoy_pct(s) is None


def test_real_rate():
    assert real_rate(4.5, 3.0) == pytest.approx(1.5)
    assert real_rate(4.5, None) is None
    assert real_rate(None, 3.0) is None


def test_real_rate_is_nowhere_near_the_cpi_level_bug():
    # Eroarea veche scădea nivelul indicelui (~320) din dobândă: 4.5 - 320 = -315.5
    s = _monthly([310.0] * 12 + [320.0])
    assert real_rate(4.5, yoy_pct(s)) == pytest.approx(4.5 - (320 / 310 - 1) * 100)
