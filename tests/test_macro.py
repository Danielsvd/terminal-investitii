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


# --- Curba randamentelor pe piața acțiunii -------------------------------------

def test_piata_actiunii():
    from analytics.macro import market_of
    assert market_of("SNP.RO", "RON") == "RO"
    assert market_of("SAP.DE", "EUR") == "EU" and market_of("MC.PA", None) == "EU"
    assert market_of("AAPL", "USD") == "US" and market_of("BRK-B", "USD") == "US" and market_of("TSM", "USD") == "US"
    assert market_of("SHEL.L", "GBP") is None and market_of("NESN.SW", "CHF") is None


def test_panta_lunara_pe_aceeasi_luna():
    from analytics.macro import monthly_spread
    import pandas as pd
    ro10 = pd.Series({"2026-06": 7.30, "2026-07": 7.25, "2026-08": 7.12})
    robor = pd.Series({"2026-06": 6.10, "2026-07": 6.02})          # august nepublicat încă
    res = monthly_spread(ro10, robor)
    assert res["month"] == "2026-07" and res["spread"] == pytest.approx(1.23)
    assert monthly_spread(ro10, pd.Series({"2025-01": 6.0})) is None
    assert monthly_spread(None, robor) is None


def test_panta_zilnica_aceeasi_zi():
    from analytics.macro import same_day_spread
    import pandas as pd
    t10 = pd.Series([4.30, 4.25], index=pd.to_datetime(["2026-10-20", "2026-10-21"]).tz_localize("America/New_York"))
    t3 = pd.Series([3.95], index=pd.to_datetime(["2026-10-20"]).tz_localize("America/New_York"))
    res = same_day_spread(t10, t3)                                  # 21.10 lipsește la 3 luni: se ia 20.10
    assert res["spread"] == pytest.approx(0.35) and res["date"] == pd.Timestamp("2026-10-20")
    assert same_day_spread(t10, pd.Series(dtype=float)) is None
