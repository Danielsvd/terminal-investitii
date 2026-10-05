import numpy as np
import pandas as pd
import pytest

from analytics.technical import atr_trailing_stop, atr_wilder, macd, rma, rsi_wilder, true_range


def _ohlc(closes):
    """High = Close + 1, Low = Close - 1: True Range ușor de calculat de mână."""
    c = pd.Series(closes, dtype="float64", index=pd.bdate_range("2024-01-01", periods=len(closes)))
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1000})


def test_rma_seed_is_simple_mean_then_recursive():
    out = rma(pd.Series([2.0, 4.0, 6.0, 8.0]), 2)
    # seed = (2+4)/2 = 3 ; apoi (3*1+6)/2 = 4.5 ; (4.5*1+8)/2 = 6.25
    assert np.isnan(out.iloc[0])
    assert out.iloc[1:].tolist() == [3.0, 4.5, 6.25]


def test_true_range_uses_previous_close_on_gaps():
    df = _ohlc([100, 101, 102, 103, 95, 96])
    # ziua 4: High 96, Low 94, Close precedent 103 -> max(2, 7, 9) = 9
    assert true_range(df).tolist() == [2, 2, 2, 2, 9, 2]


def test_atr_wilder_hand_computed():
    df = _ohlc([100, 101, 102, 103, 95, 96])
    atr = atr_wilder(df, 2)
    assert np.isnan(atr.iloc[0])
    assert atr.iloc[1:].tolist() == [2.0, 2.0, 2.0, 5.5, 3.75]


def test_trailing_stop_ratchets_up_and_resets_when_hit():
    df = _ohlc([100, 101, 102, 103, 95, 96])
    out = atr_trailing_stop(df, window=2, multiplier=1)
    # nivel brut = Close - ATR: [nan, 99, 100, 101, 89.5, 92.25]
    # ziua 4: Close 95 <= stopul precedent 101 -> atins, resetare la 89.5
    assert np.isnan(out["ATR_Stop"].iloc[0])
    assert out["ATR_Stop"].iloc[1:].tolist() == [99.0, 100.0, 101.0, 89.5, 92.25]
    assert out["ATR_Stop_Hit"].tolist() == [False, False, False, False, True, False]


def test_trailing_stop_never_sits_above_the_close():
    rng = np.random.default_rng(7)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 600)))
    out = atr_trailing_stop(_ohlc(closes), window=14, multiplier=2.5)
    valid = out.dropna(subset=["ATR_Stop"])
    assert (valid["ATR_Stop"] < valid["Close"]).all()
    assert out["ATR_Stop_Hit"].sum() > 0  # seria are corecții, deci stopul e atins măcar o dată


def test_trailing_stop_does_not_mutate_input_and_handles_short_history():
    df = _ohlc([100, 101, 102, 103, 95, 96])
    cols = list(df.columns)
    atr_trailing_stop(df, window=2, multiplier=1)
    assert list(df.columns) == cols
    assert atr_trailing_stop(df, window=14) is None
    assert atr_trailing_stop(None) is None


def test_rsi_wilder_reference_series():
    # Seria clasică din documentația StockCharts pentru RSI(14).
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
              45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
    rsi = rsi_wilder(pd.Series(closes), 14)
    assert rsi.iloc[:14].isna().all()
    # primele 14 variații: câștiguri 3.34 -> medie 0.238571; pierderi 1.40 -> medie 0.1
    # RS = 2.385714 -> RSI = 100 - 100/3.385714 = 70.4641
    assert rsi.iloc[14] == pytest.approx(70.4641, abs=1e-3)
    # următoarea zi (46.00, variație -0.28): câștig mediu = 0.238571*13/14 = 0.221531
    # pierdere medie = (0.1*13 + 0.28)/14 = 0.112857 -> RSI = 66.2496
    assert rsi.iloc[15] == pytest.approx(66.2496, abs=1e-3)


def test_rsi_is_100_when_there_are_no_losses():
    rsi = rsi_wilder(pd.Series(range(1, 30), dtype="float64"), 14)
    assert rsi.iloc[-1] == 100.0


def test_macd_is_zero_on_flat_prices_and_positive_on_uptrend():
    flat_line, flat_sig = macd(pd.Series([50.0] * 60))
    assert flat_line.abs().max() == 0 and flat_sig.abs().max() == 0
    up_line, _ = macd(pd.Series(np.arange(1.0, 101.0)))
    assert up_line.iloc[-1] > 0
