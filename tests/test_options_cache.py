"""get_options_analysis_ai: un răspuns gol de la Yahoo nu rămâne în cache."""
import types
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

ai_engine = pytest.importorskip("ai_engine")


class _Fast:
    last_price = 100.0


def _fake_ticker_factory(state):
    class FakeTicker:
        def __init__(self, sym, *a, **k):
            self.sym = sym

        @property
        def options(self):
            state["calls"] += 1
            if self.sym != "AAPLX" or state["empty"]:
                return ()
            today = ai_engine.now_ro().date()
            return tuple((today + timedelta(days=d)).isoformat() for d in (3, 30, 60))

        def option_chain(self, exp):
            strikes = [90.0, 100.0, 110.0]
            side = lambda: pd.DataFrame({"strike": strikes, "openInterest": [10.0, 20.0, 10.0],
                                         "volume": [5.0, 5.0, 5.0], "impliedVolatility": [0.3, 0.25, 0.3],
                                         "bid": [1.0, 1.0, 1.0]})
            return types.SimpleNamespace(calls=side(), puts=side())

        @property
        def fast_info(self):
            return _Fast()

        def history(self, period="10d", **k):
            return pd.DataFrame({"Close": [3.9]}, index=pd.to_datetime(["2026-10-21"]))
    return FakeTicker


def test_empty_yahoo_response_is_retried_not_cached(monkeypatch):
    state = {"calls": 0, "empty": True}
    monkeypatch.setattr(ai_engine.yf, "Ticker", _fake_ticker_factory(state))
    ai_engine._options_analysis_cached.clear()

    data, msg = ai_engine.get_options_analysis_ai("AAPLX")
    assert data is None and "Yahoo nu a trimis lista de expirări" in msg

    state["empty"] = False   # Yahoo își revine
    data, msg = ai_engine.get_options_analysis_ai("AAPLX")
    assert msg == "Succes" and data["days"] == 30
    assert data["oi_pc_ratio"] == pytest.approx(1.0)
    assert data["iv"] == pytest.approx(25.0)

    calls = state["calls"]
    ai_engine.get_options_analysis_ai("AAPLX")   # succesul vine din cache: fără apel nou
    assert state["calls"] == calls
    ai_engine._options_analysis_cached.clear()


def test_message_for_suffixed_symbol_without_options(monkeypatch):
    state = {"calls": 0, "empty": True}
    monkeypatch.setattr(ai_engine.yf, "Ticker", _fake_ticker_factory(state))
    ai_engine._options_analysis_cached.clear()
    data, msg = ai_engine.get_options_analysis_ai("SAP.DE")
    assert data is None and msg.startswith("Fără opțiuni listate în Yahoo")
