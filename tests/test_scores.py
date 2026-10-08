"""Scorul Master AI și SWOT cu piloni lipsă (ai_engine.py).

ai_engine importă Prophet și transformers; aici sunt înlocuite cu module goale,
pentru că testele verifică doar logica de punctare.
"""
import sys
import types

import numpy as np
import pandas as pd

for _name in ("prophet", "transformers"):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.Prophet = object
        _mod.pipeline = lambda *a, **k: None
        sys.modules[_name] = _mod

from ai_engine import calculate_master_ai_score, generate_ai_swot_analysis  # noqa: E402


def _hist(rsi=50.0, above_sma=True):
    idx = pd.bdate_range("2024-01-01", periods=60)
    close = pd.Series(np.linspace(100, 110, 60), index=idx)
    return pd.DataFrame({"Close": close, "SMA50": close - (1 if above_sma else -1), "RSI": rsi})


def _score(**kw):
    args = dict(info={}, hist=_hist(), h_score=8, mos_val=30.0, inst_pct=70.0, rvol=1.0, s_score=0.0,
                opt_data=None, spread=1.0, z_score=None, q_ratio=1.2, regime_msg="⚖️ REGIM CONSOLIDARE")
    args.update(kw)
    return calculate_master_ai_score(**args)


def test_full_data_gives_a_normal_verdict():
    score, action, _, _, reasons = _score()
    # DCF 20 + bilanț 15 + cash 10 + instituții 10 + sentiment neutru 5 + trend 5 + RSI neutru 5 + macro 5 = 75
    assert score == 75
    assert action == "CUMPĂRĂ (STRONG BUY)"
    assert not any("ALTMAN" in r for r in reasons)


def test_missing_fundamentals_are_not_scored_and_verdict_says_so():
    score, action, color, advice, reasons = _score(h_score=None, mos_val=None, inst_pct=None, q_ratio=None)
    assert action == "DATE INSUFICIENTE"
    assert color == "#8B949E"
    for pillar in ("evaluare DCF", "bilanț", "calitatea profitului", "acționariat"):
        assert pillar in advice
    # rămân doar sentiment 5 + trend 5 + RSI 5 + macro 5
    assert score == 20
    assert sum("ℹ️" in r for r in reasons) == 4


def test_two_missing_pillars_still_give_a_verdict():
    _, action, _, _, _ = _score(mos_val=None, inst_pct=None)
    assert action != "DATE INSUFICIENTE"


def test_altman_none_never_triggers_the_bankruptcy_penalty():
    with_none = _score(z_score=None)[0]
    distress = _score(z_score=1.0)[0]
    assert with_none - distress == 20


def test_swot_handles_missing_pillars():
    swot = generate_ai_swot_analysis({}, None, None, None, None, 0.0, yield_spread=None)
    assert swot["Weaknesses"] == [] and swot["Threats"] == []
    swot = generate_ai_swot_analysis({}, 9, 1.0, 30.0, -0.1, 0.0, yield_spread=-0.2)
    assert any("Z-Score" in w for w in swot["Weaknesses"])
    assert any("Recesiune" in t for t in swot["Threats"])
