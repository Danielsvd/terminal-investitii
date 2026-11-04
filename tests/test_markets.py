"""Piețele: regimul tehnic pe indici și randamentul în lei (local × curs)."""
import numpy as np
import pandas as pd
import pytest

from analytics.markets import MARKETS, horizon_start, market_regime, ron_return, trend_count


def _bdays(n, end="2026-11-04"):
    return pd.bdate_range(end=end, periods=n)


def test_trend_complet_si_descendent():
    up = pd.Series(np.linspace(100, 200, 300), index=_bdays(300))
    down = pd.Series(np.linspace(200, 100, 300), index=_bdays(300))
    assert trend_count(up) == 3
    assert trend_count(down) == 0
    assert trend_count(up.iloc[-199:]) is None          # sub 200 de ședințe


def test_regim_valori():
    idx = _bdays(300)
    s = pd.Series(np.linspace(100, 200, 300), index=idx)
    s.iloc[-1] = 180.0                                   # maximul pe 52 săpt. e penultima (≈199,67)
    r = market_regime(s)
    assert r["last"] == 180.0 and r["date"] == idx[-1] and r["n"] == 300
    assert r["from_high"] == pytest.approx(180.0 / s.iloc[-2] - 1)
    assert r["momentum"] is not None and r["momentum"] > 0
    assert 0 <= r["score"] <= 100


def test_regim_istoric_scurt():
    s = pd.Series(np.linspace(100, 110, 30), index=_bdays(30))
    r = market_regime(s)
    assert r["trend_n"] is None and r["momentum"] is None and r["score"] is None and r["from_high"] is None
    assert market_regime(None)["last"] is None


def test_start_orizont():
    d = pd.Timestamp("2026-11-04")
    assert horizon_start(d, "1 lună") == pd.Timestamp("2026-10-04")
    assert horizon_start(d, "1 an") == pd.Timestamp("2025-11-04")
    assert horizon_start(d, "De la 1 ian.") == pd.Timestamp("2025-12-31")


def _series(pairs):
    return pd.Series([v for _, v in pairs], index=pd.to_datetime([d for d, _ in pairs]))


def test_randament_in_lei_descompus():
    idx = _series([("2025-10-31", 95.0), ("2025-11-04", 100.0), ("2026-10-30", 108.0), ("2026-11-04", 110.0)])
    fx = _series([("2025-11-03", 4.40), ("2025-11-04", 4.50), ("2026-11-03", 4.30), ("2026-11-05", 4.60)])
    r = ron_return(idx, fx, "1 an")
    assert r["local"] == pytest.approx(0.10)                      # 110 / 100 − 1
    # curs: 04.11.2025 = 4,50 → 04.11.2026: ultimul ≤ 04.11 = 03.11 (4,30); nu cel din 05.11
    assert r["fx"] == pytest.approx(4.30 / 4.50 - 1)
    assert r["ron"] == pytest.approx(1.10 * (4.30 / 4.50) - 1)


def test_randament_piata_in_lei_fara_efect_valutar():
    idx = _series([("2025-12-31", 20.0), ("2026-11-04", 23.0)])
    r = ron_return(idx, None, "De la 1 ian.")
    assert r["local"] == pytest.approx(0.15) and r["fx"] == 0.0 and r["ron"] == pytest.approx(0.15)


def test_orizont_neacoperit_sau_curs_lipsa():
    idx = _series([("2026-06-01", 100.0), ("2026-11-04", 110.0)])
    assert ron_return(idx, None, "1 an")["local"] is None           # indicele nu acoperă un an
    idx2 = _series([("2025-11-01", 100.0), ("2026-11-04", 110.0)])
    fx_short = _series([("2026-06-01", 4.4), ("2026-11-04", 4.5)])
    r = ron_return(idx2, fx_short, "1 an")
    assert r["local"] == pytest.approx(0.10) and r["fx"] is None and r["ron"] is None   # lipsă = N/A, nu local


def test_piete_unice():
    keys = [m[0] for m in MARKETS]
    assert len(keys) == len(set(keys)) and {m[3] for m in MARKETS} <= {"USD", "EUR", "RON"}
