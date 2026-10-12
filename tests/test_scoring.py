"""Teste pentru analytics/scoring.py. Valori sintetice, calculate de mână în comentarii."""
import pandas as pd
import pytest

from analytics.scoring import (Pillar, combine, linear_score, momentum_12_1, technical_pillars,
                               technical_subscore, trend_score, verdict)


# --- linear_score ---------------------------------------------------------

def test_linear_score_scara_directa_si_inversa():
    assert linear_score(0.0, -0.30, 0.30) == pytest.approx(0.5)
    assert linear_score(0.15, -0.30, 0.30) == pytest.approx(0.75)    # (0,15 + 0,30) / 0,60
    assert linear_score(-0.50, -0.30, 0.30) == 0.0                   # plafonat jos
    assert linear_score(0.90, -0.30, 0.30) == 1.0                    # plafonat sus
    assert linear_score(40, 70, 30) == pytest.approx(0.75)           # (40 − 70) / (30 − 70)


def test_linear_score_lipsa_ramane_none():
    assert linear_score(None, 0, 1) is None
    assert linear_score(float("nan"), 0, 1) is None
    assert linear_score(5, 3, 3) is None


# --- combine --------------------------------------------------------------

def _pillars():
    return [
        Pillar("a", "A", "fundamental", 0.8),
        Pillar("b", "B", "fundamental", None),
        Pillar("c", "C", "fundamental", 0.4),
        Pillar("t", "T", "tehnic", 0.5),
    ]


def test_combine_renormalizeaza_pe_pilonii_si_grupurile_cu_date():
    r = combine(_pillars())
    # Fundamental: (0,8 + 0,4) / 2 = 0,6 → 60,0 (B lipsește, nu valorează 0).
    assert r["groups"]["fundamental"]["score"] == 60.0
    assert r["groups"]["fundamental"]["missing"] == ["B"]
    assert r["groups"]["tehnic"]["score"] == 50.0
    assert r["groups"]["context"]["score"] is None
    # Context lipsă: ponderi 0,60 și 0,25 → total 0,85.
    # (0,6 × 0,60 + 0,5 × 0,25) / 0,85 = 0,485 / 0,85 = 0,570588 → 57,1
    assert r["score"] == 57.1
    assert r["weights_used"] == {"fundamental": pytest.approx(0.7059, abs=1e-4),
                                 "tehnic": pytest.approx(0.2941, abs=1e-4)}
    assert r["coverage"] == (3, 4)
    assert r["missing"] == ["B"]


def test_combine_ponderi_in_grup():
    # (1,0 × 2 + 0,0 × 1) / 3 = 0,6667 → 66,7
    r = combine([Pillar("a", "A", "fundamental", 1.0, weight=2), Pillar("c", "C", "fundamental", 0.0)])
    assert r["groups"]["fundamental"]["score"] == 66.7
    assert r["score"] == 66.7


def test_combine_fara_date():
    r = combine([Pillar("a", "A", "fundamental", None)])
    assert r["score"] is None and r["weights_used"] == {}


def test_combine_grup_necunoscut():
    with pytest.raises(ValueError):
        combine([Pillar("m", "Macro", "macro", 0.5)])


# --- verdict --------------------------------------------------------------

def test_verdict_fara_fundamental_date_insuficiente():
    r = combine([Pillar("t", "T", "tehnic", 0.9)])
    assert r["score"] == 90.0
    assert verdict(r)[0] == "DATE INSUFICIENTE"


def test_verdict_benzi_si_nota_netestat():
    label, _, note = verdict(combine(_pillars()))            # 57,1
    assert label == "NEUTRU" and "netestate" in note
    r = combine([Pillar("a", "A", "fundamental", 0.70)])      # exact 70,0
    assert verdict(r)[0] == "FAVORABIL"
    assert verdict(r, tested=True)[2] == ""
    assert verdict(combine([Pillar("a", "A", "fundamental", 0.39)]))[0] == "NEFAVORABIL"
    assert verdict(None)[0] == "DATE INSUFICIENTE"


# --- tehnic ---------------------------------------------------------------

def test_trend_crescator_descrescator_si_istoric_scurt():
    assert trend_score(list(range(1, 251))) == 1.0
    assert trend_score(list(range(250, 0, -1))) == 0.0
    assert trend_score(list(range(1, 121))) == 1.0           # sub 200: doar Close > SMA50
    assert trend_score(list(range(1, 41))) is None


def test_trend_doua_din_trei():
    # 1..249 apoi 160. SMA50 = (Σ201..249 + 160) / 50 = (11025 + 160) / 50 = 223,7 → 160 < 223,7.
    # SMA200 = (Σ51..249 + 160) / 200 = (29850 + 160) / 200 = 150,05 → 160 > 150,05; 223,7 > 150,05.
    assert trend_score(list(range(1, 250)) + [160]) == pytest.approx(2 / 3)


def _calendar_series(last_month_value=120.0):
    idx = pd.bdate_range("2024-01-01", "2025-01-31")
    s = pd.Series(100.0, index=idx)
    s[s.index > "2024-01-31"] = 120.0
    s[s.index > "2024-12-31"] = last_month_value
    return s


def test_momentum_12_1_pe_date_calendaristice():
    # Final 2025-01-31: P(≤2024-01-31) = 100, P(≤2024-12-31) = 120 → +20%. Ultima lună e exclusă.
    assert momentum_12_1(_calendar_series()) == pytest.approx(0.20)
    assert momentum_12_1(_calendar_series(last_month_value=50.0)) == pytest.approx(0.20)


def test_momentum_istoric_insuficient():
    s = _calendar_series()
    assert momentum_12_1(s[s.index >= "2024-03-01"]) is None
    assert momentum_12_1(pd.Series([1.0, 2.0])) is None       # fără index de date


def test_subscor_tehnic_si_piloni():
    idx = pd.bdate_range("2024-01-01", periods=250)
    close = pd.Series(range(1, 251), index=idx, dtype="float64")
    # Trend 1,0; momentum N/A (sub 12 luni); RSI 50 → 0,5 → (1,0 + 0,5) / 2 = 0,75 → 75,0
    assert technical_subscore(close, 50) == 75.0
    pillars = technical_pillars(close, 50)
    assert [p.score for p in pillars] == [1.0, None, 0.5]
    assert pillars[1].detail == "N/A" and pillars[2].detail == "50"
    assert technical_subscore(close.iloc[:10], None) is None
