"""Explicația scorului macro din Piață Globală: abaterile față de neutru însumate dau scorul."""
import pytest

from analytics import macro as m


def _p(score):
    return {"score": score, "value": None, "detail": ""}


def test_abateri_insumate_dau_scorul():
    # credit 0 (30), vix 50 (20), dolar 100 (15), sectoare 50 (15), curba 0 (10), sentiment lipsă
    pillars = {"credit": _p(0), "vix": _p(50), "dolar": _p(100), "sectoare": _p(50),
               "curba": _p(0), "sentiment": _p(None)}
    mv = m.macro_verdict(pillars)
    # w = 90; scor = (0·30 + 50·20 + 100·15 + 50·15 + 0·10) / 90 = 3250 / 90 = 36,11
    assert mv["weight_used"] == 90
    assert mv["score"] == pytest.approx(3250 / 90)
    ex = m.macro_explanation(pillars, mv)
    d = {k: dev for k, _, dev in ex["pulls"]}
    assert d["credit"] == pytest.approx(-50 * 30 / 90)   # −16,67
    assert d["dolar"] == pytest.approx(50 * 15 / 90)     # +8,33
    assert d["curba"] == pytest.approx(-50 * 10 / 90)    # −5,56
    assert d["vix"] == 0 and d["sectoare"] == 0
    assert 50 + sum(d.values()) == pytest.approx(mv["score"])
    assert ex["pulls"][0][0] == "credit"
    assert ex["missing"] == ["sentiment"]
    txt = " ".join(ex["lines"])
    assert "Scorul 36 din 100" in txt
    assert "Credit 0 → -16,7" in txt
    assert "Dolar 100 → +8,3" in txt
    assert "Total -13,9 → 36" in txt
    assert "Piața de credit (HYG/IEF) (-16,7)" in txt
    assert "Dolar vs acțiuni (corelație SPY/UUP) (+8,3)" in txt
    assert "Fără date acum: Sentiment" in txt


def test_fara_scor():
    pillars = {k: _p(None) for k in m.MACRO_WEIGHTS}
    pillars["credit"] = _p(100)
    mv = m.macro_verdict(pillars)
    ex = m.macro_explanation(pillars, mv)
    assert mv["score"] is None
    assert ex["pulls"] == []
    assert ex["lines"][0] == "Fără scor: doar 30% din ponderi au date (minimum 50%)."


def test_toti_neutri_fara_extreme():
    pillars = {k: _p(50) for k in m.MACRO_WEIGHTS}
    ex = m.macro_explanation(pillars, m.macro_verdict(pillars))
    assert not any(l.startswith("Azi ") for l in ex["lines"])
    assert ex["missing"] == []
