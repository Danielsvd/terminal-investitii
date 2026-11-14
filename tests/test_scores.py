"""SWOT și sentimentul cu piloni lipsă (ai_engine.py). Scorul Master AI vechi a fost șters (13.11).

ai_engine importă Prophet și transformers; aici sunt înlocuite cu module goale,
pentru că testele verifică doar logica de punctare.
"""
import sys
import types




for _name in ("prophet", "transformers"):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.Prophet = object
        _mod.pipeline = lambda *a, **k: None
        sys.modules[_name] = _mod

from ai_engine import generate_ai_swot_analysis  # noqa: E402


def test_swot_handles_missing_pillars():
    swot = generate_ai_swot_analysis({}, None, None, None, None, 0.0, yield_spread=None)
    assert swot["Weaknesses"] == [] and swot["Threats"] == []
    swot = generate_ai_swot_analysis({}, 9, 1.0, 30.0, -0.1, 0.0, yield_spread=-0.2)
    assert any("Z-Score" in w for w in swot["Weaknesses"])
    assert any("Recesiune" in t for t in swot["Threats"])


# --- analyze_sentiment_ai: lipsa nu devine 0 (sentiment neutru inventat) ---------------------------

def test_sentiment_media_pe_titluri(monkeypatch):
    import ai_engine
    labels = iter([{"label": "positive", "score": 0.8}, {"label": "negative", "score": 0.4},
                   {"label": "neutral", "score": 0.9}])
    monkeypatch.setattr(ai_engine, "get_finbert_pipeline", lambda: (lambda text: [next(labels)]))
    s = ai_engine.analyze_sentiment_ai([{"title": "a"}, {"title": "b"}, {"title": "c"}])
    assert abs(s - (0.8 - 0.4 + 0.0) / 3) < 1e-12          # 0,1333…


def test_sentiment_eroare_finbert_e_none(monkeypatch):
    import ai_engine

    def _boom(text):
        raise RuntimeError("model indisponibil")
    monkeypatch.setattr(ai_engine, "get_finbert_pipeline", lambda: _boom)
    assert ai_engine.analyze_sentiment_ai([{"title": "a"}]) is None


def test_sentiment_fara_stiri_e_none():
    import ai_engine
    assert ai_engine.analyze_sentiment_ai([]) is None
    assert ai_engine.analyze_sentiment_ai(None) is None


def test_swot_accepta_sentiment_lipsa():
    swot = generate_ai_swot_analysis({}, None, None, None, None, None)
    assert not any("Narațiune" in s for s in swot["Strengths"])
