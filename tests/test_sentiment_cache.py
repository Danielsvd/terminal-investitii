"""Sentimentul FinBERT și fluxul RSS nu se mai recalculează la fiecare rerulare (cache), iar eșecurile nu intră în cache."""
import sys
import types

import pytest


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture(scope="module")
def ai_engine():
    # prophet / transformers sunt grele și lipsesc în mediul de test: substitute doar dacă nu sunt instalate
    added = []
    for name, attrs in (("prophet", {"Prophet": object}), ("transformers", {"pipeline": lambda *a, **k: None})):
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = _stub(name, **attrs)
            added.append(name)
    import ai_engine as mod
    yield mod
    for name in added:
        sys.modules.pop(name, None)


def _analyzer(state, fail=False):
    def run(text):
        state["calls"] += 1
        if fail:
            raise RuntimeError("model indisponibil")
        return [{"label": "positive", "score": 0.8}] if "up" in text else [{"label": "negative", "score": 0.6}]
    return run


def test_finbert_scor_si_cache(ai_engine, monkeypatch):
    ai_engine._finbert_mean.clear()
    state = {"calls": 0}
    monkeypatch.setattr(ai_engine, "get_finbert_pipeline", lambda: _analyzer(state))
    news = [{"title": "stocks up"}, {"title": "bonds down"}]
    # (+0,8 − 0,6) ÷ 2 = 0,1
    assert ai_engine.analyze_sentiment_ai(news) == pytest.approx(0.1)
    assert state["calls"] == 2
    assert ai_engine.analyze_sentiment_ai(news) == pytest.approx(0.1)
    assert state["calls"] == 2                       # aceleași titluri: din cache, modelul nu mai rulează
    ai_engine.analyze_sentiment_ai(news + [{"title": "gold up"}])
    assert state["calls"] == 5                       # titluri noi: cheie nouă, se recalculează


def test_finbert_doar_primele_10(ai_engine, monkeypatch):
    ai_engine._finbert_mean.clear()
    state = {"calls": 0}
    monkeypatch.setattr(ai_engine, "get_finbert_pipeline", lambda: _analyzer(state))
    news = [{"title": f"up {i}"} for i in range(10)] + [{"title": "down 11"}]
    assert ai_engine.analyze_sentiment_ai(news) == pytest.approx(0.8)
    assert state["calls"] == 10


def test_finbert_eroare_none_si_fara_cache(ai_engine, monkeypatch):
    ai_engine._finbert_mean.clear()
    state = {"calls": 0}
    monkeypatch.setattr(ai_engine, "get_finbert_pipeline", lambda: _analyzer(state, fail=True))
    news = [{"title": "stocks up"}]
    assert ai_engine.analyze_sentiment_ai(news) is None
    monkeypatch.setattr(ai_engine, "get_finbert_pipeline", lambda: _analyzer(state))
    assert ai_engine.analyze_sentiment_ai(news) == pytest.approx(0.8)   # eșecul nu a rămas în cache


def test_fara_stiri_sau_fara_titlu(ai_engine):
    assert ai_engine.analyze_sentiment_ai([]) is None
    assert ai_engine.analyze_sentiment_ai(None) is None
    assert ai_engine.analyze_sentiment_ai([{"link": "x"}]) is None      # KeyError → None, nu excepție
