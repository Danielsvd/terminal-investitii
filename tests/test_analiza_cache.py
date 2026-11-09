"""„2. Analiză Companie”: un clic pe un widget nu mai trimite cereri Yahoo / FinBERT deja făcute (Etapa 5, viteză).
Valori sintetice; prophet / transformers substituite doar dacă lipsesc."""
import sys
import types

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def ai_engine():
    added = []
    for name, attrs in (("prophet", {"Prophet": object}), ("transformers", {"pipeline": lambda *a, **k: None})):
        try:
            __import__(name)
        except ImportError:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod
            added.append(name)
    import ai_engine as mod
    yield mod
    for name in added:
        sys.modules.pop(name, None)


def _ticker_factory(state, fail=False, holders=True):
    class T:
        def __init__(self, sym, *a, **k):
            self.sym = sym

        @property
        def info(self):
            state["info"] += 1
            if fail:
                raise RuntimeError("429 Too Many Requests")
            return {"companyOfficers": [{"title": "CEO", "name": "Ana Pop"}], "heldPercentInstitutions": 0.6,
                    "heldPercentInsiders": 0.02}

        @property
        def major_holders(self):
            state["major"] += 1
            return None

        @property
        def institutional_holders(self):
            state["inst"] += 1
            if not holders:
                return None
            return pd.DataFrame({"Holder": ["Fond A"], "pctHeld": [0.08], "Value": [1e9]})

        @property
        def options(self):
            state["options"] += 1
            return ()
    return T


def _state():
    return {"info": 0, "major": 0, "inst": 0, "options": 0}


def test_actionariatul_din_cache(ai_engine, monkeypatch):
    st_ = _state()
    monkeypatch.setattr(ai_engine.yf, "Ticker", _ticker_factory(st_))
    ai_engine._ownership_cached.clear()
    summ, det, ceo, inst = ai_engine.get_detailed_ownership_and_execs("XYZ")
    assert ceo == "Ana Pop" and inst == pytest.approx(60.0)
    # 100 − (2 + 60) = 38% retail
    assert summ["Procent (%)"].tolist() == pytest.approx([2.0, 60.0, 38.0])
    assert det["Deținere (%)"].iloc[0] == pytest.approx(8.0)
    before = dict(st_)
    ai_engine.get_detailed_ownership_and_execs("XYZ")
    assert st_ == before                               # al doilea apel: fără cereri Yahoo
    ai_engine._ownership_cached.clear()


def test_actionariatul_eroare_nu_ramane_in_cache(ai_engine, monkeypatch):
    st_ = _state()
    monkeypatch.setattr(ai_engine.yf, "Ticker", _ticker_factory(st_, fail=True))
    ai_engine._ownership_cached.clear()
    assert ai_engine.get_detailed_ownership_and_execs("XYZ") == (None, None, "N/A", None)
    monkeypatch.setattr(ai_engine.yf, "Ticker", _ticker_factory(st_))
    assert ai_engine.get_detailed_ownership_and_execs("XYZ")[2] == "Ana Pop"   # Yahoo și-a revenit: se recitește
    ai_engine._ownership_cached.clear()


def test_optiuni_simbol_cu_sufix_fara_cerere_la_fiecare_clic(ai_engine, monkeypatch):
    st_ = _state()
    monkeypatch.setattr(ai_engine.yf, "Ticker", _ticker_factory(st_))
    ai_engine._options_analysis_cached.clear()
    for _ in range(3):
        data, msg = ai_engine.get_options_analysis_ai("TLV.RO")
        assert data is None and msg.startswith("Fără opțiuni listate în Yahoo")
    assert st_["options"] == 1                         # o singură cerere, apoi din cache
    ai_engine._options_analysis_cached.clear()


def test_optiuni_simbol_sua_gol_se_reincearca(ai_engine, monkeypatch):
    st_ = _state()
    monkeypatch.setattr(ai_engine.yf, "Ticker", _ticker_factory(st_))
    ai_engine._options_analysis_cached.clear()
    for _ in range(2):
        data, msg = ai_engine.get_options_analysis_ai("AAPLX")
        assert data is None and "Yahoo nu a trimis lista de expirări" in msg
    assert st_["options"] == 2                         # la SUA golul e tratat ca răspuns întâmplător: nu se păstrează
    ai_engine._options_analysis_cached.clear()


def test_eticheta_finbert_pe_titlu_din_cache(ai_engine, monkeypatch):
    calls = {"n": 0}

    def analyzer(text):
        calls["n"] += 1
        return [{"label": "negative" if "down" in text else "positive", "score": 0.9}]

    monkeypatch.setattr(ai_engine, "get_finbert_pipeline", lambda: analyzer)
    ai_engine._finbert_label.clear()
    assert ai_engine.get_gemini_sentiment_label("shares down") == ("Negativ", "impact-neg", "📉")
    assert ai_engine.get_gemini_sentiment_label("shares down") == ("Negativ", "impact-neg", "📉")
    assert ai_engine.get_gemini_sentiment_label("profit up") == ("Pozitiv", "impact-poz", "📈")
    assert calls["n"] == 2                             # titlul repetat nu mai trece prin model
    ai_engine._finbert_label.clear()


def test_eticheta_finbert_eroare_este_na_nu_neutru(ai_engine, monkeypatch):
    def broken(text):
        raise RuntimeError("model indisponibil")

    monkeypatch.setattr(ai_engine, "get_finbert_pipeline", lambda: broken)
    ai_engine._finbert_label.clear()
    assert ai_engine.get_gemini_sentiment_label("shares down") == ("N/A", "impact-neu", "❔")
    # eroarea nu rămâne în cache: când modelul merge, titlul primește eticheta reală
    monkeypatch.setattr(ai_engine, "get_finbert_pipeline", lambda: (lambda t: [{"label": "neutral", "score": 0.7}]))
    assert ai_engine.get_gemini_sentiment_label("shares down") == ("Neutru", "impact-neu", "➡️")
    ai_engine._finbert_label.clear()
    monkeypatch.setattr(ai_engine, "get_finbert_pipeline", lambda: (lambda t: [{"label": "LABEL_7", "score": 0.7}]))
    assert ai_engine.get_gemini_sentiment_label("altceva") == ("N/A", "impact-neu", "❔")
    ai_engine._finbert_label.clear()
