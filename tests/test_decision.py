"""Teste pentru analytics/decision.py (sinteza „Decizie Master AI” în trei pași). Valori sintetice."""
import math

import pytest

from analytics.decision import characterize, context_flags, count_alerts, moment_band


@pytest.mark.parametrize("score, band", [(80, "high"), (79.9, "mid"), (40, "mid"), (39.9, "low"),
                                         (None, None), (float("nan"), None), (True, None)])
def test_moment_band(score, band):
    assert moment_band(score) == band


def test_characterize_bvb_neutru_moment_favorabil():
    txt = characterize(is_bvb=True, fund_label="NEUTRU", fund_score=54, moment_score=100,
                       history_text="Istoric, combinația aceasta: +4,6% față de media BVB la 6 luni.")
    assert txt.startswith("Fundamental, compania e la mijloc față de celelalte companii BVB din fișierele tale (54/100)")  # BVB: întreg
    assert "trendul și momentumul sunt pozitive (100/100)" in txt
    assert "Tehnicul e mai bun decât fundamentalul." in txt
    assert txt.endswith("+4,6% față de media BVB la 6 luni.")
    assert "netestat" not in txt


def test_characterize_global_pillars_and_untested():
    pillars = [("Evaluare", 12), ("Calitate", 89), ("Sănătate", 100), ("Rentabilitate", None)]
    txt = characterize(is_bvb=False, fund_label="NEUTRU", fund_score=66.8, moment_score=30,
                       pillars=pillars, n_alerts=2)
    assert "pe pilonii cu date (66,8/100)" in txt
    assert "; trendul e negativ (30/100)." in txt
    assert "Punct tare: Sănătate 100/100; punct slab: Evaluare 12/100." in txt
    assert "2 alerte de context" in txt
    assert txt.endswith("(netestat).")


@pytest.mark.parametrize("label, score, conn, cons", [
    ("FAVORABIL", 90, ", iar ", "Ambii pași sunt aliniați."),
    ("FAVORABIL", 20, ", dar ", "prețul e încă în scădere"),
    ("NEFAVORABIL", 85, ", dar ", "Creșterea prețului nu e susținută de fundamental."),
    ("NEFAVORABIL", 10, ", iar ", "Niciun pas nu susține compania acum."),
])
def test_characterize_connectors(label, score, conn, cons):
    txt = characterize(is_bvb=False, fund_label=label, fund_score=50, moment_score=score)
    assert conn in txt and cons in txt


def test_characterize_missing_data():
    txt = characterize(is_bvb=False, fund_label="DATE INSUFICIENTE", moment_score=None)
    assert txt.startswith("Fundamentalul nu se poate evalua (date insuficiente); momentul nu se poate calcula")
    # pilonii fără date nu dau „punct tare/slab"; o singură notă nu face comparație
    txt2 = characterize(is_bvb=False, fund_label="FAVORABIL", fund_score=80, moment_score=50,
                        pillars=[("Evaluare", 90), ("Calitate", None)])
    assert "Punct tare" not in txt2


def test_characterize_has_no_buy_sell_words():
    for label in ("FAVORABIL", "NEUTRU", "NEFAVORABIL"):
        for m in (10, 60, 95):
            txt = characterize(is_bvb=True, fund_label=label, fund_score=50, moment_score=m).upper()
            for word in ("CUMPĂRĂ", "VINDE", "EVITĂ", "BUY", "SELL", "ACUMULEAZĂ"):
                assert word not in txt


def _levels(flags):
    return {f["title"]: f["level"] for f in flags}


def test_context_flags_order_and_levels():
    flags = context_flags(is_bvb=False, altman_zone="distress", spread=-0.35, spread_label="SUA",
                          regime_msg="🚨 REGIM PANICĂ (Volatilitate Extremă)", opt_pc=1.3, opt_iv=55.0,
                          iv_rank=None, iv_days=3, sentiment=0.4, n_news=8, inst_pct=72, q_ratio=0.5, rsi=78, rvol=2.0)
    lv = _levels(flags)
    assert lv["Altman Z"] == "alerta" and lv["Curba randamentelor"] == "alerta"
    assert lv["Regim K-Means"] == "atentie"   # model netestat: cel mult galben
    assert lv["Opțiuni P/C"] == "atentie" and lv["Calitatea profitului"] == "atentie"
    assert lv["Știri"] == "pozitiv"
    assert lv["IV la bani"] == "info" and "IV Rank N/A (3 din 20 zile" in [f for f in flags if f["title"] == "IV la bani"][0]["text"]
    order = [f["level"] for f in flags]
    assert order == sorted(order, key=["alerta", "atentie", "pozitiv", "info"].index)
    assert count_alerts(flags) == 5


def test_context_flags_bvb_and_financial_rules():
    flags = context_flags(is_bvb=True, is_financial=True, altman_zone="distress", spread=0.8,
                          inst_pct=7, q_ratio=0.2)
    titles = {f["title"] for f in flags}
    assert "Instituționali" not in titles       # fără sens la BVB
    assert "Altman Z" not in titles              # nu se aplică financiarelor
    assert "Calitatea profitului" not in titles  # nici CFO/profit la financiare
    assert count_alerts(flags) == 0


def test_context_flags_missing_values_give_no_signal():
    flags = context_flags(is_bvb=False, spread=None, spread_label="curba SUA indisponibilă",
                          opt_pc=None, opt_iv=float("nan"), sentiment=None, n_news=0,
                          inst_pct=None, q_ratio=None, rsi=None, rvol=None)
    lv = _levels(flags)
    assert lv == {"Curba randamentelor": "info", "Știri": "info"}
    assert count_alerts(flags) == 0


def test_context_flags_iv_rank_and_news_caveat():
    flags = context_flags(is_bvb=False, spread=0.5, opt_pc=0.9, opt_iv=30.0, iv_rank=85, iv_days=25,
                          sentiment=-0.3, n_news=5)
    by = {f["title"]: f for f in flags}
    assert by["IV Rank"]["level"] == "atentie" and "85%" in by["IV Rank"]["text"]
    assert by["Opțiuni P/C"]["level"] == "info"
    assert by["Știri"]["level"] == "atentie" and "nesigură" in by["Știri"]["text"]


def test_flags_do_not_return_scores():
    flags = context_flags(is_bvb=False, spread=1.0)
    for f in flags:
        assert set(f) == {"level", "title", "text"}
        assert not any(isinstance(v, (int, float)) and not math.isnan(v) for v in f.values() if not isinstance(v, str))
