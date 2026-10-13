"""Teste pentru analytics/bvb_fundamental.py. Valori sintetice, calculate de mână."""
import numpy as np
import pandas as pd
import pytest

from analytics.bvb_fundamental import cross_section, current_indicators, final_verdict, piotroski_simplified
from data.company_files import load_company
from tests.test_company_files import _company


@pytest.fixture
def company():
    q, a = _company()
    df, _, _ = load_company(q, a)
    return df


def test_indicatori_la_zi(company):
    # Ultimul raport T1 2026: profit TTM 53, capital 600, datorii 280, acțiuni 1.000. Preț 10 → capitalizare 10.000.
    ind = current_indicators(company, 10.0)
    assert ind["raport_bilant"] == "T1 2026" and ind["raport_flux"] == "T1 2026" and ind["raport_anual"] == "2025"
    assert ind["mcap"] == 10000
    assert ind["EP"] == pytest.approx(53 / 10000) and ind["PE"] == pytest.approx(10000 / 53)
    assert ind["BP"] == pytest.approx(0.06) and ind["PBV"] == pytest.approx(10000 / 600)
    assert ind["DY"] == pytest.approx(25 / 10000)                     # profitul distribuit din 2025
    assert ind["GN"] == pytest.approx(10000 / 53 * 10000 / 600)
    assert ind["ROE"] == pytest.approx(53 / 600) and ind["indatorare"] == pytest.approx(280 / 600)
    # marja: cifră de afaceri TTM T1 2026 = 520 − 120 + 130 = 530
    assert ind["marja"] == pytest.approx(53 / 530)
    assert ind["avertismente"] == []


def test_piotroski_pe_t1(company):
    # T1 2026 vs T1 2025 (TTM T1 2025 = 42 − 10 + 12 = 44; cifră de afaceri 440 − 100 + 120 = 460)
    # ROA 53/880 > 0 ✓; 53/880 = 0,0602 > 44/850 = 0,0518 ✓; lichiditate 380/100 > 350/100 ✓;
    # TL/active 180/880 = 0,2045 < 190/850 = 0,2235 ✓; acțiuni 1.000 = 1.000 ✓; rotație 530/880 = 0,602 > 460/850 = 0,541 ✓
    ind = current_indicators(company, 10.0)
    assert ind["PIO"] == 6
    assert [c["trece"] for c in ind["piotroski"]] == [True] * 6


def test_piotroski_criterii_lipsa_si_split():
    cur = {"ni_ttm": 10, "assets": 100, "assets_cur": 50, "liab_st": 25, "liab_lt": 10, "shares": 2000, "rev_ttm": 90}
    prev = {"ni_ttm": 12, "assets": 100, "assets_cur": 40, "liab_st": 25, "liab_lt": 20, "shares": 1000, "rev_ttm": 80}
    score, crit = piotroski_simplified(cur, prev)
    # acțiuni ×2 → split, neevaluabil; restul: ROA>0 ✓, ROA în scădere ✗, lichiditate ✓, TL ✓, rotație ✓ → 4/5 × 6 = 4,8
    assert crit[4]["trece"] is None and score == pytest.approx(4.8)
    # emisiune mică (+10%) = diluare
    score, crit = piotroski_simplified(dict(cur, shares=1100), prev)
    assert crit[4]["trece"] is False and score == pytest.approx(4)
    # fără an precedent: doar ROA > 0 → sub 5 criterii → NaN
    score, _ = piotroski_simplified(cur, None)
    assert np.isnan(score)


def test_suspiciune_split_fara_multipli(company):
    ind = current_indicators(company, 35.0)                           # 35 / 10 = 3,5× față de prețul din fișier
    assert np.isnan(ind["EP"]) and np.isnan(ind["BP"]) and np.isnan(ind["mcap"])
    assert "posibil split" in ind["avertismente"][0]
    assert ind["ROE"] == pytest.approx(53 / 600)                      # indicatorii fără preț rămân


def test_fara_pret(company):
    ind = current_indicators(company, None)
    assert np.isnan(ind["EP"]) and any("Lipsește prețul" in w for w in ind["avertismente"])


def test_percentile_si_treimi():
    # 6 companii; A cel mai bun pe toți indicatorii, F cel mai slab
    inds = {tk: {"EP": v, "BP": v, "DY": v, "PIO": v} for tk, v in zip("ABCDEF", (6, 5, 4, 3, 2, 1))}
    inds["G"] = {"EP": 0.5, "BP": np.nan, "DY": np.nan, "PIO": np.nan}          # doar 1 din 4 → fără scor
    cs = cross_section(inds)
    # A: rang 7/7 la EP, 6/6 la celelalte → (1 + 1 + 1 + 1) / 4 = 100
    assert cs.loc["A", "scor"] == pytest.approx(100)
    # F: EP 2/7, restul 1/6 → (0,2857 + 3 × 0,1667) / 4 = 0,1964 → 19,6
    assert cs.loc["F", "scor"] == pytest.approx((2 / 7 + 3 / 6) / 4 * 100)
    assert np.isnan(cs.loc["G", "scor"]) and np.isnan(cs.loc["G", "treapta"])
    assert cs.loc["A", "treapta"] == 2 and cs.loc["B", "treapta"] == 2
    assert cs.loc["C", "treapta"] == 1 and cs.loc["D", "treapta"] == 1
    assert cs.loc["E", "treapta"] == 0 and cs.loc["F", "treapta"] == 0
    assert (cs["n_comparate"] == 6).all()


def test_sub_6_companii_fara_treapta():
    inds = {tk: {"EP": v, "BP": v, "DY": v, "PIO": v} for tk, v in zip("ABCDE", (5, 4, 3, 2, 1))}
    cs = cross_section(inds)
    assert cs["scor"].notna().all() and cs["treapta"].isna().all()


def test_verdict_doi_pasi():
    lab, _, text = final_verdict(2, 85)
    assert lab == "CANDIDAT, MOMENT FAVORABIL" and "+9.2%" in text and "N = 386" in text
    assert final_verdict(2, 60)[0] == "CANDIDAT, FĂRĂ SEMNAL DE MOMENT"
    assert final_verdict(2, 39.9)[0] == "CANDIDAT, TREND NEGATIV"
    assert final_verdict(0, 95)[0] == "NEFAVORABIL (MOMENTUL NU COMPENSEAZĂ)"
    assert final_verdict(np.nan, 90)[0] == "DATE INSUFICIENTE"
    lab, _, text = final_verdict(1, None)
    assert lab == "FUNDAMENTAL NEUTRU" and "Momentul nu se poate calcula" in text
