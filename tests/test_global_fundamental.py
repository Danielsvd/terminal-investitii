"""Teste pentru analytics/global_fundamental.py. Valori sintetice, calculate de mână."""
import math

import pytest

from analytics import fundamentals as fund
from analytics.global_fundamental import (earnings_yield, earnings_yield_vs_aaa, explain, fundamental_score,
                                          gn_score, info_notes, justified_pbv, reverse_dcf_growth)


def test_dcf_invers_identitatea_gordon():
    # Creștere constantă 2% = g terminal: valoarea = FCF1 / (r − g) = 102 / 0,08 = 1.275 → 12,75 pe acțiune
    rev = reverse_dcf_growth(12.75, 100, 0.10, 0, 100)
    assert rev["bound"] is None and rev["growth"] == pytest.approx(0.02, abs=1e-9)


def test_dcf_invers_coincide_cu_dcf_din_calculator():
    # Prețul = valoarea DCF-ului din calculator la creștere 12%: DCF-ul invers trebuie să găsească 12%
    price = fund.dcf_fcf(100, 0.12, 0.09, 0.02, 250, 50)["per_share"]
    assert reverse_dcf_growth(price, 100, 0.09, 250, 50)["growth"] == pytest.approx(0.12, abs=1e-9)


def test_dcf_invers_limite_si_date_lipsa():
    assert reverse_dcf_growth(0.01, 100, 0.10, 0, 100)["bound"] == "low"
    assert reverse_dcf_growth(1e6, 100, 0.10, 0, 100)["bound"] == "high"
    assert reverse_dcf_growth(10, -5, 0.10, 0, 100) is None            # FCF negativ
    assert reverse_dcf_growth(10, 100, 0.02, 0, 100) is None           # r ≤ g
    assert reverse_dcf_growth(10, 100, 0.10, None, 100) is None


def test_gn_scara_logaritmica():
    assert gn_score(15, 1.5) == (pytest.approx(22.5), 1.0)
    gn, s = gn_score(15, 3)                                            # 45 = jumătatea scării logaritmice 22,5–90
    assert gn == pytest.approx(45) and s == pytest.approx(0.5)
    assert gn_score(-10, 2) == (None, None) and gn_score(None, 2) == (None, None)


def _industrial(**kw):
    price = fund.dcf_fcf(100, 0.10, 0.09, 0.02, 0, 100)["per_share"]   # prețul cere exact 10% → notă 0,5
    base = {"financial": False, "price": price, "dcf_na": None, "wacc": 0.09, "fcf": 100, "fcf_basis": "TTM",
            "net_debt": 0.0, "shares": 100, "pe": 40 / 3, "pbv": 3,          # E/P 7,5% / AAA 5% = 1,5× → 0,5
            "eps": 2.0, "aaa": 5.0, "aaa_label": "AAA test",                  # GN 40: doar informativ
            "piotroski": {"passed": 6, "evaluable": 9},                      # 0,667
            "altman": {"value": 2.5, "variant": "Z", "zone": "grey"},        # 0,5
            "ebitda": 40.0, "coverage": 4.75}                                 # acoperire (4,75 − 1,5) / 6,5 = 0,5
    base.update(kw)
    return base


def test_scor_industrial_calculat_de_mana():
    # Datorie netă 0 → raport 0 → notă 1. Sănătate = (0,5 + 1 + 0,5) / 3 = 0,667
    # Evaluare (0,5 + 0,5) / 2 = 0,5; calitate 6/9 = 0,667 → (0,5 + 0,667 + 0,667) / 3 = 0,6111 → 61,1 NEUTRU
    res = fundamental_score(_industrial())
    by = {p["key"]: p["score"] for p in res["pillars"]}
    assert by["evaluare"] == pytest.approx(0.5, abs=1e-6)
    assert by["calitate"] == pytest.approx(6 / 9) and by["sanatate"] == pytest.approx(2 / 3)
    assert res["score"] == pytest.approx(61.1) and res["label"] == "NEUTRU"
    assert "netestate" in res["note"]


def test_datorie_neta_ebitda_si_cazuri_limita():
    # Datorie netă 100 la EBITDA 40 = 2,5 → (2,5 − 4) / (1 − 4) = 0,5
    res = fundamental_score(_industrial(net_debt=100.0, price=None, dcf_na="test"))
    comp = {c["label"]: c for p in res["pillars"] for c in p["components"]}
    assert comp["Datorie netă / EBITDA"]["score"] == pytest.approx(0.5)
    assert comp["DCF invers"]["score"] is None and "nu se aplică" in comp["DCF invers"]["text"]
    # EBITDA negativ: cu datorie netă → 0; cu numerar net → 1
    res = fundamental_score(_industrial(ebitda=-5.0, net_debt=10.0))
    assert {c["label"]: c["score"] for p in res["pillars"] for c in p["components"]}["Datorie netă / EBITDA"] == 0.0
    res = fundamental_score(_industrial(ebitda=-5.0, net_debt=-10.0))
    assert {c["label"]: c["score"] for p in res["pillars"] for c in p["components"]}["Datorie netă / EBITDA"] == 1.0


def test_date_lipsa_nu_devin_zero():
    # Fără Piotroski (sub 6 criterii) și fără nimic la sănătate: un singur pilon → fără scor
    res = fundamental_score(_industrial(piotroski={"passed": 3, "evaluable": 4}, altman={}, ebitda=None,
                                        net_debt=None, coverage=None, dcf_na="x"))
    assert res["score"] is None and res["label"] == "DATE INSUFICIENTE"
    # Piotroski pe 7 criterii evaluabile: 5/7, nu 5/9
    res = fundamental_score(_industrial(piotroski={"passed": 5, "evaluable": 7}))
    assert {p["key"]: p["score"] for p in res["pillars"]}["calitate"] == pytest.approx(5 / 7)


def test_banca_pbv_justificat():
    # ROE 12%, ke 10%, g 2%: P/BV justificat = 0,10 / 0,08 = 1,25. P/BV 1,0 → raport 0,8 → (0,8 − 1,3) / (0,7 − 1,3) = 0,833
    # ROE − ke = +2 pp → (0,02 + 0,05) / 0,10 = 0,7. Scor (0,833 + 0,7) / 2 = 76,7 → FAVORABIL
    assert justified_pbv(0.12, 0.10) == pytest.approx(1.25)
    res = fundamental_score({"financial": True, "roe": 0.12, "ke": 0.10, "pbv": 1.0})
    assert [p["key"] for p in res["pillars"]] == ["evaluare", "rentabilitate"]
    assert res["score"] == pytest.approx(76.7) and res["label"] == "FAVORABIL"


def test_banca_roe_sub_g_si_ke_lipsa():
    res = fundamental_score({"financial": True, "roe": 0.01, "ke": 0.10, "pbv": 0.5})
    assert res["pillars"][0]["score"] == 0.0                         # ROE sub g: P/BV justificat negativ
    assert res["pillars"][1]["score"] == 0.0                         # −9 pp, sub −5 pp
    res = fundamental_score({"financial": True, "roe": 0.15, "ke": None, "pbv": 1.0})
    assert res["score"] is None and all(p["score"] is None for p in res["pillars"])


def test_explicatia_pe_note():
    strong, middle, weak, missing = explain(fundamental_score(_industrial(coverage=None)))
    assert any(t.startswith("Datorie netă / EBITDA = 0.00") for t in strong)
    assert any(t.startswith("DCF invers: prețul cere o creștere a FCF de 10.0%") for t in middle)
    assert any(t.startswith("Acoperirea dobânzii: N/A") for t in missing) and weak == []
    assert math.isclose(fundamental_score(_industrial())["score"], 61.1)


def test_ep_fata_de_aaa_scara():
    # E/P 5% la AAA 5% = 1× → 0; E/P 10% = 2× → 1; E/P 7,5% = 1,5× → 0,5
    assert earnings_yield_vs_aaa(0.05, 5.0) == (pytest.approx(1.0), 0.0)
    assert earnings_yield_vs_aaa(0.10, 5.0) == (pytest.approx(2.0), 1.0)
    assert earnings_yield_vs_aaa(0.075, 5.0)[1] == pytest.approx(0.5)
    assert earnings_yield_vs_aaa(0.05, None) == (None, None) and earnings_yield_vs_aaa(None, 5.0) == (None, None)


def test_randamentul_profitului_din_pe_si_eps():
    assert earnings_yield(20, 5.0) == (pytest.approx(0.05), False)
    assert earnings_yield(None, -1.2) == (None, True)                 # pierdere: Yahoo nu dă P/E
    assert earnings_yield(None, None) == (None, False)                # lipsă, nu zero


def _comp(res):
    return {c["label"]: c for p in res["pillars"] for c in p["components"]}


def test_pierdere_da_zero_iar_aaa_lipsa_da_na():
    c = _comp(fundamental_score(_industrial(pe=None, eps=-1.0)))["E/P față de AAA"]
    assert c["score"] == 0.0 and c["value"] == "pierdere"
    c = _comp(fundamental_score(_industrial(aaa=None, aaa_label="FRED indisponibil")))["E/P față de AAA"]
    assert c["score"] is None and "FRED indisponibil" in c["text"]
    # Fără AAA, evaluarea rămâne doar pe DCF invers (0,5), nu scade
    res = fundamental_score(_industrial(aaa=None))
    assert {p["key"]: p["score"] for p in res["pillars"]}["evaluare"] == pytest.approx(0.5, abs=1e-6)


def test_gn_informativ_nu_intra_in_scor():
    # Caz tip AAPL: P/E 35, P/BV 52 → GN 1.820, neinformativ; E/P 2,86% / 5,3% = 0,54× → 0
    res = fundamental_score(_industrial(pe=35, pbv=52, aaa=5.3, dcf_na="test"))
    comp = _comp(res)
    assert comp["GN (Graham)"]["info"] and comp["GN (Graham)"]["score"] is None
    assert "neinformativ" in comp["GN (Graham)"]["value"]
    assert {p["key"]: p["score"] for p in res["pillars"]}["evaluare"] == 0.0   # doar E/P, GN exclus
    strong, middle, weak, missing = explain(res)
    assert not any("GN" in t for t in strong + middle + weak + missing)
    notes = info_notes(res)
    assert len(notes) == 1 and "P/BV 52.0 > 10" in notes[0]
    # P/BV normal: GN afișat cu pragul, tot informativ
    notes = info_notes(fundamental_score(_industrial(pe=15, pbv=1.2)))
    assert notes[0].startswith("GN = P/E × P/BV = 18.0, sub pragul")
