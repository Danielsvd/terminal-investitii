"""Teste pentru analytics/global_fundamental.py. Valori sintetice, calculate de mână."""
import math

import pandas as pd

import pytest

from analytics import fundamentals as fund
from analytics.global_fundamental import (_equity_per_share, earnings_yield, earnings_yield_vs_aaa, explain,
                                          fundamental_score, gn_score, info_notes, justified_pbv, reverse_dcf_growth)


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
    # DCF invers pe 10 ani: prețul cere exact 10%; livrat 2,5% → diferență 7,5 pp → (7,5 − 15) / (0 − 15) = 0,5
    price = _equity_per_share(100, 0.10, 0.09, 0.02, 0, 100, 10, 5)
    base = {"financial": False, "price": price, "dcf_na": None, "wacc": 0.09, "fcf": 100, "fcf_basis": "TTM",
            "net_debt": 0.0, "shares": 100, "pe": 40 / 3, "pbv": 3,          # E/P 7,5% / AAA 5% = 1,5×: informativ
            "eps": 2.0, "aaa": 5.0, "aaa_label": "AAA test",                  # GN 40: doar informativ
            "rev_growth": 0.025, "rev_growth_years": 3,
            "eps_growth": 0.10, "eps_growth_years": 3,                       # PEG 13,33 / 10 = 1,33: informativ
            "piotroski": {"passed": 6, "evaluable": 9},                      # 0,667
            "altman": {"value": 2.5, "variant": "Z", "zone": "grey"},        # 0,5
            "ebitda": 40.0, "coverage": 4.75}                                 # acoperire (4,75 − 1,5) / 6,5 = 0,5
    base.update(kw)
    return base


def test_scor_industrial_calculat_de_mana():
    # Datorie netă 0 → raport 0 → notă 1. Sănătate = (0,5 + 1 + 0,5) / 3 = 0,667
    # Evaluare = DCF invers 0,5 (PEG, E/P/AAA și GN informative); calitate 6/9 = 0,667 → (0,5 + 0,667 + 0,667) / 3 = 0,6111 → 61,1 NEUTRU
    res = fundamental_score(_industrial())
    by = {p["key"]: p["score"] for p in res["pillars"]}
    assert by["evaluare"] == pytest.approx(0.5, abs=1e-6)
    assert by["calitate"] == pytest.approx(6 / 9) and by["sanatate"] == pytest.approx(2 / 3)
    assert res["score"] == pytest.approx(61.1) and res["label"] == "NEUTRU"
    assert "descriptiv" in res["note"] and "IC −0,03" in res["note"]


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
    assert any(t.startswith("Datorie netă / EBITDA = 0,00") for t in strong)
    assert any(t.startswith("DCF invers: prețul cere o creștere a FCF de 10,0% pe an timp de 5 ani") for t in middle)
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


def test_ep_aaa_informativ_si_pierdere():
    # E/P față de AAA nu mai intră în notă (14.11): nici la pierdere, nici fără AAA
    c = _comp(fundamental_score(_industrial(pe=None, eps=-1.0)))["E/P față de AAA"]
    assert c["info"] and c["score"] is None and c["value"] == "pierdere"
    c = _comp(fundamental_score(_industrial(aaa=None, aaa_label="FRED indisponibil")))["E/P față de AAA"]
    assert c["info"] and c["score"] is None and "FRED indisponibil" in c["text"]
    res = fundamental_score(_industrial(aaa=None))
    assert {p["key"]: p["score"] for p in res["pillars"]}["evaluare"] == pytest.approx(0.5, abs=1e-6)
    # Pierdere și DCF invers indisponibil: PEG intră în notă cu zero (cum dădea înainte E/P)
    res = fundamental_score(_industrial(pe=None, eps=-1.0, dcf_na="test"))
    c = _comp(res)["PEG"]
    assert not c["info"] and c["score"] == 0.0 and c["value"] == "pierdere"
    assert {p["key"]: p["score"] for p in res["pillars"]}["evaluare"] == 0.0


def test_gn_informativ_nu_intra_in_scor():
    # Caz tip AAPL: P/E 35, P/BV 52 → GN 1.820, neinformativ. Fără DCF invers, nota vine din PEG:
    # 35 / (10% × 100) = 3,5 ≥ 3 → 0
    res = fundamental_score(_industrial(pe=35, pbv=52, aaa=5.3, dcf_na="test"))
    comp = _comp(res)
    assert comp["GN (Graham)"]["info"] and comp["GN (Graham)"]["score"] is None
    assert "neinformativ" in comp["GN (Graham)"]["value"]
    assert comp["PEG"]["value"] == "3,50" and comp["PEG"]["score"] == 0.0
    assert {p["key"]: p["score"] for p in res["pillars"]}["evaluare"] == 0.0   # doar PEG, GN și E/P excluse
    strong, middle, weak, missing = explain(res)
    assert not any("GN" in t for t in strong + middle + weak + missing)
    notes = info_notes(res)
    assert len(notes) == 2 and any("P/BV 52,0 > 10" in n for n in notes)
    # P/BV normal: GN afișat cu pragul, tot informativ
    notes = info_notes(fundamental_score(_industrial(pe=15, pbv=1.2)))
    assert any(n.startswith("GN = P/E × P/BV = 18,0, sub pragul") for n in notes)


# --- Evaluarea din 14.11.2026: DCF invers pe 10 ani față de creșterea livrată; PEG doar ca rezervă ---

def test_dcf_invers_10_ani_identitatea_gordon():
    # Creștere 2% = g în toți cei 10 ani: valoarea = FCF1 / (r − g) = 102 / 0,08 = 1.275 → 12,75 pe acțiune
    rev = reverse_dcf_growth(12.75, 100, 0.10, 0, 100, years=10, high_years=5)
    assert rev["bound"] is None and rev["growth"] == pytest.approx(0.02, abs=1e-9)


def test_schema_10_ani_calculata_de_mana():
    # FCF 100, creștere 10% 5 ani, apoi 8%, 6%, 4%, 2%, 2% (anul 10 = g), r 9%, g 2%, 100 de acțiuni, fără datorie
    rates = [0.10] * 5 + [0.10 + (0.02 - 0.10) * k / 5 for k in range(1, 6)]
    fcf, pv = 100.0, 0.0
    for t, gt in enumerate(rates, start=1):
        fcf *= 1 + gt
        pv += fcf / 1.09 ** t
    pv += fcf * 1.02 / 0.07 / 1.09 ** 10
    assert _equity_per_share(100, 0.10, 0.09, 0.02, 0, 100, 10, 5) == pytest.approx(pv / 100)
    assert reverse_dcf_growth(pv / 100, 100, 0.09, 0, 100, years=10, high_years=5)["growth"] == pytest.approx(0.10)


def test_diferenta_fata_de_cresterea_livrata():
    c = lambda **kw: _comp(fundamental_score(_industrial(**kw)))["DCF invers"]
    assert c(rev_growth=0.10)["score"] == pytest.approx(1.0)          # cere 10%, a livrat 10% → 0 pp
    assert c(rev_growth=-0.05)["score"] == 0.0                        # +15 pp
    assert c(rev_growth=0.025)["score"] == pytest.approx(0.5)         # +7,5 pp
    # Livrat 60%: plafonat la 30% → diferență −20 pp → notă maximă; textul spune plafonarea
    big = c(rev_growth=0.60)
    assert big["score"] == 1.0 and "plafonată la 30%" in big["text"]
    # Fără creștere livrată: DCF-ul invers nu are notă, iar PEG intră în notă (13,33 / 10 = 1,33 → 0,833)
    res = fundamental_score(_industrial(rev_growth=None))
    comp = _comp(res)
    assert comp["DCF invers"]["score"] is None and comp["DCF invers"]["value"] == "+10,0%"
    assert not comp["PEG"]["info"] and comp["PEG"]["score"] == pytest.approx((3 - 4 / 3) / 2)


def test_peg_informativ_cand_exista_dcf_si_na_la_crestere_negativa():
    comp = _comp(fundamental_score(_industrial()))
    assert comp["PEG"]["info"] and comp["PEG"]["score"] is None and comp["PEG"]["value"] == "1,33"
    comp = _comp(fundamental_score(_industrial(eps_growth=-0.05, dcf_na="test")))
    assert comp["PEG"]["score"] is None and "PEG nu are sens" in comp["PEG"]["text"]
    comp = _comp(fundamental_score(_industrial(eps_growth=None, dcf_na="test")))
    assert comp["PEG"]["score"] is None and comp["PEG"]["value"] == "N/A"   # lipsă, nu zero


def test_statement_cagr_pe_2_sau_3_ani():
    cols = pd.to_datetime(["2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31"])
    inc = pd.DataFrame({c: {"Total Revenue": v} for c, v in zip(cols, (133.1, 121.0, 110.0, 100.0))})
    r = fund.statement_cagr(inc, fund.REVENUE_ROWS)
    assert r["years"] == 3 and r["growth"] == pytest.approx(0.10)              # 133,1 / 100 = 1,1³
    r = fund.statement_cagr(inc.iloc[:, :3], fund.REVENUE_ROWS)
    assert r["years"] == 2 and r["growth"] == pytest.approx(0.10)              # 133,1 / 110 = 1,1²
    assert fund.statement_cagr(inc.iloc[:, :2], fund.REVENUE_ROWS) is None    # un singur an: prea scurt
    neg = inc.copy(); neg.iloc[0, 3] = -5.0
    assert fund.statement_cagr(neg, fund.REVENUE_ROWS) is None               # capăt ≤ 0: CAGR nedefinit
