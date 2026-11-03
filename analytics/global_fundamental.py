"""Scorul fundamental pentru acțiunile din SUA și UE. Funcții pure (fără st.*, fără rețea).

Structura e aceeași ca la BVB (scorul fundamental = scorul companiei; momentul e separat și,
pe SUA/UE, fără verdict), dar etalonul e absolut, nu relativ la un univers: pragurile vin din
literatură și din convențiile proiectului și sunt NETESTATE pe datele noastre până la testul
istoric. Eticheta o spune.

Companii nefinanciare, trei piloni cu pondere egală (fiecare 0–1, media componentelor cu date):
  Evaluare   DCF invers (creșterea FCF pe care o cere prețul) și randamentul profitului E/P față de
             randamentul obligațiunilor AAA (criteriul lui Graham: E/P ≥ 2 × AAA). GN = P/E × P/BV
             se afișează doar informativ: la companiile cu răscumpărări masive capitalul propriu
             contabil e micșorat, iar P/BV și GN nu mai măsoară evaluarea.
  Calitate   Piotroski F-Score (9 criterii, ultimii doi ani fiscali), pe criteriile evaluabile
  Sănătate   zona Altman, datorie netă / EBITDA, acoperirea dobânzii
Bănci și asigurători (bilanțul lor nu se citește ca al unei companii industriale):
  Evaluare       P/BV față de P/BV justificat = (ROE − g) / (ke − g)
  Rentabilitate  ROE față de costul capitalului propriu ke (CAPM)

O componentă fără date e None și nu intră în medie; nu devine 0. Scorul cere cel puțin
doi piloni cu date.
"""
import math

from analytics.scoring import UNTESTED_NOTE, VERDICT_BANDS, linear_score
from data.helpers import ro

TERMINAL_GROWTH = 0.02          # g terminal al DCF-ului invers și al P/BV justificat (ca valoarea implicită din calculator)
REVERSE_DCF_BOUNDS = (-0.50, 1.00)
REVERSE_DCF_SCALE = (0.20, 0.0)      # prețul cere ≥ 20% creștere/an → 0; ≤ 0% → 1
GN_LIMIT = 22.5                      # Graham: P/E × P/BV ≤ 22,5
GN_WORST = 90.0                      # de 4 ori pragul (de ex. P/E 30 × P/BV 3) → 0; scară logaritmică
GN_UNINFORMATIVE_PBV = 10.0          # peste acest P/BV, GN e marcat „neinformativ” (capital contabil micșorat)
EY_AAA_SCALE = (1.0, 2.0)            # E/P / AAA ≤ 1 → 0; ≥ 2 → 1 (Graham cerea E/P ≥ 2 × AAA)
ND_EBITDA_SCALE = (4.0, 1.0)         # datorie netă / EBITDA ≥ 4 → 0; ≤ 1 → 1
COVERAGE_SCALE = (1.5, 8.0)          # EBIT / dobânzi ≤ 1,5 → 0; ≥ 8 → 1
ALTMAN_SCORE = {"safe": 1.0, "grey": 0.5, "distress": 0.0}
PIOTROSKI_MIN_EVALUABLE = 6
PBV_RATIO_SCALE = (1.3, 0.7)         # P/BV / P/BV justificat ≥ 1,3 → 0; ≤ 0,7 → 1
ROE_SPREAD_SCALE = (-0.05, 0.05)     # ROE − ke: −5 pp → 0; +5 pp → 1
MIN_PILLARS = 2

PILLAR_LABELS = {"evaluare": "Evaluare", "calitate": "Calitate", "sanatate": "Sănătate financiară",
                 "rentabilitate": "Rentabilitate"}
# Testul istoric (tools/backtest_fundamental.py, rulat de Daniel pe 22.10.2026): 56 de companii S&P 500,
# 2021–2026, exces față de media universului. IC 6 luni +0,04, 12 luni −0,03; banda ≥ 70 a rămas în urmă
# (median −10,9% la 12 luni, 33% din cazuri peste medie). Decizia lui Daniel: scorul și etichetele rămân,
# dar ca descriere, nu ca predicție (la fel ca momentul tehnic pe SUA).
TEST_NOTE = ("Scor descriptiv, fără putere de predicție demonstrată: în testul istoric pe SUA (56 de companii "
             "S&P 500, 2022–2026) corelația scor → randament a fost ≈ 0 (IC −0,03 la 12 luni), iar banda "
             "FAVORABIL a rămas în urma mediei (median −10,9% la 12 luni). Etichetele descriu compania, nu "
             "prezic prețul. UE: netestat.")


def _num(x):
    return x is not None and not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def _pct(x, digits=1):
    return f"{ro(x * 100):+.{digits}f}%" if _num(x) else "N/A"


def _component(label, value_txt, score, text, info=False):
    """info=True: componenta se afișează, dar nu intră în scor (score rămâne None)."""
    return {"label": label, "value": value_txt, "score": None if info else score, "text": text, "info": info}


# --- Evaluare -----------------------------------------------------------------

def _equity_per_share(fcf0, growth, r, g, net_debt, shares, years):
    """Valoarea capitalului pe acțiune din DCF (poate fi negativă). Aceeași schemă ca fund.dcf_fcf."""
    fcf, pv = fcf0, 0.0
    step = (g - growth) / (years - 1) if years > 1 else 0.0
    for t in range(1, years + 1):
        fcf *= 1 + growth + step * (t - 1)
        pv += fcf / (1 + r) ** t
    tv = fcf * (1 + g) / (r - g)
    return (pv + tv / (1 + r) ** years - net_debt) / shares


def reverse_dcf_growth(price, fcf0, discount_rate, net_debt, shares, terminal_growth=TERMINAL_GROWTH, years=5):
    """Creșterea FCF din primul an pe care o cere prețul (DCF invers).

    Aceeași schemă ca DCF-ul din calculator: creșterea scade liniar până la g terminal în anul 5,
    apoi Gordon. Se caută prin bisecție creșterea pentru care valoarea pe acțiune = prețul.
    Întoarce {"growth", "bound"}: bound = "low" dacă și cu −50% valoarea depășește prețul,
    "high" dacă nici cu +100% nu ajunge la preț, altfel None. None dacă datele nu permit modelul.
    """
    if not all(_num(x) for x in (price, fcf0, discount_rate, net_debt, shares, terminal_growth)):
        return None
    if price <= 0 or fcf0 <= 0 or shares <= 0 or discount_rate <= terminal_growth:
        return None
    lo, hi = REVERSE_DCF_BOUNDS
    f = lambda gr: _equity_per_share(fcf0, gr, discount_rate, terminal_growth, net_debt, shares, years) - price
    if f(lo) >= 0:
        return {"growth": lo, "bound": "low"}
    if f(hi) <= 0:
        return {"growth": hi, "bound": "high"}
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return {"growth": (lo + hi) / 2, "bound": None}


def gn_score(pe, pbv):
    """(GN, notă 0–1) pentru GN = P/E × P/BV, pe scară logaritmică: ≤ 22,5 → 1, ≥ 90 → 0.
    P/E sau P/BV lipsă ori ≤ 0 (pierdere, capital negativ) → (None, None)."""
    if not (_num(pe) and _num(pbv)) or pe <= 0 or pbv <= 0:
        return None, None
    gn = pe * pbv
    return gn, linear_score(math.log(gn), math.log(GN_WORST), math.log(GN_LIMIT))


def earnings_yield(pe, eps):
    """(E/P, pierdere): E/P = 1 / P/E când P/E > 0. Fără P/E, dar cu EPS ≤ 0 → (None, True):
    compania e pe pierdere, randamentul profitului e negativ. Altfel (None, False) = date lipsă."""
    if _num(pe) and pe > 0:
        return 1.0 / pe, False
    if _num(eps) and eps <= 0:
        return None, True
    return None, False


def earnings_yield_vs_aaa(ey, aaa_pct):
    """(raport E/P / AAA, notă 0–1). aaa_pct e în procente (5,3 = 5,3%). ≤ 1 → 0, ≥ 2 → 1.
    Date lipsă sau AAA ≤ 0 → (None, None)."""
    if not (_num(ey) and _num(aaa_pct)) or aaa_pct <= 0:
        return None, None
    ratio = ey / (aaa_pct / 100)
    return ratio, linear_score(ratio, *EY_AAA_SCALE)


def ey_component(inp):
    """Componenta de evaluare „E/P față de AAA” (criteriul lui Graham pe randament)."""
    label = "E/P față de AAA"
    aaa, aaa_label = inp.get("aaa"), inp.get("aaa_label") or "randament AAA"
    ey, loss = earnings_yield(inp.get("pe"), inp.get("eps"))
    if loss:
        return _component(label, "pierdere", 0.0,
                          "E/P față de AAA: compania e pe pierdere în ultimele 12 luni, randamentul profitului e negativ "
                          "(notă zero).")
    if ey is None:
        return _component(label, "N/A", None, "E/P față de AAA: N/A (lipsește P/E).")
    if not _num(aaa) or aaa <= 0:
        return _component(label, "N/A", None, f"E/P față de AAA: N/A ({aaa_label}).")
    ratio, score = earnings_yield_vs_aaa(ey, aaa)
    return _component(label, f"{ro(ratio):.2f}×", score,
                      f"Randamentul profitului E/P = 1 / P/E = {ro(ey * 100):.2f}% față de {ro(aaa):.2f}% ({aaa_label}): "
                      f"{ro(ratio):.2f}× (Graham cerea minimum 2×; ≤ 1× → zero, ≥ 2× → notă maximă).")


def gn_component(inp):
    """GN = P/E × P/BV, doar informativ (nu intră în scor). Marcat „neinformativ” peste P/BV 10."""
    pe, pbv = inp.get("pe"), inp.get("pbv")
    gn, _ = gn_score(pe, pbv)
    if gn is None:
        return _component("GN (Graham)", "N/A", None,
                          "GN = P/E × P/BV: N/A (P/E sau P/BV lipsă sau negativ: pierdere ori capital propriu negativ). "
                          "Informativ, nu intră în scor.", info=True)
    if pbv > GN_UNINFORMATIVE_PBV:
        return _component("GN (Graham)", f"{ro(gn):.1f} (neinformativ)", None,
                          f"GN = P/E × P/BV = {ro(gn):.1f}, neinformativ: P/BV {ro(pbv):.1f} > {ro(GN_UNINFORMATIVE_PBV):.0f}, "
                          "capitalul propriu contabil e micșorat (răscumpărări de acțiuni, pierderi acumulate), "
                          "deci P/BV nu mai măsoară evaluarea. Nu intră în scor.", info=True)
    rel = "sub" if gn <= GN_LIMIT else "peste"
    return _component("GN (Graham)", f"{ro(gn):.1f}", None,
                      f"GN = P/E × P/BV = {ro(gn):.1f}, {rel} pragul lui Graham de 22,5. Informativ: nu intră în scor "
                      "(evaluarea pe profit se face prin E/P față de AAA).", info=True)


def valuation_components(inp):
    out = []
    dcf_na = inp.get("dcf_na")
    wacc = inp.get("wacc")
    rev = None
    if dcf_na:
        out.append(_component("DCF invers", "N/A", None, f"DCF invers: nu se aplică ({dcf_na})"))
    elif not _num(wacc):
        out.append(_component("DCF invers", "N/A", None,
                              "DCF invers: N/A (WACC nu se poate calcula: lipsește rata fără risc, beta sau capitalizarea)."))
    else:
        rev = reverse_dcf_growth(inp.get("price"), inp.get("fcf"), wacc, inp.get("net_debt"), inp.get("shares"))
        if rev is None:
            reason = ("FCF negativ sau zero" if _num(inp.get("fcf")) and inp["fcf"] <= 0
                      else "lipsesc FCF, datoria netă sau numărul de acțiuni")
            out.append(_component("DCF invers", "N/A", None, f"DCF invers: N/A ({reason})."))
        else:
            g = rev["growth"]
            score = linear_score(g, *REVERSE_DCF_SCALE)
            if rev["bound"] == "low":
                val, txt = "< −50%", "prețul e sub valoarea FCF-ului actual chiar cu o scădere de 50% pe an"
            elif rev["bound"] == "high":
                val, txt = "> +100%", "prețul cere o creștere a FCF de peste 100% pe an"
            else:
                val, txt = _pct(g), f"prețul cere o creștere a FCF de {ro(g * 100):.1f}% pe an"
            base = inp.get("fcf_basis") or "FCF"
            out.append(_component("DCF invers", val, score,
                                  f"DCF invers: {txt} în primul an, care scade spre {ro(TERMINAL_GROWTH * 100):.0f}% în anul 5 "
                                  f"(WACC {ro(wacc * 100):.1f}%, pornind de la {base}). Sub 0% → notă maximă, peste 20% → zero."))
    out.append(ey_component(inp))
    out.append(gn_component(inp))
    return out, rev


# --- Calitate și sănătate -----------------------------------------------------

def quality_components(inp):
    p = inp.get("piotroski") or {}
    passed, evaluable = p.get("passed"), p.get("evaluable")
    if not _num(passed) or not _num(evaluable) or evaluable < PIOTROSKI_MIN_EVALUABLE:
        n = int(evaluable) if _num(evaluable) else 0
        return [_component("Piotroski F-Score", "N/A", None,
                           f"Piotroski: N/A ({n} criterii evaluabile din 9; minimum {PIOTROSKI_MIN_EVALUABLE}).")]
    score = passed / evaluable
    return [_component("Piotroski F-Score", f"{ro(passed):.0f}/{ro(evaluable):.0f}", score,
                       f"Piotroski: {ro(passed):.0f} din {ro(evaluable):.0f} criterii evaluabile îndeplinite "
                       "(ultimul an fiscal față de cel precedent).")]


def health_components(inp):
    out = []
    alt = inp.get("altman") or {}
    zone = alt.get("zone")
    if zone in ALTMAN_SCORE:
        names = {"safe": "zona sigură", "grey": "zona gri", "distress": "zona de dificultate"}
        out.append(_component(f"Altman {alt.get('variant') or 'Z'}", f"{ro(alt['value']):.2f}", ALTMAN_SCORE[zone],
                              f"Altman {alt.get('variant') or 'Z'} = {ro(alt['value']):.2f}: {names[zone]} "
                              "(sigură → notă maximă, gri → jumătate, dificultate → zero)."))
    else:
        out.append(_component("Altman", "N/A", None, "Altman: N/A (date insuficiente în bilanț)."))

    nd, ebitda = inp.get("net_debt"), inp.get("ebitda")
    if _num(nd) and _num(ebitda) and ebitda > 0:
        ratio = nd / ebitda
        txt = (f"Datorie netă / EBITDA = {ro(ratio):.2f}" + (" (numerar net)" if ratio < 0 else "")
               + " (≤ 1 → notă maximă, ≥ 4 → zero).")
        out.append(_component("Datorie netă / EBITDA", f"{ro(ratio):.2f}", linear_score(ratio, *ND_EBITDA_SCALE), txt))
    elif _num(nd) and _num(ebitda):
        if nd <= 0:
            out.append(_component("Datorie netă / EBITDA", "numerar net", 1.0,
                                  "EBITDA negativ, dar compania are numerar net: datoria nu e o problemă."))
        else:
            out.append(_component("Datorie netă / EBITDA", "EBITDA ≤ 0", 0.0,
                                  "EBITDA negativ sau zero, cu datorie netă: datoria nu e acoperită de rezultatul operațional."))
    else:
        out.append(_component("Datorie netă / EBITDA", "N/A", None, "Datorie netă / EBITDA: N/A (lipsește datoria netă sau EBITDA)."))

    cov = inp.get("coverage")
    if _num(cov):
        out.append(_component("Acoperirea dobânzii", f"{ro(cov):.1f}x", linear_score(cov, *COVERAGE_SCALE),
                              f"Acoperirea dobânzii (EBIT / dobânzi) = {ro(cov):.1f}x (≤ 1,5 → zero, ≥ 8 → notă maximă)."))
    else:
        out.append(_component("Acoperirea dobânzii", "N/A", None,
                              "Acoperirea dobânzii: N/A (dobânda nu e raportată sau rândul include alte costuri financiare)."))
    return out


# --- Bănci și asigurători ------------------------------------------------------

def justified_pbv(roe, ke, g=TERMINAL_GROWTH):
    """P/BV justificat = (ROE − g) / (ke − g) (Gordon pe valoarea contabilă). None dacă ke ≤ g sau lipsesc date."""
    if not (_num(roe) and _num(ke) and _num(g)) or ke <= g:
        return None
    return (roe - g) / (ke - g)


def financial_components(inp):
    roe, ke, pbv = inp.get("roe"), inp.get("ke"), inp.get("pbv")
    val, prof = [], []
    if not _num(ke):
        msg = "lipsește costul capitalului propriu (rata fără risc sau beta)"
        val.append(_component("P/BV față de ROE", "N/A", None, f"P/BV față de ROE: N/A ({msg})."))
        prof.append(_component("ROE − ke", "N/A", None, f"ROE față de costul capitalului: N/A ({msg})."))
        return val, prof
    if not _num(roe):
        val.append(_component("P/BV față de ROE", "N/A", None, "P/BV față de ROE: N/A (lipsește ROE)."))
        prof.append(_component("ROE − ke", "N/A", None, "ROE față de costul capitalului: N/A (lipsește ROE)."))
        return val, prof
    just = justified_pbv(roe, ke)
    if not _num(pbv) or pbv <= 0:
        val.append(_component("P/BV față de ROE", "N/A", None, "P/BV față de ROE: N/A (lipsește P/BV)."))
    elif just is None or just <= 0:
        val.append(_component("P/BV față de ROE", f"P/BV {ro(pbv):.2f}", 0.0,
                              f"ROE {ro(roe * 100):.1f}% e sub creșterea pe termen lung ({ro(TERMINAL_GROWTH * 100):.0f}%): "
                              "capitalul nu se remunerează, orice P/BV pozitiv e scump."))
    else:
        ratio = pbv / just
        val.append(_component("P/BV față de ROE", f"{ro(pbv):.2f} / {ro(just):.2f}", linear_score(ratio, *PBV_RATIO_SCALE),
                              f"P/BV {ro(pbv):.2f} față de P/BV justificat {ro(just):.2f} = (ROE {ro(roe * 100):.1f}% − "
                              f"{ro(TERMINAL_GROWTH * 100):.0f}%) / (ke {ro(ke * 100):.1f}% − {ro(TERMINAL_GROWTH * 100):.0f}%): "
                              f"raport {ro(ratio):.2f} (≤ 0,7 → notă maximă, ≥ 1,3 → zero)."))
    spread = roe - ke
    prof.append(_component("ROE − ke", f"{ro(spread * 100):+.1f} pp", linear_score(spread, *ROE_SPREAD_SCALE),
                           f"ROE {ro(roe * 100):.1f}% față de costul capitalului propriu {ro(ke * 100):.1f}%: "
                           f"{ro(spread * 100):+.1f} pp (−5 pp → zero, +5 pp → notă maximă)."))
    return val, prof


# --- Scorul -------------------------------------------------------------------

def _pillar(key, components):
    scores = [c["score"] for c in components if _num(c["score"])]
    return {"key": key, "label": PILLAR_LABELS[key], "score": sum(scores) / len(scores) if scores else None,
            "components": components}


def fundamental_score(inp):
    """Scorul fundamental SUA/UE dintr-un dict de intrări deja calculate (vezi app.py).

    Chei: financial (bool), price, dcf_na, wacc, fcf, fcf_basis, net_debt, shares, pe, eps, aaa (procente),
    aaa_label, pbv, piotroski
    ({passed, evaluable}), altman ({value, variant, zone}), ebitda, coverage, roe, ke.
    Întoarce {score (0–100 sau None), label, color, note, pillars, financial, reverse_dcf}.
    """
    financial = bool(inp.get("financial"))
    rev = None
    if financial:
        val, prof = financial_components(inp)
        pillars = [_pillar("evaluare", val), _pillar("rentabilitate", prof)]
    else:
        val, rev = valuation_components(inp)
        pillars = [_pillar("evaluare", val), _pillar("calitate", quality_components(inp)),
                   _pillar("sanatate", health_components(inp))]
    present = [p["score"] for p in pillars if _num(p["score"])]
    out = {"score": None, "label": "DATE INSUFICIENTE", "color": "#8B949E", "pillars": pillars,
           "financial": financial, "reverse_dcf": rev,
           "note": f"Sub {MIN_PILLARS} piloni cu date: fără scor."}
    if len(present) < MIN_PILLARS:
        return out
    score = round(sum(present) / len(present) * 100, 1)
    out.update(score=score, note=TEST_NOTE)
    for floor, label, color in VERDICT_BANDS:
        if score >= floor:
            out.update(label=label, color=color)
            break
    return out


def explain(result, good=0.67, bad=0.33):
    """(puncte tari, la mijloc, puncte slabe, fără date): textele componentelor după notă.
    Componentele informative (care nu intră în scor) sunt în info_notes."""
    strong, middle, weak, missing = [], [], [], []
    for p in result["pillars"]:
        for c in p["components"]:
            if c.get("info"):
                continue
            if not _num(c["score"]):
                missing.append(c["text"])
            elif c["score"] >= good:
                strong.append(c["text"])
            elif c["score"] <= bad:
                weak.append(c["text"])
            else:
                middle.append(c["text"])
    return strong, middle, weak, missing


def info_notes(result):
    """Textele componentelor informative (afișate, dar fără notă în scor)."""
    return [c["text"] for p in result["pillars"] for c in p["components"] if c.get("info")]


__all__ = ["fundamental_score", "explain", "info_notes", "reverse_dcf_growth", "justified_pbv", "gn_score",
           "earnings_yield", "earnings_yield_vs_aaa", "TEST_NOTE", "UNTESTED_NOTE"]
