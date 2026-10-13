"""Motorul scorului unic. Funcții pure (fără st.*, fără rețea).

Principii (instrucțiunile proiectului și financial-metrics-reference):
- Fiecare pilon dă o notă între 0 și 1, sau None când nu are date. None nu devine 0.
- Pilonii se grupează: fundamental, tehnic, context. În fiecare grup, nota e media ponderată
  doar a pilonilor cu date; scorul final e media ponderată doar a grupurilor cu date
  (ponderile se renormalizează).
- Fundamentalul e baza: fără niciun pilon fundamental, scorul nu primește verdict.
- Pragurile verdictului nu sunt încă justificate printr-un test istoric: eticheta spune asta.
- Factorii macro comuni tuturor acțiunilor nu intră în scorul unei companii; se afișează separat.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

GROUP_WEIGHTS = {"fundamental": 0.60, "tehnic": 0.25, "context": 0.15}
GROUP_LABELS = {"fundamental": "Fundamental", "tehnic": "Tehnic", "context": "Context (sentiment, opțiuni)"}

# Benzile verdictului, pe scorul 0–100. NETESTATE: se înlocuiesc după testul istoric.
VERDICT_BANDS = ((70.0, "FAVORABIL", "#3FB950"), (40.0, "NEUTRU", "#D29922"), (0.0, "NEFAVORABIL", "#F85149"))
UNTESTED_NOTE = "Praguri netestate istoric: verdictul e orientativ până la testul pe date din trecut."


@dataclass(frozen=True)
class Pillar:
    """Un pilon al scorului. `score` între 0 și 1 sau None (fără date)."""
    key: str
    label: str
    group: str
    score: float = None
    weight: float = 1.0
    detail: str = ""


def _valid(x):
    return x is not None and not isinstance(x, bool) and np.isfinite(float(x))


def linear_score(value, worst, best):
    """Nota 0–1 pe o scară liniară, plafonată la capete.

    `worst` primește 0, `best` primește 1. Merge și invers (worst > best), de exemplu
    RSI: worst=70, best=30. Valoare lipsă → None. worst == best → None (scară nedefinită).
    """
    if not _valid(value) or not _valid(worst) or not _valid(best) or worst == best:
        return None
    s = (float(value) - worst) / (best - worst)
    return float(min(1.0, max(0.0, s)))


def _weighted_mean(pairs):
    """Media ponderată a perechilor (notă, pondere) cu notă validă și pondere > 0; None dacă nu există."""
    good = [(s, w) for s, w in pairs if _valid(s) and _valid(w) and w > 0]
    total = sum(w for _, w in good)
    if not good or total <= 0:
        return None
    return sum(s * w for s, w in good) / total


def combine(pillars, group_weights=None):
    """Combină pilonii în scorul unic.

    Întoarce un dict:
      score        scor 0–100 (o zecimală) sau None
      groups       {grup: {"score": 0–100 sau None, "used": [...], "missing": [...]}}
      used         etichetele pilonilor cu date
      missing      etichetele pilonilor fără date
      coverage     (pilonii cu date, total piloni)
      weights_used ponderile efective ale grupurilor, după renormalizare
    """
    group_weights = dict(GROUP_WEIGHTS if group_weights is None else group_weights)
    groups = {}
    for g in group_weights:
        members = [p for p in pillars if p.group == g]
        score = _weighted_mean([(p.score, p.weight) for p in members])
        groups[g] = {
            "score": None if score is None else round(score * 100, 1),
            "used": [p.label for p in members if _valid(p.score)],
            "missing": [p.label for p in members if not _valid(p.score)],
        }
    unknown = sorted({p.group for p in pillars} - set(group_weights))
    if unknown:
        raise ValueError(f"Grupuri necunoscute: {unknown}")

    present = {g: w for g, w in group_weights.items() if groups[g]["score"] is not None and w > 0}
    total = sum(present.values())
    overall = None
    if total > 0:
        overall = sum(groups[g]["score"] / 100 * w for g, w in present.items()) / total
    used = [p.label for p in pillars if _valid(p.score)]
    return {
        "score": None if overall is None else round(overall * 100, 1),
        "groups": groups,
        "used": used,
        "missing": [p.label for p in pillars if not _valid(p.score)],
        "coverage": (len(used), len(pillars)),
        "weights_used": {g: round(w / total, 4) for g, w in present.items()} if total > 0 else {},
    }


def verdict(result, bands=VERDICT_BANDS, tested=False):
    """Eticheta scorului: (text, culoare, notă).

    Fără niciun pilon fundamental: „DATE INSUFICIENTE" (fundamentalul e baza deciziei).
    Cât timp pragurile nu sunt testate istoric, nota o spune explicit.
    """
    if result is None or result.get("score") is None:
        return "DATE INSUFICIENTE", "#8B949E", "Niciun pilon nu are date."
    if result["groups"].get("fundamental", {}).get("score") is None:
        return ("DATE INSUFICIENTE", "#8B949E",
                "Lipsesc toți pilonii fundamentali; scorul ar descrie doar tehnicul și contextul.")
    note = "" if tested else UNTESTED_NOTE
    for floor, label, color in bands:
        if result["score"] >= floor:
            return label, color, note
    return bands[-1][1], bands[-1][2], note


# ---------------------------------------------------------------------------
# Pilonul tehnic. Calculat doar din istoricul de până la ultima zi din serie,
# ca aceeași funcție să servească și aplicației, și testului istoric.
# ---------------------------------------------------------------------------

def trend_score(close):
    """Trendul pe medii mobile, la ultima ședință.

    Componente (fiecare 1 = adevărat, 0 = fals): Close > SMA50, Close > SMA200, SMA50 > SMA200.
    Cu mai puțin de 200 de ședințe valide se folosește doar Close > SMA50; sub 50 → None.
    """
    c = pd.Series(close, dtype="float64").dropna()
    if len(c) < 50:
        return None
    last = c.iloc[-1]
    sma50 = c.iloc[-50:].mean()
    parts = [float(last > sma50)]
    if len(c) >= 200:
        sma200 = c.iloc[-200:].mean()
        parts += [float(last > sma200), float(sma50 > sma200)]
    return float(np.mean(parts))


def momentum_12_1(close):
    """Randamentul de la acum 12 luni până acum 1 lună (exclude ultima lună), pe date calendaristice.

    Prețul la o dată = ultima închidere disponibilă la sau înainte de acea dată.
    None dacă istoricul nu acoperă 12 luni sau prețul de pornire nu e pozitiv.
    """
    c = pd.Series(close, dtype="float64").dropna()
    if c.empty or not isinstance(c.index, pd.DatetimeIndex):
        return None
    end = c.index[-1]
    start_date, mid_date = end - pd.DateOffset(months=12), end - pd.DateOffset(months=1)
    if c.index[0] > start_date:
        return None
    p0 = c.loc[:start_date].iloc[-1]
    p1 = c.loc[:mid_date].iloc[-1]
    if not p0 > 0:
        return None
    return float(p1 / p0 - 1)


# Scările componentelor. NETESTATE: testul istoric le evaluează pe fiecare separat.
MOMENTUM_SCALE = (-0.30, 0.30)   # −30% → 0, +30% → 1
RSI_SCALE = (70.0, 30.0)         # RSI 70 → 0 (supracumpărat), RSI 30 → 1 (supravândut)


def technical_components(close, rsi_last):
    """Notele 0–1 ale componentelor tehnice și valorile brute din care vin.

    `close`: seria de închideri (ajustate) până la data analizei, cu index de date.
    `rsi_last`: RSI Wilder 14 la ultima ședință (sau None).
    """
    mom = momentum_12_1(close)
    rsi = float(rsi_last) if _valid(rsi_last) else None
    return {
        "trend": trend_score(close),
        "momentum": linear_score(mom, *MOMENTUM_SCALE),
        "rsi": linear_score(rsi, *RSI_SCALE),
        "raw": {"momentum_12_1": mom, "rsi": rsi},
    }


TECH_LABELS = {"trend": "Trend (SMA 50/200)", "momentum": "Momentum 12-1 luni", "rsi": "RSI 14 (Wilder)"}

# Componentele care intră în sub-scorul tehnic. RSI NU intră: testul istoric (2012–2025) a arătat
# IC ≈ 0 pe SUA și IC negativ pe BVB (−0,15: „supravândut" a continuat să scadă). RSI rămâne doar descriptiv.
TECH_SCORED = ("trend", "momentum")


def technical_pillars(close, rsi_last):
    """Pilonii tehnici care intră în scor (trend, momentum), cu ponderi egale."""
    comp = technical_components(close, rsi_last)
    mom = comp["raw"]["momentum_12_1"]
    return [
        Pillar("trend", TECH_LABELS["trend"], "tehnic", comp["trend"]),
        Pillar("momentum", TECH_LABELS["momentum"], "tehnic", comp["momentum"],
               detail="N/A" if mom is None else f"{mom*100:+.1f}%"),
    ]


def subscore_from_components(comp):
    """Sub-scorul tehnic 0–100: media trendului și a momentumului (cele cu date), sau None."""
    s = _weighted_mean([(comp[k], 1.0) for k in TECH_SCORED])
    return None if s is None else round(s * 100, 1)


def technical_subscore(close, rsi_last=None):
    """Sub-scorul tehnic 0–100 (trend + momentum) sau None. `rsi_last` e păstrat pentru compatibilitate."""
    return subscore_from_components(technical_components(close, rsi_last))


def rsi_description(rsi):
    """Text descriptiv pentru RSI, fără semnal de intrare (testul istoric nu a confirmat unul)."""
    if not _valid(rsi):
        return "N/A"
    if rsi >= 70:
        zone = "supracumpărat"
    elif rsi <= 30:
        zone = "supravândut"
    else:
        zone = "neutru"
    return f"{float(rsi):.0f} ({zone})"


# Pragurile momentului la BVB, din testul istoric 2012–2025 (26 de companii, exces față de media BVB la 6 luni):
#   ≥ 80: +6,1%, 54% cazuri pozitive (N = 1.259); 40–80: −1,5%, 40% (N = 769); < 40: −7,2%, 32% (N = 829). Stabile în 2012–2018 și 2019–2025.
MOMENT_BANDS_BVB = ((80.0, "FAVORABIL", "#3FB950"), (40.0, "NEUTRU", "#D29922"), (0.0, "NEFAVORABIL", "#F85149"))


def moment_label(score, is_bvb):
    """(eticheta, culoarea, nota) pentru sub-scorul tehnic.

    BVB: verdict pe pragurile testate. Alte piețe: fără verdict (testul pe acțiunile mari din SUA
    nu a găsit putere de predicție), doar valoarea.
    """
    if not _valid(score):
        return "N/A", "#8B949E", "Istoric de preț insuficient (trendul cere 50 de ședințe, momentumul 12 luni)."
    if not is_bvb:
        return ("FĂRĂ VERDICT", "#8B949E",
                "Testul istoric pe acțiunile mari din SUA (2012–2025) nu a găsit putere de predicție; valoare descriptivă.")
    for floor, label, color in MOMENT_BANDS_BVB:
        if score >= floor:
            return label, color, "Prag testat pe BVB 2012–2025."
    return MOMENT_BANDS_BVB[-1][1], MOMENT_BANDS_BVB[-1][2], "Prag testat pe BVB 2012–2025."


def explain_moment(close, rsi_last=None):
    """Explicația momentului în propoziții scurte (descriptiv, fără îndemnuri). Listă de șiruri."""
    c = pd.Series(close, dtype="float64").dropna()
    if len(c) < 50:
        return ["Istoric de preț insuficient: trendul are nevoie de cel puțin 50 de ședințe."]
    out = []
    last, sma50 = c.iloc[-1], c.iloc[-50:].mean()
    if len(c) >= 200:
        sma200 = c.iloc[-200:].mean()
        checks = [(last > sma50, f"prețul ({last:.4g}) e {'peste' if last > sma50 else 'sub'} media pe 50 de zile ({sma50:.4g})"),
                  (last > sma200, f"{'peste' if last > sma200 else 'sub'} media pe 200 de zile ({sma200:.4g})"),
                  (sma50 > sma200, f"media pe 50 e {'peste' if sma50 > sma200 else 'sub'} cea pe 200")]
        n_ok = sum(ok for ok, _ in checks)
        verdict = {3: "trend ascendent complet", 2: "trend majoritar ascendent", 1: "trend majoritar descendent",
                   0: "trend descendent complet"}[n_ok]
        out.append(f"Trend: {n_ok} din 3 condiții ({verdict}) — " + "; ".join(t for _, t in checks) + ".")
    else:
        out.append(f"Trend: prețul e {'peste' if last > sma50 else 'sub'} media pe 50 de zile "
                   f"(sub 200 de ședințe, media pe 200 nu se calculează).")
    mom = momentum_12_1(c)
    if mom is None:
        out.append("Momentum 12-1 luni: N/A (istoric sub 12 luni).")
    else:
        cap = " (peste +30%: notă maximă)" if mom >= MOMENTUM_SCALE[1] else " (sub −30%: notă minimă)" if mom <= MOMENTUM_SCALE[0] else ""
        out.append(f"Momentum: {mom * 100:+.1f}% de acum 12 luni până acum o lună{cap}.")
    if _valid(rsi_last):
        out.append(f"RSI {rsi_description(rsi_last)}: doar informativ, nu intră în scor (nu a dat semnal în testul pe BVB).")
    return out
