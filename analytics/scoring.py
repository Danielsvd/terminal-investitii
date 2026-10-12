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


def technical_pillars(close, rsi_last):
    """Pilonii tehnici, gata de `combine`, cu ponderi egale între ei."""
    comp = technical_components(close, rsi_last)
    mom, rsi = comp["raw"]["momentum_12_1"], comp["raw"]["rsi"]
    return [
        Pillar("trend", TECH_LABELS["trend"], "tehnic", comp["trend"]),
        Pillar("momentum", TECH_LABELS["momentum"], "tehnic", comp["momentum"],
               detail="N/A" if mom is None else f"{mom*100:+.1f}%"),
        Pillar("rsi", TECH_LABELS["rsi"], "tehnic", comp["rsi"],
               detail="N/A" if rsi is None else f"{rsi:.0f}"),
    ]


def subscore_from_components(comp):
    """Sub-scorul tehnic 0–100 din componentele lui `technical_components`: media celor cu date, sau None."""
    s = _weighted_mean([(comp[k], 1.0) for k in ("trend", "momentum", "rsi")])
    return None if s is None else round(s * 100, 1)


def technical_subscore(close, rsi_last):
    """Sub-scorul tehnic 0–100 (media componentelor cu date) sau None."""
    return subscore_from_components(technical_components(close, rsi_last))
