"""Calcule macroeconomice. Funcții pure."""
import numpy as np
import pandas as pd

from data.helpers import ro


def yoy_pct(series, max_gap_days=45):
    """Variația procentuală an/an a unei serii lunare (ex. indicele CPI).

    Compară ultima observație cu observația de acum 12 luni. Întoarce None dacă
    seria nu acoperă 12 luni sau dacă observația de referință e mai veche cu
    peste `max_gap_days` față de data-țintă.

    Inflația se calculează din NIVELUL indicelui: CPI_t / CPI_{t-12} - 1.
    Nivelul indicelui (ex. 320) nu este o rată a inflației.
    """
    if series is None:
        return None
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return None
    last_date = s.index[-1]
    target = last_date - pd.DateOffset(years=1)
    base = s[s.index <= target]
    if base.empty:
        return None
    base_date = base.index[-1]
    if (target - base_date).days > max_gap_days:
        return None
    base_val = float(base.iloc[-1])
    if base_val == 0:
        return None
    return (float(s.iloc[-1]) / base_val - 1) * 100


def real_rate(policy_rate_pct, inflation_yoy_pct):
    """Rata reală a dobânzii = dobânda de politică monetară - inflația an/an (ambele în %)."""
    if policy_rate_pct is None or inflation_yoy_pct is None:
        return None
    return float(policy_rate_pct) - float(inflation_yoy_pct)


# --- Curba randamentelor pe piața acțiunii -------------------------------------

EUROZONE_SUFFIXES = (".DE", ".F", ".PA", ".AS", ".MI", ".MC", ".BR", ".LS", ".HE", ".VI", ".IR", ".AT")


def market_of(symbol, currency=None):
    """Piața pentru curba randamentelor: "RO" (.RO), "EU" (bursă din zona euro sau acțiune în EUR),
    "US" (acțiune în USD fără sufix de bursă străină, inclusiv ADR-urile din SUA) sau None (altă piață:
    Londra, Elveția etc., pentru care nu există încă o curbă în aplicație)."""
    sym = str(symbol or "").upper()
    cur = str(currency or "").upper()
    if sym.endswith(".RO"):
        return "RO"
    if sym.endswith(EUROZONE_SUFFIXES) or cur == "EUR":
        return "EU"
    if cur == "USD" and "." not in sym.replace("-", ""):
        return "US"
    return None


def monthly_spread(long_rate, short_rate):
    """Dobânda lungă − dobânda scurtă pe ultima lună prezentă în ambele serii lunare (procente, indexate
    pe perioadă „AAAA-LL” sau pe dată). Nu amestecă luni diferite. {"spread", "long", "short", "month"}
    (month = "AAAA-LL") sau None dacă nu există nicio lună comună."""
    def _monthly(series):
        s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
        if s.empty:
            return s
        s.index = pd.PeriodIndex([pd.Period(str(i)[:7], "M") for i in s.index])
        return s[~s.index.duplicated(keep="last")]

    if long_rate is None or short_rate is None:
        return None
    both = pd.concat([_monthly(long_rate).rename("l"), _monthly(short_rate).rename("s")], axis=1, join="inner").dropna()
    if both.empty:
        return None
    month = both.index.max()
    lv, sv = float(both.loc[month, "l"]), float(both.loc[month, "s"])
    return {"spread": lv - sv, "long": lv, "short": sv, "month": str(month)}


TBILL_13W_DAYS = 91


def tbill_discount_to_bey(discount_pct, days=TBILL_13W_DAYS):
    """Randamentul de discount al unui titlu de stat SUA pe termen scurt (ex. ^IRX, în procente) → randament echivalent
    de obligațiune (bond-equivalent yield), convenția în care sunt ^TNX și FRED DGS3MO:
    BEY = 365 × d ÷ (360 − d × t), cu d zecimal și t = zilele până la scadență (≤ 182; 13 săptămâni = 91).
    Exemplu: d = 4,00% → 365 × 0,04 ÷ (360 − 0,04 × 91) = 4,097%.
    Primește un număr sau o serie; lipsă/neinterpretabil → None (număr) sau NaN (serie)."""
    if isinstance(discount_pct, pd.Series):
        d = pd.to_numeric(discount_pct, errors="coerce") / 100.0
        return 100.0 * 365.0 * d / (360.0 - d * days)
    v = _finite(discount_pct)
    if v is None:
        return None
    d = v / 100.0
    return 100.0 * 365.0 * d / (360.0 - d * days)


def same_day_spread(long_close, short_close):
    """10 ani − 3 luni din închideri zilnice (ex. ^TNX, ^IRX, în procente), pe ultima zi prezentă în ambele.
    {"spread", "long", "short", "date"} sau None."""
    if long_close is None or short_close is None:
        return None
    def _clean(s):
        s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
        idx = pd.DatetimeIndex(s.index)
        s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        return s[~s.index.duplicated(keep="last")]
    both = pd.concat([_clean(long_close).rename("l"), _clean(short_close).rename("s")], axis=1, join="inner").dropna()
    if both.empty:
        return None
    d = both.index.max()
    lv, sv = float(both.loc[d, "l"]), float(both.loc[d, "s"])
    return {"spread": lv - sv, "long": lv, "short": sv, "date": d}


# ---------------------------------------------------------------------------
# Verdictul macro din „Piață Globală”: piloni cu scor 0–100, ponderi fixe, lipsă = exclus.
# Pragurile pilonilor și ale verdictului sunt orientative, netestate (nu există încă un test istoric).
# ---------------------------------------------------------------------------
MACRO_WEIGHTS = {"credit": 30, "vix": 20, "dolar": 15, "sectoare": 15, "curba": 10, "sentiment": 10}
MACRO_NAMES = {"credit": "Piața de credit (HYG/IEF)", "vix": "Volatilitate (VIX)",
               "dolar": "Dolar vs acțiuni (corelație SPY/UUP)", "sectoare": "Rotația sectorială (1 lună)",
               "curba": "Curba randamentelor (10A − 3L)", "sentiment": "Sentimentul știrilor"}
MACRO_MIN_WEIGHT = 50          # sub atâta pondere cu date, fără verdict
MACRO_FAVORABLE, MACRO_ADVERSE = 60.0, 40.0
CREDIT_SMA_DAYS, CREDIT_BAND = 50, 0.01
VIX_CALM, VIX_PANIC = 15.0, 25.0
DOLLAR_CORR_SEVERE = -0.70
SECTOR_CYCLICAL = ("Tehnologie", "Consum Discreționar", "Financiar", "Comunicații")
SECTOR_DEFENSIVE = ("Utilități", "Consum de Bază")
SENTIMENT_BAND = 0.15


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _pillar(score, value, detail):
    return {"score": None if score is None else float(np.clip(score, 0.0, 100.0)), "value": value, "detail": detail}


def pillar_credit(ratio):
    """HYG/IEF față de media lui pe 50 de ședințe: > +1% = 100, < −1% = 0, între = 50."""
    s = pd.to_numeric(pd.Series(ratio, dtype="float64"), errors="coerce").dropna() if ratio is not None else pd.Series(dtype="float64")
    if len(s) < CREDIT_SMA_DAYS:
        return _pillar(None, None, f"sub {CREDIT_SMA_DAYS} de ședințe de istoric HYG/IEF")
    sma = float(s.iloc[-CREDIT_SMA_DAYS:].mean())
    dev = float(s.iloc[-1]) / sma - 1.0
    score = 100.0 if dev > CREDIT_BAND else 0.0 if dev < -CREDIT_BAND else 50.0
    return _pillar(score, dev, f"HYG/IEF la {ro(dev * 100):+.2f}% față de media pe {CREDIT_SMA_DAYS} de ședințe")


def credit_sentence(pillar):
    """Propoziția despre piața de credit din Sinteza AI, din pilonul `pillar_credit` (aceeași regulă și bandă).
    None dacă pilonul n-are date: Sinteza nu afirmă nimic despre credit."""
    if not pillar or pillar.get("score") is None or pillar.get("value") is None:
        return None
    dev = f"HYG/IEF {ro(pillar['value'] * 100):+.2f}% față de media pe {CREDIT_SMA_DAYS} de ședințe"
    if pillar["score"] >= 100:
        return f"Piața de credit: apetit pentru risc ({dev}, peste banda de ±{CREDIT_BAND * 100:.0f}%)."
    if pillar["score"] <= 0:
        return f"Piața de credit: retragere din risc ({dev}, sub banda de ±{CREDIT_BAND * 100:.0f}%)."
    return f"Piața de credit: neutră ({dev}, în banda de ±{CREDIT_BAND * 100:.0f}%)."


def credit_band_frame(ratio):
    """Graficul pilonului „Piața de credit”: seria HYG/IEF, media ei pe CREDIT_SMA_DAYS ședințe (aceeași ca în
    `pillar_credit`) și banda ±CREDIT_BAND în jurul mediei. Coloane ratio, sma, upper, lower; media e NaN până la
    prima fereastră completă (nu se completează). DataFrame gol pe serie goală."""
    s = pd.to_numeric(pd.Series(ratio, dtype="float64"), errors="coerce").dropna() if ratio is not None else pd.Series(dtype="float64")
    if s.empty:
        return pd.DataFrame(columns=["ratio", "sma", "upper", "lower"], dtype="float64")
    sma = s.rolling(CREDIT_SMA_DAYS, min_periods=CREDIT_SMA_DAYS).mean()
    return pd.DataFrame({"ratio": s, "sma": sma, "upper": sma * (1 + CREDIT_BAND), "lower": sma * (1 - CREDIT_BAND)})


def pillar_vix(vix):
    """VIX ≤ 15 = 100, ≥ 25 = 0, liniar între (20 = 50)."""
    v = _finite(vix)
    if v is None or v <= 0:
        return _pillar(None, None, "VIX indisponibil")
    score = (VIX_PANIC - v) / (VIX_PANIC - VIX_CALM) * 100.0
    return _pillar(score, v, f"VIX {ro(v):.1f}")


def pillar_dollar(corr):
    """Corelația SPY/UUP pe 3 luni: ≤ −0,70 = 0, ≥ 0 = 100, liniar între."""
    c = _finite(corr)
    if c is None:
        return _pillar(None, None, "corelația SPY/UUP indisponibilă")
    score = (c - DOLLAR_CORR_SEVERE) / (0.0 - DOLLAR_CORR_SEVERE) * 100.0
    return _pillar(score, c, f"corelație {ro(c):+.2f}")


def pillar_sectors(month_returns):
    """Primele 3 sectoare SUA după randamentul pe o lună: ≥ 2 ciclice = 100; altfel, un defensiv
    printre ele = 0; altfel 50. `month_returns`: dict sector -> randament (fracție)."""
    clean = {k: _finite(v) for k, v in (month_returns or {}).items()}
    clean = {k: v for k, v in clean.items() if v is not None}
    if len(clean) < 6:
        return _pillar(None, None, "sub 6 sectoare cu randament pe o lună")
    top = [k for k, _ in sorted(clean.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    n_cyc = sum(1 for k in top if k in SECTOR_CYCLICAL)
    if n_cyc >= 2:
        score = 100.0
    elif any(k in SECTOR_DEFENSIVE for k in top):
        score = 0.0
    else:
        score = 50.0
    return _pillar(score, top, "primele 3 pe o lună: " + ", ".join(top))


def pillar_curve(spread):
    """10A − 3L, aceeași zi: < 0 (inversată) = 0; 0–0,5 pp = 50; ≥ 0,5 pp = 100."""
    s = _finite(spread)
    if s is None:
        return _pillar(None, None, "spread 10A − 3L indisponibil")
    score = 0.0 if s < 0 else 50.0 if s < 0.5 else 100.0
    return _pillar(score, s, f"spread {ro(s):+.2f} pp")


def pillar_sentiment(sentiment):
    """Media FinBERT pe titluri (−1…+1): > 0,15 = 100, < −0,15 = 0, între = 50."""
    s = _finite(sentiment)
    if s is None:
        return _pillar(None, None, "fără știri analizate")
    score = 100.0 if s > SENTIMENT_BAND else 0.0 if s < -SENTIMENT_BAND else 50.0
    return _pillar(score, s, f"sentiment mediu {ro(s):+.2f}")


def macro_verdict(pillars):
    """Scorul agregat = media ponderată a pilonilor cu date (`MACRO_WEIGHTS`), renormalizată pe ponderea
    disponibilă. Fără verdict dacă ponderea cu date e sub `MACRO_MIN_WEIGHT`.

    `pillars`: dict cheie -> rezultatul `pillar_*`. Întoarce dict: score (None = N/A), label
    ('FAVORABIL'/'MIXT'/'DEFAVORABIL'/None), weight_used, missing (chei fără date)."""
    used = {k: p for k, p in pillars.items() if k in MACRO_WEIGHTS and p and p.get("score") is not None}
    missing = [k for k in MACRO_WEIGHTS if k not in used]
    w = sum(MACRO_WEIGHTS[k] for k in used)
    if w < MACRO_MIN_WEIGHT:
        return {"score": None, "label": None, "weight_used": w, "missing": missing}
    score = sum(MACRO_WEIGHTS[k] * used[k]["score"] for k in used) / w
    label = "FAVORABIL" if score >= MACRO_FAVORABLE else "DEFAVORABIL" if score < MACRO_ADVERSE else "MIXT"
    return {"score": score, "label": label, "weight_used": w, "missing": missing}


MACRO_SHORT = {"credit": "Credit", "vix": "VIX", "dolar": "Dolar", "sectoare": "Sectoare",
               "curba": "Curba", "sentiment": "Sentiment"}


def macro_explanation(pillars, verdict):
    """Explicația scorului macro, afișată în pagină (text simplu, fără HTML).

    Pentru fiecare pilon cu date: abaterea față de neutru = (scor − 50) × pondere ÷ ponderea cu date.
    Suma abaterilor + 50 = scorul. Întoarce dict: lines (listă de propoziții), pulls (listă
    (cheie, scor, abatere) sortată crescător după abatere), missing (chei fără date)."""
    w = verdict.get("weight_used") or 0
    missing = list(verdict.get("missing") or [])
    score = verdict.get("score")
    lines = []
    if score is None:
        lines.append(f"Fără scor: doar {w}% din ponderi au date (minimum {MACRO_MIN_WEIGHT}%).")
        if missing:
            lines.append("Fără date: " + ", ".join(MACRO_SHORT[k] for k in missing) + ".")
        return {"lines": lines, "pulls": [], "missing": missing}
    pulls = sorted(((k, pillars[k]["score"], (pillars[k]["score"] - 50.0) * MACRO_WEIGHTS[k] / w)
                    for k in MACRO_WEIGHTS if k not in missing), key=lambda t: t[2])
    lines.append(f"Scorul {ro(score):.0f} din 100 = media ponderată a pilonilor cu date ({w}% din ponderi): "
                 "0 = mediu nefavorabil, 50 = neutru, 100 = favorabil. "
                 f"FAVORABIL de la {MACRO_FAVORABLE:.0f}, DEFAVORABIL sub {MACRO_ADVERSE:.0f}, între ele MIXT.")
    parts = [f"{MACRO_SHORT[k]} {ro(sc):.0f} → {ro(d):+.1f}" for k, sc, d in pulls]
    lines.append(f"Pornind de la neutru (50), fiecare pilon adaugă (scor − 50) × pondere ÷ {w}: "
                 + "; ".join(parts) + f". Total {ro(score - 50):+.1f} → {ro(score):.0f}.")
    neg = [t for t in pulls if t[2] < 0]
    pos = [t for t in pulls if t[2] > 0]
    tail = []
    if neg:
        tail.append(f"a tras cel mai mult în jos {MACRO_NAMES[neg[0][0]]} ({ro(neg[0][2]):+.1f})")
    if pos:
        tail.append(f"a ajutat cel mai mult {MACRO_NAMES[pos[-1][0]]} ({ro(pos[-1][2]):+.1f})")
    if tail:
        lines.append("Azi " + ", ".join(tail) + ".")
    if missing:
        lines.append("Fără date acum: " + ", ".join(MACRO_SHORT[k] for k in missing)
                     + " — ponderea lor se reîmparte la ceilalți piloni (nu contează ca neutri).")
    return {"lines": lines, "pulls": pulls, "missing": missing}


def period_return(close, months=1):
    """Randamentul pe ultimele `months` luni calendaristice: ultima închidere ÷ ultima închidere
    de la sau înainte de (ultima dată − months) − 1. None dacă istoricul nu acoperă perioada."""
    s = pd.to_numeric(pd.Series(close, dtype="float64"), errors="coerce").dropna() if close is not None else pd.Series(dtype="float64")
    if len(s) < 2:
        return None
    s = s.sort_index()
    cutoff = s.index[-1] - pd.DateOffset(months=months)
    before = s[s.index <= cutoff]
    if before.empty or before.iloc[-1] <= 0:
        return None
    return float(s.iloc[-1] / before.iloc[-1] - 1.0)


# ---------------------------------------------------------------------------
# „Indicatori Macroeconomici” și „Ghidul macro” din Piață Globală: variații pe ultimele două ședințe valide
# ale fiecărei serii (piețele au calendare diferite), randamentele în puncte procentuale, lipsă = None.
# Pragurile sunt orientative, netestate.
# ---------------------------------------------------------------------------
YIELD_MOVE_PP = 0.25          # variația randamentului pe 10 ani într-o lună care contează (25 de puncte de bază)


def last_change(close):
    """Ultima valoare și variația față de ședința precedentă cu valoare: dict last, prev, abs, pct (%), date.
    None pe serie cu mai puțin de două valori."""
    s = pd.to_numeric(pd.Series(close, dtype="float64"), errors="coerce").dropna() if close is not None else pd.Series(dtype="float64")
    if len(s) < 2:
        return None
    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
    return {"last": last, "prev": prev, "abs": last - prev, "pct": (last / prev - 1.0) * 100.0 if prev != 0 else None,
            "date": s.index[-1]}


def change_over(close, months):
    """Variația absolută față de ultima valoare de la sau înainte de aceeași zi de acum `months` luni; None dacă seria nu acoperă."""
    s = pd.to_numeric(pd.Series(close, dtype="float64"), errors="coerce").dropna() if close is not None else pd.Series(dtype="float64")
    if len(s) < 2:
        return None
    cutoff = s.index[-1] - pd.DateOffset(months=months)
    before = s[s.index <= cutoff]
    if before.empty:
        return None
    return float(s.iloc[-1] - before.iloc[-1])


def yield_trend(month_change_pp):
    """'creste' / 'scade' / 'stabil' după variația randamentului pe 10 ani într-o lună (± 0,25 pp), sau None."""
    v = _finite(month_change_pp)
    if v is None:
        return None
    return "creste" if v > YIELD_MOVE_PP else "scade" if v < -YIELD_MOVE_PP else "stabil"


def daily_cross_asset_signals(pct, tnx_pp=None, vix_pct=None):
    """Semnalele zilei din „Ghidul macro”, pe variațiile de azi (procente; randamentul 10 ani în puncte procentuale).

    `pct`: dict cheie -> variația % a zilei pentru 'usd' (DX-Y.NYB), 'gold', 'oil', 'copper'; lipsă = None (regula se sare).
    Întoarce lista de chei de semnal (texte în interfață). Pragurile (0,3%, 0,5%, 1%, 10 pb, VIX +10%) sunt cele vechi,
    exprimate corect, netestate."""
    g = lambda k: _finite((pct or {}).get(k))
    usd, gold, oil, copper = g("usd"), g("gold"), g("oil"), g("copper")
    tnx, vix = _finite(tnx_pp), _finite(vix_pct)
    out = []
    if usd is not None and gold is not None and usd < -0.3 and gold > 0.3:
        out.append("aur_refugiu")
    if usd is not None and oil is not None and usd > 0.3 and oil < -0.5:
        out.append("petrol_dolar")
    if copper is not None and copper > 1.0:
        out.append("cupru_expansiune")
    if tnx is not None and tnx > 0.10:
        out.append("dobanzi_cresc")
    elif tnx is not None and tnx < -0.10:
        out.append("dobanzi_scad")
    if gold is not None and usd is not None and vix is not None and gold > 1.0 and usd > 0.5 and vix > 10.0:
        out.append("fuga_siguranta")
    return out


# Barometrul FED din Piață Globală
def interpret_fred(df_macro):
    """Interpretarea Barometrului FED: reguli pe variațiile lunare din `get_fred_macro_data`.
    Un indicator lipsă nu mai e 0: regulile care au nevoie de el se sar (înainte, șomajul lipsă = 0 declanșa „Goldilocks”).
    Pragurile sunt orientative, netestate."""
    if df_macro.empty: return []
    
    bullets = []
    m = {row['_simbol']: row for _, row in df_macro.iterrows()}

    def _f(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    def gv(sym): return _f(m[sym].get('Valoare Curentă')) if sym in m else None
    def gc(sym): return _f(m[sym].get('_change_raw')) if sym in m else None
    def has(*vals): return all(v is not None for v in vals)

    # 1. INFLAȚIA (variația lunară a indicilor de bază) și dobânda reală (Fed − CPI an/an)
    pce, cpi_core, fed_rate = gc('PCEPILFE'), gc('CPILFESL'), gv('FEDFUNDS')
    cpi_yoy = _f(m['CPIAUCSL'].get('_yoy')) if 'CPIAUCSL' in m else None
    real_rate_pct = real_rate(fed_rate, cpi_yoy) if has(fed_rate, cpi_yoy) else None

    if (pce is not None and pce > 0.2) or (cpi_core is not None and cpi_core > 0.3):
        bullets.append("🔥 **INFLAȚIE REZISTENTĂ:** Core PCE (peste +0,2% pe lună) sau Core CPI (peste +0,3%) indică prețuri „lipicioase”. "
                       "Fed-ul nu poate tăia dobânzile fără riscul unui al doilea val inflaționar.")
    elif pce is not None and pce <= 0.15 and real_rate_pct is not None and real_rate_pct > 2:
        bullets.append(f"❄️ **DEZINFLAȚIE CU DOBÂNZI REALE MARI:** inflația de bază încetinește, iar dobânda reală e {ro(real_rate_pct):.1f}%. "
                       "Context în care Fed-ul are spațiu să taie dobânzile.")

    # 2. PIAȚA MUNCII
    nfp, adp, unrate, unrate_chg, jolts_chg = gc('PAYEMS'), gc('ADPCHGS'), gv('UNRATE'), gc('UNRATE'), gc('JTSJOL')
    if has(nfp, adp) and nfp > 200 and adp < 100:
        bullets.append("🔄 **DIVERGENȚĂ PRIVAT-STAT:** Angajările oficiale (NFP) sunt susținute de sectorul public, în timp ce sectorul privat (ADP) frânează. Semnal de ciclu târziu.")
    if has(unrate, unrate_chg) and unrate_chg > 0.1 and unrate > 4.0:
        bullets.append("🚨 **ȘOMAJ ÎN CREȘTERE:** Rata șomajului a urcat peste 4% și crește de la o lună la alta. "
                       "Regula Sahm (media pe 3 luni cu 0,5 pp peste minimul din ultimele 12 luni) a semnalat istoric începutul recesiunilor; "
                       "aici nu e calculată, e doar o atenționare.")
    if jolts_chg is not None and jolts_chg < -400:
        bullets.append("📉 **SCĂDERE A LOCURILOR DE MUNCĂ VACANTE:** Peste 400 de mii de posturi vacante mai puțin (JOLTS) într-o lună: firmele își taie planurile de angajare. "
                       "Presiunea pe salarii scade (ajută inflația), dar consumul poate încetini.")

    # 3. CONSUMUL ȘI SENTIMENTUL
    retail, sentiment = gc('RSAFS'), gc('UMCSENT')
    if has(retail, sentiment) and retail < 0 and sentiment < 0:
        bullets.append("⚠️ **STAGNARE CONSUM:** Vânzările de retail și încrederea consumatorilor scad în aceeași lună. Consumul e ~70% din PIB-ul SUA.")
    elif has(retail, sentiment) and retail > 0.5 and sentiment < -2.0:
        bullets.append("💸 **CHELTUIELI DE NECESITATE:** Vânzările cresc, dar încrederea scade: o parte din creștere poate fi doar efectul prețurilor.")

    # 4. INDICATORI AVANSAȚI
    housing, ind_prod = gc('HOUST'), gc('INDPRO')
    if has(housing, ind_prod) and housing < -3.0 and ind_prod < -0.3:
        bullets.append("🏗️ **FRÂNĂ INDUSTRIALĂ ȘI IMOBILIARĂ:** Construcțiile noi și producția industrială scad în aceeași lună; sunt sectoarele cele mai sensibile la dobânzi.")
    elif housing is not None and housing > 2.0:
        bullets.append("🏠 **REZILIENȚĂ IMOBILIARĂ:** Construcțiile noi cresc cu peste 2% într-o lună, în ciuda dobânzilor.")

    # 5. SINTEZA (doar cu toate datele necesare)
    if has(unrate, pce, retail) and unrate < 4.0 and pce < 0.2 and retail > 0:
        bullets.append("🌟 **REGIM GOLDILOCKS:** Șomaj sub 4%, inflație de bază sub +0,2% pe lună, consum în creștere.")
    elif has(unrate, pce) and unrate > 4.2 and pce > 0.25:
        bullets.append("🌪️ **RISC DE STAGFLAȚIE:** Șomaj peste 4,2% și inflație de bază peste +0,25% pe lună. Mediul în care și acțiunile, și obligațiunile suferă.")

    missing = [n for n, v in (("Core PCE", pce), ("Core CPI", cpi_core), ("șomaj", unrate), ("NFP", nfp), ("ADP", adp),
                               ("JOLTS", jolts_chg), ("retail", retail), ("sentiment", sentiment), ("construcții", housing),
                               ("producție industrială", ind_prod)) if v is None]
    if bullets or missing:
        bullets.append("ℹ️ Reguli orientative, netestate, pe ultima lună publicată."
                       + (f" Fără date: {', '.join(missing)} (regulile lor nu s-au evaluat)." if missing else ""))
    return bullets
