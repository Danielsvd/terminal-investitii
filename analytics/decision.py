"""Sinteza „Decizie Master AI” în trei pași. Funcții pure (fără st.*, fără rețea).

Pașii se afișează separat și NU se adună:
1. Fundamental: scorul companiei, preluat neschimbat din „Decizie BVB în doi pași” (BVB, testat)
   sau din scorul fundamental SUA/UE (descriptiv). Aici nu se recalculează nimic.
2. Moment: sub-scorul tehnic (trend SMA 50/200 + momentum 12-1). Pragurile 80/40 sunt testate
   doar pe BVB; în afara BVB benzile servesc doar la descriere.
3. Context: alerte și informații, fără puncte. Nu au un test istoric posibil cu datele disponibile
   (opțiuni și știri fără istoric, curba e la nivel de piață), deci nu modifică niciun scor.
"""
import numpy as np

from data.helpers import ro, score_ro

MOMENT_HIGH, MOMENT_LOW = 80.0, 40.0
FUND_LABELS = ("FAVORABIL", "NEUTRU", "NEFAVORABIL")
LEVEL_ORDER = {"alerta": 0, "atentie": 1, "pozitiv": 2, "info": 3}
NEWS_CAVEAT = "potrivirea știrilor cu emitentul e nesigură; netestat"
PC_BULLISH, PC_BEARISH = 0.7, 1.1          # P/C pe open interest (praguri netestate)
IV_RANK_HIGH, IV_RANK_LOW = 80.0, 20.0     # IV Rank (praguri netestate)
SENT_POS, SENT_NEG = 0.15, -0.15           # scorul de sentiment al știrilor, între −1 și 1
SENT_EXTREME = 0.25                        # |scor| ≥ 0,25: euforie / panică (aceeași scară, netestat)
Q_LOW, Q_HIGH = 0.7, 1.0                   # flux de numerar operațional / profit net


def _num(x):
    """float sau None (None, NaN, bool și text nu sunt numere)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def moment_band(score):
    """'high' (≥ 80), 'mid' (40–80), 'low' (< 40) sau None când lipsește."""
    s = _num(score)
    if s is None:
        return None
    return "high" if s >= MOMENT_HIGH else "mid" if s >= MOMENT_LOW else "low"


_FUND_TEXT = {
    "FAVORABIL": "compania arată bine",
    "NEUTRU": "compania e la mijloc",
    "NEFAVORABIL": "compania arată slab",
}
_MOMENT_TEXT = {
    "high": "trendul și momentumul sunt pozitive",
    "mid": "tehnicul nu dă un semnal clar",
    "low": "trendul e negativ",
}


def _connector(fund_label, band):
    """„, iar” când pașii merg în aceeași direcție, „, dar” când se contrazic, „; ” altfel."""
    if (fund_label, band) in (("FAVORABIL", "high"), ("NEFAVORABIL", "low")):
        return ", iar "
    if (fund_label, band) in (("FAVORABIL", "low"), ("NEFAVORABIL", "high"), ("FAVORABIL", "mid")):
        return ", dar "
    return "; "


def _consequence(fund_label, band, is_bvb):
    """Ce înseamnă combinația pentru momentul intrării (descriere, nu îndemn)."""
    if fund_label == "FAVORABIL" and band == "high":
        return "Ambii pași sunt aliniați."
    if fund_label == "FAVORABIL" and band == "mid":
        return "Compania e candidat; momentul intrării nu e confirmat de tehnic."
    if fund_label == "FAVORABIL" and band == "low":
        return "Compania e candidat, dar prețul e încă în scădere."
    if fund_label == "NEFAVORABIL" and band == "high":
        return ("Pe BVB, istoric, momentul bun nu a compensat un fundamental slab." if is_bvb
                else "Creșterea prețului nu e susținută de fundamental.")
    if fund_label == "NEFAVORABIL":
        return "Niciun pas nu susține compania acum."
    if fund_label == "NEUTRU" and band == "high":
        return "Tehnicul e mai bun decât fundamentalul."
    if fund_label == "NEUTRU" and band == "low":
        return "Tehnicul e mai slab decât fundamentalul."
    return ""


def characterize(*, is_bvb, fund_label, fund_score=None, moment_score=None, pillars=(),
                 n_alerts=0, history_text=None):
    """Fraza care caracterizează analiza companiei, din pașii 1–3.

    `pillars`: perechi (nume, notă 0–100 sau None), doar pentru SUA/UE (pilonii scorului fundamental).
    `history_text`: la BVB, rezultatul istoric al combinației (din `final_verdict`).
    Întoarce textul; nu conține îndemnuri de cumpărare/vânzare.
    """
    ref = " față de celelalte companii BVB din fișierele tale" if is_bvb else " pe pilonii cu date"
    band = moment_band(moment_score)
    parts = []
    if fund_label in FUND_LABELS:
        # Același număr de zecimale ca în secțiunea de scor: BVB întreg, SUA/UE o zecimală.
        score_txt = "" if _num(fund_score) is None else (f" ({ro(_num(fund_score)):.0f}/100)" if is_bvb
                                                          else f" ({score_ro(_num(fund_score))}/100)")
        first = f"Fundamental, {_FUND_TEXT[fund_label]}{ref}{score_txt}"
    else:
        first = "Fundamentalul nu se poate evalua (date insuficiente)"
    if band is None:
        parts.append(first + "; momentul nu se poate calcula (istoric de preț insuficient).")
    else:
        conn = _connector(fund_label, band) if fund_label in FUND_LABELS else "; "
        parts.append(f"{first}{conn}{_MOMENT_TEXT[band]} ({score_ro(_num(moment_score))}/100).")
        if fund_label in FUND_LABELS:
            cons = _consequence(fund_label, band, is_bvb)
            if cons:
                parts.append(cons)

    scored = [(n, _num(s)) for n, s in pillars if _num(s) is not None]
    if len(scored) >= 2:
        best = max(scored, key=lambda p: p[1])
        worst = min(scored, key=lambda p: p[1])
        if best[1] - worst[1] >= 20:
            parts.append(f"Punct tare: {best[0]} {ro(best[1]):.0f}/100; punct slab: {worst[0]} {ro(worst[1]):.0f}/100.")

    if n_alerts == 1:
        parts.append("O alertă de context de verificat (lista de mai jos).")
    elif n_alerts > 1:
        parts.append(f"{n_alerts} alerte de context de verificat (lista de mai jos).")

    if is_bvb:
        if history_text:
            parts.append(history_text)
    else:
        parts.append("Descriptiv: testul istoric pe SUA nu a găsit putere de predicție (netestat).")
    return " ".join(parts)


def _flag(level, title, text):
    return {"level": level, "title": title, "text": text}


def context_flags(*, is_bvb, is_financial=False, altman_zone=None, spread=None, spread_label=None,
                  regime_msg=None, opt_pc=None, opt_iv=None, iv_rank=None, iv_days=0, min_iv_days=20,
                  sentiment=None, n_news=0, inst_pct=None, q_ratio=None, rsi=None, rvol=None):
    """Lista alertelor și informațiilor de context. Fără puncte: nu modifică niciun scor.

    Fiecare element: {"level": alerta | atentie | pozitiv | info, "title", "text"}.
    Ordinea: alerte, atenționări, pozitive, informații. Valorile lipsă nu dau semnal.
    """
    flags = []

    # Altman: nu se aplică băncilor și asigurătorilor
    if not is_financial:
        if altman_zone == "distress":
            flags.append(_flag("alerta", "Altman Z", "zona de dificultate financiară (risc statistic de faliment sau restructurare)."))
        elif altman_zone == "grey":
            flags.append(_flag("atentie", "Altman Z", "zona gri (nici sigură, nici de dificultate)."))
        elif altman_zone == "safe":
            flags.append(_flag("info", "Altman Z", "zona sigură."))

    # Curba 10 ani − 3 luni a pieței acțiunii (comună tuturor acțiunilor de pe acea piață)
    sp = _num(spread)
    lab = spread_label or "curba pieței acțiunii"
    if sp is None:
        flags.append(_flag("info", "Curba randamentelor", f"indisponibilă ({lab})."))
    elif sp < 0:
        flags.append(_flag("alerta", "Curba randamentelor",
                           f"10 ani − 3 luni inversată: {ro(sp):+.2f} pp ({lab}). Semnal de piață, nu al companiei."))
    else:
        flags.append(_flag("info", "Curba randamentelor", f"10 ani − 3 luni normală: {ro(sp):+.2f} pp ({lab})."))

    # Regimul de piață K-Means (pe istoricul acțiunii)
    if regime_msg:
        clean = str(regime_msg).replace("**", "").strip()
        # Model netestat: cel mult atenționare galbenă; roșul rămâne pentru Altman și curba inversată.
        level = "atentie" if "PANICĂ" in clean.upper() else "info"
        flags.append(_flag(level, "Regim K-Means", clean + " (model, netestat)."))

    # Opțiuni: doar cu valori calculabile
    pc, iv = _num(opt_pc), _num(opt_iv)
    if pc is not None:
        if pc < PC_BULLISH:
            flags.append(_flag("pozitiv", "Opțiuni P/C", f"{ro(pc):.2f} pe open interest: mai multe call-uri decât put-uri (netestat)."))
        elif pc > PC_BEARISH:
            flags.append(_flag("atentie", "Opțiuni P/C", f"{ro(pc):.2f} pe open interest: mai multe put-uri, cerere de protecție (netestat)."))
        else:
            flags.append(_flag("info", "Opțiuni P/C", f"{ro(pc):.2f} pe open interest: echilibrat (netestat)."))
    if iv is not None:
        rank = _num(iv_rank)
        if rank is not None and iv_days >= min_iv_days:
            if rank > IV_RANK_HIGH:
                flags.append(_flag("atentie", "IV Rank", f"{ro(rank):.0f}%: opțiuni scumpe față de propriul istoric pe un an; IV la bani {ro(iv):.1f}% (netestat)."))
            elif rank < IV_RANK_LOW:
                flags.append(_flag("info", "IV Rank", f"{ro(rank):.0f}%: opțiuni ieftine față de propriul istoric; IV la bani {ro(iv):.1f}% (netestat)."))
            else:
                flags.append(_flag("info", "IV Rank", f"{ro(rank):.0f}%; IV la bani {ro(iv):.1f}% (netestat)."))
        else:
            flags.append(_flag("info", "IV la bani", f"{ro(iv):.1f}%; IV Rank N/A ({int(iv_days)} din {min_iv_days} zile de istoric)."))

    # Sentimentul știrilor (FinBERT), cu avertismentul despre potrivire
    s = _num(sentiment)
    if not n_news or s is None:
        flags.append(_flag("info", "Știri", "fără știri citite; sentiment N/A."))
    elif s > SENT_POS:
        flags.append(_flag("pozitiv", "Știri", f"sentiment optimist ({ro(s):+.2f}, {n_news} titluri; {NEWS_CAVEAT})."))
    elif s < SENT_NEG:
        flags.append(_flag("atentie", "Știri", f"sentiment negativ ({ro(s):+.2f}, {n_news} titluri; {NEWS_CAVEAT})."))
    else:
        flags.append(_flag("info", "Știri", f"sentiment neutru ({ro(s):+.2f}, {n_news} titluri; {NEWS_CAVEAT})."))

    # Instituționali: fără sens la BVB (acționari majoritari stat / grup)
    ip = _num(inst_pct)
    if not is_bvb and ip is not None:
        flags.append(_flag("info", "Instituționali", f"{ro(ip):.0f}% din acțiuni la fonduri."))

    # Calitatea profitului: flux de numerar operațional / profit net
    q = _num(q_ratio)
    if q is not None and not is_financial:
        if q < Q_LOW:
            flags.append(_flag("atentie", "Calitatea profitului", f"CFO / profit net = {ro(q):.2f}: profitul e acoperit slab de numerar."))
        elif q > Q_HIGH:
            flags.append(_flag("pozitiv", "Calitatea profitului", f"CFO / profit net = {ro(q):.2f}: numerarul depășește profitul raportat."))
        else:
            flags.append(_flag("info", "Calitatea profitului", f"CFO / profit net = {ro(q):.2f}."))

    # Descriptive
    r = _num(rsi)
    if r is not None and r > 70:
        flags.append(_flag("info", "RSI 14", f"{ro(r):.0f}: zonă de supracumpărare (descriptiv)."))
    elif r is not None and r < 30:
        flags.append(_flag("info", "RSI 14", f"{ro(r):.0f}: zonă de supravânzare (descriptiv)."))
    v = _num(rvol)
    if v is not None and v > 1.5:
        flags.append(_flag("info", "Volum relativ", f"RVOL {ro(v):.2f}x față de media pe 20 de zile."))

    return sorted(flags, key=lambda f: LEVEL_ORDER[f["level"]])


def count_alerts(flags):
    """Numărul alertelor roșii și al atenționărilor (fără pozitive și informații)."""
    return sum(1 for f in flags if f["level"] in ("alerta", "atentie"))


# --- Verdictul final (aprobat de Daniel 13.11): tabel de reguli, nu sumă de puncte -------------------

VERDICT_NOTE_US = ("Descriptiv: pe SUA (2012–2026) nici scorul fundamental, nici momentul n-au arătat putere de "
                   "predicție în testul istoric; UE netestat.")


def verdict_note(is_bvb, history_text=None):
    """Baza verdictului: la BVB rezultatul testului istoric pe combinația exactă fundamental × moment
    (`bvb_fundamental.TWO_STEP`, 2017–2026); în rest, nota „descriptiv”."""
    if is_bvb:
        base = "Descriptiv, cu bază testată pe BVB (31 de emitenți, 2017–2026). "
        return base + (history_text or "Combinația nu are rezultat istoric (date insuficiente).")
    return VERDICT_NOTE_US


def final_verdict(fund_label, moment_score):
    """(eticheta, nivel) din tabelul aprobat:
    FAVORABIL + moment ≥ 80 → CANDIDAT DE CUMPĂRARE; FAVORABIL + moment < 80 sau necunoscut → DE URMĂRIT;
    NEUTRU → NEUTRU; NEFAVORABIL → EVITĂ; fără fundamental → FĂRĂ VERDICT.
    Nivel: 'up' / 'warn' / 'muted' / 'down' (culoarea din temă)."""
    if fund_label == "FAVORABIL":
        if moment_band(moment_score) == "high":
            return "CANDIDAT DE CUMPĂRARE", "up"
        return "DE URMĂRIT, AȘTEAPTĂ MOMENTUL", "warn"
    if fund_label == "NEUTRU":
        return "NEUTRU", "muted"
    if fund_label == "NEFAVORABIL":
        return "EVITĂ", "down"
    return "FĂRĂ VERDICT", "muted"


# Criteriile minime ale lui Daniel („Hartă în investiții”): (cheie, nume, prag afișat, test)
MY_CRITERIA = (
    ("margin", "Marjă de profit", "≥ 10%", lambda v: v >= 0.10),
    ("rev_growth", "Cifra de afaceri", "în creștere", lambda v: v > 0),
    ("ni_growth", "Profit net", "în creștere", lambda v: v > 0),
    ("debt", "Grad de îndatorare (datorii totale / active totale)", "≤ 70% (≈ levier datorii / capital ≤ 2,33)",
     lambda v: v <= 0.70),
    ("pe", "P/E", "≤ 15", lambda v: 0 < v <= 15),
    ("pbv", "P/BV", "≤ 1,5", lambda v: 0 < v <= 1.5),
    ("gn", "GN (P/E × P/BV)", "≤ 22,5", lambda v: 0 < v <= 22.5),
    ("roa", "ROA", "≥ 5%", lambda v: v >= 0.05),
    ("roe", "ROE", "≥ 15%", lambda v: v >= 0.15),
    ("cap_ron", "Capitalizare", "≥ 1 mld. lei", lambda v: v >= 1e9),
)


def _crit_value_txt(key, v):
    if key == "debt":
        lev = f"levier {ro(v / (1 - v)):.2f}" if v < 1 else "capital propriu negativ"
        return f"{ro(v * 100):.1f}% ({lev})"
    if key in ("margin", "debt", "roa", "roe"):
        return f"{ro(v * 100):.1f}%"
    if key in ("rev_growth", "ni_growth"):
        return f"{ro(v * 100):+.1f}% față de anul precedent"
    if key == "cap_ron":
        return f"{ro(v / 1e9):.2f} mld. lei"
    return f"{ro(v):.2f}"


def my_criteria(values):
    """Criteriile lui Daniel pe valorile companiei. `values`: cheie → număr (fracții la marje/ROA/ROE/datorii,
    variație fracțională la creșteri, lei la capitalizare) sau None. P/E sau P/BV ≤ 0 (pierdere / capital negativ)
    nu trec. Întoarce (rânduri, îndeplinite, evaluabile); rând = {name, threshold, value, ok (True/False/None)}."""
    rows, ok_n, n = [], 0, 0
    for key, name, thr, test in MY_CRITERIA:
        v = _num(values.get(key))
        ok = None if v is None else bool(test(v))
        if ok is not None:
            n += 1
            ok_n += int(ok)
        rows.append({"key": key, "name": name, "threshold": thr,
                     "value": "N/A" if v is None else _crit_value_txt(key, v), "ok": ok})
    return rows, ok_n, n


def growth(new, old):
    """Variația new / old − 1, doar când vechiul e pozitiv (altfel procentul nu are sens); None la lipsă."""
    new, old = _num(new), _num(old)
    if new is None or old is None or old <= 0:
        return None
    return new / old - 1.0


def upside(target, price):
    """Distanța până la o valoare țintă, în fracție: țintă / preț − 1; None la lipsă sau preț ≤ 0."""
    target, price = _num(target), _num(price)
    if target is None or price is None or price <= 0 or target <= 0:
        return None
    return target / price - 1.0


# --- Sentimentul știrilor: un singur set de praguri pentru card, Context, sinteză și SWOT (14.11) -----

def sentiment_band(score):
    """(eticheta, nivel) pentru scorul FinBERT între −1 și +1, pe pragurile SENT_POS / SENT_NEG / SENT_EXTREME.
    Nivel: 'up' / 'down' / 'muted'. Scor lipsă → ('N/A', 'muted')."""
    s = _num(score)
    if s is None:
        return "N/A", "muted"
    if s >= SENT_EXTREME:
        return "Euforie media", "up"
    if s > SENT_POS:
        return "Optimism", "up"
    if s <= -SENT_EXTREME:
        return "Panică", "down"
    if s < SENT_NEG:
        return "Pesimism", "down"
    return "Neutru", "muted"


def news_window(items):
    """„N titluri, dd.mm–dd.mm” din câmpul `date_str` („AAAA-LL-ZZ HH:MM”) al știrilor analizate.
    Știrile fără dată se numără, dar nu intră în interval. Listă goală → „0 titluri”."""
    items = list(items or [])
    n = len(items)
    head = f"{n} {'titlu' if n == 1 else 'titluri'}"
    days = sorted(str(it.get("date_str") or "")[:10] for it in items if len(str(it.get("date_str") or "")) >= 10)
    if not days:
        return head
    fmt = lambda d: f"{d[8:10]}.{d[5:7]}"
    return head + (f", {fmt(days[0])}" if days[0] == days[-1] else f", {fmt(days[0])}–{fmt(days[-1])}")
