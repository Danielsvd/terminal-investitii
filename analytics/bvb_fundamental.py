"""Scorul fundamental BVB, din fișierele „Analiză companii". Funcții pure.

Testul istoric (mai 2017 – sept. 2026, 31 de emitenți BVB, rapoarte folosite doar de la termenul
legal de publicare, randament față de media BVB la 6 luni) a ales patru indicatori, toți relativi
la celelalte companii BVB din fișiere:
  - E/P  = profit net TTM / capitalizare (inversul P/E)          IC +0,17
  - B/P  = capital propriu / capitalizare (inversul P/BV)        IC +0,13
  - DY   = profitul distribuit din ultimul an / capitalizare     IC +0,13
  - Piotroski simplificat (6 criterii, an la an)                 IC +0,28
Scorul = media percentilelor celor patru (minim 3 cu date). Treimea de sus = FAVORABIL,
mijlocul = NEUTRU, treimea de jos = NEFAVORABIL. Scorul pe 4 indicatori: IC +0,25 (t 5,4 fără
suprapunere), stabil în 2017–2021 și 2022–2026. Selecția celor patru s-a făcut după test.
ROE, marja netă și îndatorarea NU au prezis randamentul (IC ≈ 0): se afișează ca informație.
"""
import numpy as np
import pandas as pd

from data.company_files import period_label
from data.helpers import ro

GN_LIMIT = 22.5
SCORED = ("EP", "BP", "DY", "PIO")
MIN_SCORED = 3
MIN_COMPANIES = 6
SPLIT_SUSPECT = 3.0          # preț Yahoo / preț din fișier în afara [1/3, 3] -> posibil split, multipli N/A
SHARE_SPLIT_RATIO = 1.65     # creștere a numărului de acțiuni peste acest raport = split / acțiuni gratuite

TIER_LABELS = {2: ("FAVORABIL", "#3FB950"), 1: ("NEUTRU", "#D29922"), 0: ("NEFAVORABIL", "#F85149")}

# Rezultatele testului (refăcut pe 13 oct. 2026, după corecturile din fișiere): exces mediu față de media BVB
# la 6 luni, % cazuri pozitive, N.
TIER_STATS = {2: (0.060, 0.556, 741), 1: (0.010, 0.473, 706), 0: (-0.070, 0.326, 785)}
GN_STATS = {"ok": (0.029, 0.50, 1219), "over": (-0.036, 0.39, 918)}
TWO_STEP = {
    (2, "high"): ("CANDIDAT, MOMENT FAVORABIL", "#3FB950", (0.092, 0.59, 386)),
    (2, "mid"): ("CANDIDAT, FĂRĂ SEMNAL DE MOMENT", "#238636", (0.043, 0.53, 198)),
    (2, "low"): ("CANDIDAT, TREND NEGATIV", "#D29922", (0.006, 0.52, 157)),
    (1, "high"): ("NEUTRU, MOMENT FAVORABIL", "#D29922", (0.046, 0.54, 340)),
    (1, "mid"): ("NEUTRU", "#D29922", (0.007, 0.46, 170)),
    (1, "low"): ("NEUTRU, TREND NEGATIV", "#F85149", (-0.061, 0.33, 181)),
    (0, "high"): ("NEFAVORABIL (MOMENTUL NU COMPENSEAZĂ)", "#F85149", (-0.007, 0.45, 217)),
    (0, "mid"): ("NEFAVORABIL", "#F85149", (-0.064, 0.30, 224)),
    (0, "low"): ("NEFAVORABIL, TREND NEGATIV", "#F85149", (-0.103, 0.27, 317)),
}
TEST_NOTE = ("Test istoric pe 31 de emitenți BVB din fișierele tale, mai 2017 – sept. 2026: exces față de media "
             "BVB la 6 luni. Media e trasă de câteva câștiguri mari; de aceea contează și procentul de cazuri pozitive. "
             "Limite: doar companiile urmărite azi (bias de supraviețuire), fără costuri de tranzacționare.")


def _ok(*xs):
    return all(x is not None and pd.notna(x) for x in xs)


def _div(a, b):
    return a / b if _ok(a, b) and b > 0 else np.nan


def piotroski_simplified(cur, prev):
    """Șase criterii, raportul de flux curent față de același tip de raport de acum un an.

    Întoarce (scor pe 0–6 sau NaN, lista criteriilor). Scorul se scalează la 6 când lipsește un
    criteriu; cu mai puțin de 5 criterii evaluabile -> NaN. Fără criteriile de flux de numerar
    (nu există în fișiere) și fără marja brută."""
    crit = []

    def add(name, value, ok):
        crit.append({"criteriu": name, "valoare": value, "trece": ok})

    roa = _div(cur.get("ni_ttm"), cur.get("assets"))
    roa0 = _div(prev.get("ni_ttm"), prev.get("assets")) if prev is not None else np.nan
    add("ROA TTM pozitiv", roa, bool(roa > 0) if pd.notna(roa) else None)
    add("ROA în creștere", (roa0, roa), bool(roa > roa0) if _ok(roa, roa0) else None)
    cr = _div(cur.get("assets_cur"), cur.get("liab_st"))
    cr0 = _div(prev.get("assets_cur"), prev.get("liab_st")) if prev is not None else np.nan
    add("Lichiditate curentă în creștere", (cr0, cr), bool(cr > cr0) if _ok(cr, cr0) else None)
    lv = _div(cur.get("liab_lt"), cur.get("assets"))
    lv0 = _div(prev.get("liab_lt"), prev.get("assets")) if prev is not None else np.nan
    add("Datorii pe termen lung / active în scădere", (lv0, lv), bool(lv < lv0) if _ok(lv, lv0) else None)
    sh, sh0 = cur.get("shares"), prev.get("shares") if prev is not None else None
    ratio = sh / sh0 if _ok(sh, sh0) and sh0 > 0 else np.nan
    if pd.notna(ratio) and ratio >= SHARE_SPLIT_RATIO:
        add("Fără emisiune de acțiuni", ratio, None)          # split / acțiuni gratuite: neevaluabil
    else:
        add("Fără emisiune de acțiuni", ratio, bool(ratio <= 1.01) if pd.notna(ratio) else None)
    at = _div(cur.get("rev_ttm"), cur.get("assets"))
    at0 = _div(prev.get("rev_ttm"), prev.get("assets")) if prev is not None else np.nan
    add("Rotația activelor în creștere", (at0, at), bool(at > at0) if _ok(at, at0) else None)
    valid = [c["trece"] for c in crit if c["trece"] is not None]
    score = sum(valid) / len(valid) * 6 if len(valid) >= 5 else np.nan
    return score, crit


def current_indicators(df, price, as_of=None):
    """Indicatorii la zi ai unui emitent.

    `df`: tabelul din `data.company_files.load_company`. `price`: prețul curent (neajustat, lei).
    Ultimul raport din fișier e considerat cunoscut. Capitalizare = preț × acțiunile din ultimul raport.
    Dacă prețul diferă de cel scris în fișier de peste 3 ori (posibil split neactualizat), multiplii devin N/A.
    """
    out = {k: np.nan for k in ("EP", "BP", "DY", "PIO", "PE", "PBV", "GN", "ROE", "marja", "indatorare", "mcap")}
    out.update({"raport_bilant": None, "raport_flux": None, "raport_anual": None, "piotroski": [], "avertismente": []})
    if df is None or df.empty:
        out["avertismente"].append("Fără date în fișier.")
        return out
    if as_of is not None:
        df = df[df.period_end <= pd.Timestamp(as_of)]
        if df.empty:
            out["avertismente"].append("Niciun raport până la data analizei.")
            return out
    bal = df.iloc[-1]
    flows = df[df["ni_ttm"].notna()]
    fl = flows.iloc[-1] if not flows.empty else None
    annual = df[(df.months == 12) & df["div_total"].notna()]
    an = annual.iloc[-1] if not annual.empty else None
    out["raport_bilant"] = period_label(int(bal.year), int(bal.months))
    out["raport_flux"] = period_label(int(fl.year), int(fl.months)) if fl is not None else None
    out["raport_anual"] = period_label(int(an.year), 12) if an is not None else None

    eq = bal.get("equity")
    out["indatorare"] = _div(bal.get("liab"), eq)
    if fl is not None:
        out["ROE"] = _div(fl.get("ni_ttm"), eq)
        out["marja"] = _div(fl.get("ni_ttm"), fl.get("rev_ttm"))
    else:
        out["avertismente"].append("Profitul pe 12 luni (TTM) nu se poate calcula: lipsește anul sau perioada comparabilă.")

    manual = bal.get("price_manual")
    split_suspect = (_ok(price, manual) and manual > 0 and price > 0
                     and not (1 / SPLIT_SUSPECT <= price / manual <= SPLIT_SUSPECT))
    if split_suspect:
        out["avertismente"].append(
            f"Prețul curent ({ro(price):,.4f} lei) diferă de peste {ro(SPLIT_SUSPECT):.0f}× de cel din fișier ({ro(manual):,.4f} lei): "
            "posibil split sau acțiuni gratuite neactualizate în fișier. Multiplii de preț nu se calculează.")
    shares = bal.get("shares")
    if _ok(price, shares) and price > 0 and shares > 0 and not split_suspect:
        mcap = price * shares
        out["mcap"] = mcap
        if fl is not None:
            out["EP"] = _div(fl.get("ni_ttm"), mcap) if _ok(fl.get("ni_ttm")) else np.nan
            out["PE"] = mcap / fl.get("ni_ttm") if _ok(fl.get("ni_ttm")) and fl.get("ni_ttm") > 0 else np.nan
        out["BP"] = _div(eq, mcap) if _ok(eq) else np.nan
        out["PBV"] = mcap / eq if _ok(eq) and eq > 0 else np.nan
        if an is not None:
            out["DY"] = an.get("div_total") / mcap
        if _ok(out["PE"], out["PBV"]):
            out["GN"] = out["PE"] * out["PBV"]
    elif not split_suspect:
        out["avertismente"].append("Lipsește prețul sau numărul de acțiuni: multiplii de preț nu se calculează.")

    if fl is not None:
        prev = df[(df.year == fl.year - 1) & (df.months == fl.months)]
        score, crit = piotroski_simplified(fl.to_dict(), prev.iloc[0].to_dict() if not prev.empty else None)
        out["PIO"], out["piotroski"] = score, crit
    return out


def cross_section(indicators):
    """Percentilele față de celelalte companii BVB din fișiere și treimea fiecăreia.

    `indicators`: {simbol: dict din `current_indicators`}. Întoarce un DataFrame pe simbol cu
    coloanele <indicator>_pct, `scor` (0–100, media percentilelor, minim 3 din 4), `treapta`
    (2 / 1 / 0 sau NaN) și `n_comparate`. Cu mai puțin de 6 companii cu scor: fără treaptă."""
    tab = pd.DataFrame({tk: {k: v.get(k) for k in SCORED} for tk, v in indicators.items()}).T.astype(float)
    if tab.empty:
        return tab
    for k in SCORED:
        tab[k + "_pct"] = tab[k].rank(pct=True)
    pct = tab[[k + "_pct" for k in SCORED]]
    tab["n_indicatori"] = pct.notna().sum(axis=1)
    tab["scor"] = (pct.mean(axis=1, skipna=True) * 100).where(tab["n_indicatori"] >= MIN_SCORED)
    valid = tab["scor"].dropna()
    tab["n_comparate"] = len(valid)
    tab["treapta"] = np.nan
    if len(valid) >= MIN_COMPANIES:
        tiers = pd.qcut(valid.rank(method="first"), 3, labels=False)
        tab.loc[tiers.index, "treapta"] = tiers.astype(float)
    return tab


def moment_band(score):
    if score is None or pd.isna(score):
        return None
    return "high" if score >= 80 else "mid" if score >= 40 else "low"


def final_verdict(tier, moment_score):
    """Verdictul în doi pași: (eticheta, culoarea, textul cu rezultatul istoric al combinației)."""
    if tier is None or pd.isna(tier):
        return "DATE INSUFICIENTE", "#8B949E", "Scorul fundamental nu are destule date (minim 3 din 4 indicatori)."
    band = moment_band(moment_score)
    if band is None:
        lab, col = TIER_LABELS[int(tier)]
        m, p, n = TIER_STATS[int(tier)]
        return (f"FUNDAMENTAL {lab}", col,
                f"Momentul nu se poate calcula. Istoric, treapta fundamentală: {ro(m):+.1%} la 6 luni, {ro(p):.0%} cazuri pozitive (N = {n}).")
    lab, col, (m, p, n) = TWO_STEP[(int(tier), band)]
    return lab, col, f"Istoric, combinația aceasta: {ro(m):+.1%} față de media BVB la 6 luni, {ro(p):.0%} cazuri pozitive (N = {n})."


MOMENT_LABELS = {"high": ("FAVORABIL", "#3FB950"), "mid": ("NEUTRU", "#D29922"), "low": ("NEFAVORABIL", "#F85149")}


def score_rank(scores, symbol):
    """(locul, din câte) al lui `symbol` după scor, 1 = cel mai bun; (None, n) dacă nu are scor."""
    valid = scores.dropna()
    if symbol not in valid.index:
        return None, len(valid)
    return int((valid > valid[symbol]).sum()) + 1, len(valid)


def verdict_banner(tier, score, rank, n, moment_score):
    """Elementele bandei de verdict, fiecare pas cu eticheta și scorul lui (fără etichete combinate).

    Întoarce dict: fund_label, fund_color, fund_detail, moment_label, moment_color, moment_detail,
    border (culoarea combinației din test) și text (rezultatul istoric al combinației)."""
    if tier is None or pd.isna(tier):
        fund_label, fund_color = "DATE INSUFICIENTE", "#8B949E"
    else:
        fund_label, fund_color = TIER_LABELS[int(tier)]
    fund_detail = "N/A" if score is None or pd.isna(score) else f"{ro(score):.0f}/100"
    if rank is not None and n:
        fund_detail += f" · locul {rank} din {n}"
    band = moment_band(moment_score)
    if band is None:
        moment_label, moment_color, moment_detail = "N/A", "#8B949E", "istoric de preț insuficient"
    else:
        moment_label, moment_color = MOMENT_LABELS[band]
        moment_detail = f"{ro(moment_score):.0f}/100"
    _, border, text = final_verdict(tier, moment_score)
    return {"fund_label": fund_label, "fund_color": fund_color, "fund_detail": fund_detail,
            "moment_label": moment_label, "moment_color": moment_color, "moment_detail": moment_detail,
            "border": border, "text": text}


_FACTOR_TEXT = {
    # cheie: (nume, cheia valorii, formulare când e mai bun, formulare când e mai slab)
    "EP": ("P/E", "PE", "mai ieftin raportat la profit decât", "mai scump raportat la profit decât"),
    "BP": ("P/BV", "PBV", "mai ieftin raportat la capitalul propriu decât", "mai scump raportat la capitalul propriu decât"),
    "DY": ("Randamentul dividendului", "DY", "dividend mai mare decât", "dividend mai mic decât"),
    "PIO": ("Piotroski simplificat", "PIO", "s-a îmbunătățit mai mult decât", "s-a îmbunătățit mai puțin decât"),
}
_PIO_FAIL = {
    "ROA TTM pozitiv": "ROA negativ",
    "ROA în creștere": "ROA a scăzut",
    "Lichiditate curentă în creștere": "lichiditatea curentă a scăzut",
    "Datorii pe termen lung / active în scădere": "datoriile pe termen lung / active au crescut",
    "Fără emisiune de acțiuni": "s-au emis acțiuni noi",
    "Rotația activelor în creștere": "rotația activelor a scăzut",
}


def _value_text(key, me):
    v = me.get(_FACTOR_TEXT[key][1])
    if v is None or pd.isna(v):
        return "N/A"
    if key == "DY":
        return f"{ro(v * 100):.1f}%"
    if key == "PIO":
        return f"{ro(v):.1f}/6"
    return f"{ro(v):.2f}"


def _pio_is_pct(criteriu):
    """ROA și datoriile pe termen lung / active sunt rapoarte la active: se afișează în %.
    Lichiditatea, rotația activelor și raportul acțiunilor rămân rapoarte simple."""
    return (criteriu.startswith("ROA") or "active" in criteriu) and "Rotația" not in criteriu


def piotroski_value_text(c):
    """Valoarea unui criteriu Piotroski simplificat pentru tabel: „6,3% → 4,6%”, „1,90 → 1,34”, „4,6%”.
    O valoare lipsă devine „N/A”, nu 0."""
    pct = _pio_is_pct(c["criteriu"])

    def fmt(x):
        if x is None or pd.isna(x):
            return "N/A"
        return f"{ro(x * 100):.1f}%" if pct else f"{ro(x):.2f}"
    v = c.get("valoare")
    if isinstance(v, tuple):
        return " → ".join(fmt(x) for x in v)
    return fmt(v)


def _pio_change(c):
    v = c.get("valoare")
    if isinstance(v, tuple) and all(x is not None and pd.notna(x) for x in v):
        return f" ({piotroski_value_text(c)})"
    return ""


def explain_fundamental(me, cs, symbol):
    """Explicația pasului fundamental: (puncte tari, la mijloc, puncte slabe, alte observații), liste de șiruri.

    Pentru fiecare indicator: față de câte din celelalte companii din fișiere stă mai bine sau mai rău.
    Treimea de sus = punct tare, treimea de jos = punct slab."""
    strong, middle, weak, notes = [], [], [], []
    for key, (name, _, good, bad) in _FACTOR_TEXT.items():
        col = cs[key].dropna() if key in cs else pd.Series(dtype=float)
        mine = cs.at[symbol, key] if symbol in cs.index else np.nan
        if pd.isna(mine) or len(col) < 2:
            notes.append(f"{name}: N/A (lipsesc date).")
            continue
        others = len(col) - 1
        worse, better = int((col < mine).sum()), int((col > mine).sum())
        ties = others - worse - better
        position = (worse + 0.5 * ties) / others          # rangul mediu: egalitățile contează pe jumătate
        tie_txt = f", la egalitate cu {ro(ties / others):.0%}" if ties else ""
        head = f"{name} {_value_text(key, me)} — "
        if position >= 2 / 3:
            strong.append(f"{head}{good} {ro(worse / others):.0%} din celelalte {others} companii{tie_txt}.")
        elif position <= 1 / 3:
            weak.append(f"{head}{bad} {ro(better / others):.0%} din celelalte {others} companii{tie_txt}.")
        else:
            middle.append(f"{head}{good} {ro(worse / others):.0%} din celelalte {others} companii{tie_txt}.")
    failed = [c for c in me.get("piotroski", []) if c["trece"] is False]
    if failed:
        notes.append("Piotroski, ce s-a deteriorat față de acum un an: "
                     + "; ".join(_PIO_FAIL.get(c["criteriu"], c["criteriu"]) + _pio_change(c) for c in failed) + ".")
    gn = me.get("GN")
    if gn is not None and pd.notna(gn):
        m_ok, p_ok, _ = GN_STATS["ok"]
        m_over, p_over, _ = GN_STATS["over"]
        if gn <= GN_LIMIT:
            notes.append(f"GN {ro(gn):.1f} ≤ 22,5: trece criteriul Graham (istoric {ro(m_ok):+.1%} la 6 luni, {ro(p_ok):.0%} pozitive).")
        else:
            notes.append(f"GN {ro(gn):.1f} > 22,5: nu trece criteriul Graham (istoric {ro(m_over):+.1%} la 6 luni, {ro(p_over):.0%} pozitive).")
    return strong, middle, weak, notes
