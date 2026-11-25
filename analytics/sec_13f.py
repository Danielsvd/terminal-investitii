"""Rapoartele 13F ale administratorilor americani: parsare, validare, portofoliu pe acțiuni, mișcări, convingere, consens.

Ce este 13F: raportul trimestrial al administratorilor cu peste 100 mil. $ în titluri americane. Arată doar pozițiile
long în acțiuni (și opțiuni pe ele) la sfârșitul trimestrului, depuse în cel mult 45 de zile. Nu arată short-urile,
acoperirile sau pozițiile din afara SUA; pozițiile cu tratament confidențial pot lipsi temporar.

Reguli (propunerea din 25.11.2026, §6):
  - Portofoliul pe acțiuni = rândurile `SH` fără `putCall`, agregate pe CUSIP. Opțiunile și obligațiunile (`PRN`) se
    țin separat și nu intră în ponderi.
  - Valoarea: în dolari întregi pentru depunerile din 03.01.2023 încoace, în mii de dolari înainte (presupunere de
    confirmat pe un raport real; `implied_price` ajută la verificare).
  - Mișcările se judecă pe numărul de acțiuni (ajustat cu split-ul), nu pe valoare: variația valorii amestecă prețul
    cu tranzacția. Valoarea tranzacționată = Δ acțiuni × prețul mediu din trimestru — proxy, numit așa.
  - Emitentul = primele 6 caractere ale CUSIP (CUSIP6): GOOG și GOOGL sunt același emitent în consens.
  - Lipsă = NaN, niciodată 0.

Funcții pure: fără Streamlit, fără rețea.
"""
import logging
import xml.etree.ElementTree as ET
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DOLLAR_VALUES_FROM = date(2023, 1, 3)   # de la această dată de depunere, `value` e în dolari (înainte: mii)
MOVE_THRESHOLD = 0.05                   # |Δ acțiuni| ≤ 5% → „Păstrată”
CONVICTION_WEIGHT = 0.03                # poziție de convingere: pondere ≥ 3% ...
CONVICTION_TOP_N = 10                   # ... sau în primele 10
CONCENTRATED_MAX_POSITIONS = 100        # administrator concentrat: ≤ 100 de poziții ...
CONCENTRATED_TOP10_SHARE = 0.50         # ... sau primele 10 ≥ 50% din valoare
CONVICTION_BUY_GROWTH = 0.50            # cumpărare de convingere: nouă sau mărită ≥ +50% ...
CONVICTION_BUY_MIN_WEIGHT = 0.01        # ... cu pondere finală ≥ 1%

ACT_NEW, ACT_CLOSED, ACT_UP, ACT_DOWN, ACT_HELD = "Nouă", "Închisă", "Mărită", "Redusă", "Păstrată"

INFO_COLUMNS = ["Emitent", "Clasa_titlu", "CUSIP", "FIGI", "Valoare_raportata", "Actiuni", "Tip",
                "PutCall", "Discretie", "Alt_manager"]
BOOK_COLUMNS = ["CUSIP", "CUSIP6", "Emitent", "Clasa_titlu", "Actiuni", "Valoare_USD", "Pondere", "Rang"]
OPTION_COLUMNS = ["CUSIP", "CUSIP6", "Emitent", "PutCall", "Actiuni", "Valoare_USD"]
MOVE_COLUMNS = ["CUSIP", "CUSIP6", "Emitent", "Actiune", "Actiuni_ant", "Actiuni", "Delta_actiuni", "Delta_pct",
                "Pondere_ant", "Pondere", "Delta_pp", "Pret_mediu", "Valoare_tranzactionata", "Impact"]


class Parse13FError(ValueError):
    """XML-ul nu e un raport 13F recognoscibil."""


# --------------------------------------------------------------------------- parsare

def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _find(el, *path):
    """Primul descendent pe calea dată, ignorând spațiile de nume."""
    cur = el
    for name in path:
        nxt = next((c for c in cur if _local(c.tag) == name), None)
        if nxt is None:
            return None
        cur = nxt
    return cur


def _text(el, *path):
    node = _find(el, *path) if path else el
    if node is None or node.text is None:
        return None
    t = node.text.strip()
    return t or None


def _num(s):
    if s is None:
        return np.nan
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return np.nan


def _root(xml_bytes):
    try:
        return ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise Parse13FError(f"XML invalid: {exc}") from exc


def is_info_table(xml_bytes):
    """Adevărat dacă rădăcina e `informationTable` (așa se recunoaște fișierul cu nume liber)."""
    try:
        return _local(_root(xml_bytes).tag) == "informationTable"
    except Parse13FError:
        return False


def parse_cover(xml_bytes):
    """Coperta (`primary_doc.xml`): perioada, amendamentul, totalurile declarate, numele declarantului.

    Totalurile (`tableEntryTotal`, `tableValueTotal`) servesc la validare: suma rândurilor trebuie să le refacă."""
    root = _root(xml_bytes)
    if _local(root.tag) != "edgarSubmission":
        raise Parse13FError(f"copertă neașteptată: rădăcina {_local(root.tag)!r}")
    form = _find(root, "formData")
    cover = _find(form, "coverPage") if form is not None else None
    summary = _find(form, "summaryPage") if form is not None else None
    if cover is None:
        raise Parse13FError("lipsește coverPage")
    period = _text(cover, "reportCalendarOrQuarter")
    is_amend = (_text(cover, "isAmendment") or "").lower() == "true"
    entry_total = _num(_text(summary, "tableEntryTotal")) if summary is not None else np.nan
    value_total = _num(_text(summary, "tableValueTotal")) if summary is not None else np.nan
    return {
        "period": _parse_date(period),
        "is_amendment": is_amend,
        "amendment_type": (_text(cover, "amendmentInfo", "amendmentType") or "").upper() or None,
        "filer_name": _text(cover, "filingManager", "name"),
        "entry_total": None if np.isnan(entry_total) else int(entry_total),
        "value_total": None if np.isnan(value_total) else value_total,
        "other_managers": _num(_text(summary, "otherIncludedManagersCount")) if summary is not None else np.nan,
    }


def _parse_date(s):
    if not s:
        return None
    for fmt in ("%m-%d-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return pd.to_datetime(s, format=fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def parse_info_table(xml_bytes):
    """Tabelul de informații: un rând pe `infoTable`, coloanele `INFO_COLUMNS`. Valoarea e cea raportată (unitatea
    depinde de data depunerii, vezi `value_multiplier`)."""
    root = _root(xml_bytes)
    if _local(root.tag) != "informationTable":
        raise Parse13FError(f"tabel neașteptat: rădăcina {_local(root.tag)!r}")
    rows = []
    for it in _children(root, "infoTable"):
        put_call = (_text(it, "putCall") or "").strip().capitalize() or None
        rows.append({
            "Emitent": _text(it, "nameOfIssuer"),
            "Clasa_titlu": _text(it, "titleOfClass"),
            "CUSIP": (_text(it, "cusip") or "").upper() or None,
            "FIGI": _text(it, "figi"),
            "Valoare_raportata": _num(_text(it, "value")),
            "Actiuni": _num(_text(it, "shrsOrPrnAmt", "sshPrnamt")),
            "Tip": (_text(it, "shrsOrPrnAmt", "sshPrnamtType") or "").upper() or None,
            "PutCall": put_call,
            "Discretie": _text(it, "investmentDiscretion"),
            "Alt_manager": _text(it, "otherManager"),
        })
    return pd.DataFrame(rows, columns=INFO_COLUMNS)


def value_multiplier(filing_date):
    """× 1 pentru depunerile din 03.01.2023 încoace (dolari), × 1000 înainte (mii de dolari)."""
    d = pd.Timestamp(filing_date).date()
    return 1.0 if d >= DOLLAR_VALUES_FROM else 1000.0


def validate(cover, info):
    """Probleme care opresc scrierea raportului (listă goală = valid). Totalurile copertei sunt în aceeași unitate
    ca rândurile, deci comparația nu depinde de multiplicator. Toleranța la valoare: 1 unitate pe rând (rotunjiri)."""
    problems = []
    n = len(info)
    if cover.get("entry_total") is not None and n != cover["entry_total"]:
        problems.append(f"rânduri {n} ≠ tableEntryTotal {cover['entry_total']}")
    vt = cover.get("value_total")
    if vt is not None and n:
        s = float(info["Valoare_raportata"].sum(min_count=1))
        if np.isnan(s) or abs(s - vt) > max(1.0, n):
            problems.append(f"suma valorilor {s:,.0f} ≠ tableValueTotal {vt:,.0f}")
    if n and info["CUSIP"].isna().any():
        problems.append(f"{int(info['CUSIP'].isna().sum())} rânduri fără CUSIP")
    if cover.get("period") is None:
        problems.append("perioada lipsește din copertă")
    return problems


def apply_amendment(base, amendment, amendment_type):
    """Un 13F-HR/A „RESTATEMENT” înlocuiește tabelul; „NEW HOLDINGS” adaugă rânduri la cel inițial."""
    kind = (amendment_type or "").upper()
    if kind == "RESTATEMENT":
        return amendment.copy()
    if kind == "NEW HOLDINGS":
        return pd.concat([base, amendment], ignore_index=True)
    raise ValueError(f"tip de amendament necunoscut: {amendment_type!r}")


# --------------------------------------------------------------------------- portofoliu

def implied_price(info, multiplier):
    """Prețul implicit pe rând = valoare × multiplicator / acțiuni (doar SH). Folosit la verificarea unității valorii:
    un preț de 1000× sau 0,001× față de piață arată un multiplicator greșit."""
    sh = info["Tip"].eq("SH") & info["Actiuni"].gt(0)
    out = pd.Series(np.nan, index=info.index)
    out[sh] = info.loc[sh, "Valoare_raportata"] * multiplier / info.loc[sh, "Actiuni"]
    return out


def equity_book(info, multiplier):
    """Portofoliul pe acțiuni: SH fără opțiuni, agregat pe CUSIP (același titlu poate apărea pe mai multe rânduri,
    pe discreții sau administratori diferiți). Coloane `BOOK_COLUMNS`, sortat după valoare."""
    df = info[info["Tip"].eq("SH") & info["PutCall"].isna() & info["CUSIP"].notna()]
    if df.empty:
        return pd.DataFrame(columns=BOOK_COLUMNS)
    g = df.groupby("CUSIP", sort=False).agg(
        Emitent=("Emitent", "first"), Clasa_titlu=("Clasa_titlu", "first"),
        Actiuni=("Actiuni", lambda s: s.sum(min_count=1)),
        Valoare_raportata=("Valoare_raportata", lambda s: s.sum(min_count=1)),
    ).reset_index()
    g["Valoare_USD"] = g["Valoare_raportata"] * multiplier
    total = g["Valoare_USD"].sum(min_count=1)
    g["Pondere"] = g["Valoare_USD"] / total if total and total > 0 else np.nan
    g["CUSIP6"] = g["CUSIP"].str[:6]
    g = g.sort_values("Valoare_USD", ascending=False, na_position="last").reset_index(drop=True)
    g["Rang"] = np.arange(1, len(g) + 1)
    return g[BOOK_COLUMNS]


def options_book(info, multiplier):
    """Opțiunile din raport (valoarea acțiunilor din spatele lor, cum a raportat-o fondul), separat de ponderi."""
    df = info[info["PutCall"].notna() & info["CUSIP"].notna()]
    if df.empty:
        return pd.DataFrame(columns=OPTION_COLUMNS)
    g = df.groupby(["CUSIP", "PutCall"], sort=False).agg(
        Emitent=("Emitent", "first"),
        Actiuni=("Actiuni", lambda s: s.sum(min_count=1)),
        Valoare_raportata=("Valoare_raportata", lambda s: s.sum(min_count=1)),
    ).reset_index()
    g["Valoare_USD"] = g["Valoare_raportata"] * multiplier
    g["CUSIP6"] = g["CUSIP"].str[:6]
    return g.sort_values("Valoare_USD", ascending=False).reset_index(drop=True)[OPTION_COLUMNS]


def classify_manager(book, max_positions=CONCENTRATED_MAX_POSITIONS, top10_share=CONCENTRATED_TOP10_SHARE):
    """„Concentrat” dacă are ≤ 100 de poziții SAU primele 10 fac ≥ 50% din valoare; altfel „Larg”.
    Clasificarea automată e o propunere: Daniel o confirmă în foaia `SEC_administratori`."""
    if book is None or book.empty:
        return None
    n = len(book)
    top10 = book.nsmallest(10, "Rang")["Pondere"].sum(min_count=1)
    if n <= max_positions or (not np.isnan(top10) and top10 >= top10_share):
        return "Concentrat"
    return "Larg"


def conviction_mask(book, min_weight=CONVICTION_WEIGHT, top_n=CONVICTION_TOP_N):
    """Poziție de convingere: pondere ≥ 3% sau în primele 10."""
    return book["Pondere"].ge(min_weight) | book["Rang"].le(top_n)


# --------------------------------------------------------------------------- mișcări trimestriale

def quarter_moves(prev, curr, split_factor=None, avg_price=None, threshold=MOVE_THRESHOLD):
    """Mișcările între două portofolii (`equity_book`) ale aceluiași administrator.

    split_factor: {CUSIP: factor} — acțiunile anterioare se înmulțesc cu factorul (split 4:1 → 4).
    avg_price:    {CUSIP: prețul mediu de închidere din trimestrul curent} → valoarea tranzacționată (proxy).
    Impact = |valoarea tranzacționată| / valoarea portofoliului curent: cât a contat mișcarea pentru fond.
    Ordinea: după Impact (descrescător), mișcările fără preț la final, după |Δ pp|."""
    split_factor = split_factor or {}
    avg_price = avg_price or {}
    p = prev.set_index("CUSIP") if prev is not None and not prev.empty else pd.DataFrame(columns=BOOK_COLUMNS).set_index("CUSIP")
    c = curr.set_index("CUSIP") if curr is not None and not curr.empty else pd.DataFrame(columns=BOOK_COLUMNS).set_index("CUSIP")
    total_curr = c["Valoare_USD"].sum(min_count=1) if len(c) else np.nan
    rows = []
    for cusip in list(c.index) + [x for x in p.index if x not in c.index]:
        in_p, in_c = cusip in p.index, cusip in c.index
        sh_prev = p.at[cusip, "Actiuni"] * split_factor.get(cusip, 1.0) if in_p else 0.0
        sh_curr = c.at[cusip, "Actiuni"] if in_c else 0.0
        w_prev = p.at[cusip, "Pondere"] if in_p else 0.0
        w_curr = c.at[cusip, "Pondere"] if in_c else 0.0
        if not in_p or sh_prev == 0:
            action, pct = ACT_NEW, np.nan
        elif not in_c:
            action, pct = ACT_CLOSED, -1.0
        else:
            pct = (sh_curr - sh_prev) / sh_prev if sh_prev else np.nan
            if np.isnan(pct):
                action = ACT_HELD
            elif pct > threshold:
                action = ACT_UP
            elif pct < -threshold:
                action = ACT_DOWN
            else:
                action = ACT_HELD
        delta = sh_curr - sh_prev
        price = avg_price.get(cusip, np.nan)
        price = np.nan if price is None else float(price)
        traded = delta * price if not np.isnan(price) else np.nan
        impact = abs(traded) / total_curr if not np.isnan(traded) and total_curr and total_curr > 0 else np.nan
        src = c if in_c else p
        rows.append({
            "CUSIP": cusip, "CUSIP6": str(cusip)[:6], "Emitent": src.at[cusip, "Emitent"], "Actiune": action,
            "Actiuni_ant": sh_prev if in_p else 0.0, "Actiuni": sh_curr, "Delta_actiuni": delta, "Delta_pct": pct,
            "Pondere_ant": w_prev, "Pondere": w_curr, "Delta_pp": (w_curr - w_prev) * 100,
            "Pret_mediu": price, "Valoare_tranzactionata": traded, "Impact": impact,
        })
    out = pd.DataFrame(rows, columns=MOVE_COLUMNS)
    if out.empty:
        return out
    out["_k1"] = out["Impact"].fillna(-1.0)
    out["_k2"] = out["Delta_pp"].abs()
    return out.sort_values(["_k1", "_k2"], ascending=False).drop(columns=["_k1", "_k2"]).reset_index(drop=True)


def move_counts(moves):
    """Numărul de mișcări pe tip. Verificare: noi + mărite + reduse + păstrate = poziții curente."""
    vc = moves["Actiune"].value_counts() if moves is not None and not moves.empty else pd.Series(dtype=int)
    return {a: int(vc.get(a, 0)) for a in (ACT_NEW, ACT_CLOSED, ACT_UP, ACT_DOWN, ACT_HELD)}


# --------------------------------------------------------------------------- consens

def consensus(holdings, moves=None):
    """Consensul pe emitent (CUSIP6), doar între administratorii concentrați.

    holdings: rândurile `equity_book` ale tuturor administratorilor, cu coloanele în plus `CIK`, `Fond`, `Clasa`
              („Concentrat”/„Larg”). Clasele de acțiuni ale aceluiași emitent se adună pe administrator.
    moves:    opțional, mișcările din ultimul trimestru, cu `CIK` → cumpărările de convingere.
    Coloane: CUSIP6, Emitent, Detinatori (administratori concentrați care dețin), Convingere (cu poziție de
    convingere), Cumparari_convingere (nouă sau mărită ≥ +50%, cu pondere finală ≥ 1%), Fonduri."""
    cols = ["CUSIP6", "Emitent", "Detinatori", "Convingere", "Cumparari_convingere", "Fonduri"]
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=cols)
    h = holdings[holdings["Clasa"].eq("Concentrat")].copy()
    if h.empty:
        return pd.DataFrame(columns=cols)
    h["_conv"] = conviction_mask(h)
    per = h.groupby(["CUSIP6", "CIK"]).agg(Emitent=("Emitent", "first"), Fond=("Fond", "first"),
                                          Pondere=("Pondere", "sum"), conv=("_conv", "max")).reset_index()
    per["conv"] = per["conv"] | per["Pondere"].ge(CONVICTION_WEIGHT)
    out = per.groupby("CUSIP6").agg(
        Emitent=("Emitent", "first"), Detinatori=("CIK", "nunique"), Convingere=("conv", "sum"),
        Fonduri=("Fond", lambda s: ", ".join(sorted(set(s)))),
    ).reset_index()
    out["Cumparari_convingere"] = 0
    if moves is not None and not moves.empty:
        conc = set(h["CIK"])
        m = moves[moves["CIK"].isin(conc)]
        buy = m["Actiune"].eq(ACT_NEW) | (m["Actiune"].eq(ACT_UP) & m["Delta_pct"].ge(CONVICTION_BUY_GROWTH))
        buy &= m["Pondere"].ge(CONVICTION_BUY_MIN_WEIGHT)
        n_buy = m[buy].groupby("CUSIP6")["CIK"].nunique()
        out["Cumparari_convingere"] = out["CUSIP6"].map(n_buy).fillna(0).astype(int)
    out["Convingere"] = out["Convingere"].astype(int)
    return out.sort_values(["Convingere", "Cumparari_convingere", "Detinatori"], ascending=False).reset_index(drop=True)[cols]


def who_holds(holdings, cusip6):
    """Cine deține emitentul: concentrații întâi, apoi ceilalți, cu ponderea și rangul în fiecare portofoliu."""
    cols = ["CIK", "Fond", "Clasa", "Actiuni", "Valoare_USD", "Pondere", "Rang", "Convingere"]
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=cols)
    h = holdings[holdings["CUSIP6"].eq(str(cusip6)[:6])]
    if h.empty:
        return pd.DataFrame(columns=cols)
    g = h.groupby("CIK").agg(Fond=("Fond", "first"), Clasa=("Clasa", "first"),
                             Actiuni=("Actiuni", lambda s: s.sum(min_count=1)),
                             Valoare_USD=("Valoare_USD", lambda s: s.sum(min_count=1)),
                             Pondere=("Pondere", "sum"), Rang=("Rang", "min")).reset_index()
    g["Convingere"] = g["Pondere"].ge(CONVICTION_WEIGHT) | g["Rang"].le(CONVICTION_TOP_N)
    g["_c"] = g["Clasa"].eq("Concentrat")
    return g.sort_values(["_c", "Pondere"], ascending=False).drop(columns="_c").reset_index(drop=True)[cols]


# --------------------------------------------------------------------------- prețuri pentru mișcări

def raw_closes(close_adj, splits):
    """Închiderile în unitățile de la data lor (neajustate pentru split-urile ulterioare).

    Yahoo dă închiderile ajustate pentru split (și cu `auto_adjust=False`): închiderea de la data d e împărțită la
    produsul split-urilor de după d. Acțiunile din 13F sunt în unitățile de la sfârșitul trimestrului, deci prețul
    trebuie adus în aceleași unități: brut(d) = ajustat(d) × produsul split-urilor de după d.
    `splits`: coloana „Stock Splits” din yf.download(actions=True); 0 sau NaN = fără split."""
    if close_adj is None or close_adj.empty:
        return close_adj
    out = close_adj.copy()
    if splits is None or splits.empty:
        return out
    s = splits.reindex(index=out.index, columns=out.columns)
    s = s.where(s > 0, 1.0).fillna(1.0)
    after = s.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)   # produsul split-urilor strict după d
    return out * after


def period_inputs(close_adj, splits, prev_period, curr_period, cusip_to_symbol):
    """Factorii de split și prețurile medii pentru `quarter_moves`, între două perioade 13F.

    split_factor[CUSIP] = produsul split-urilor din (perioada anterioară, perioada curentă].
    avg_price[CUSIP]    = media închiderilor brute (unitățile de la data lor) din același interval — proxy pentru
                          prețul tranzacțiilor din trimestru. Lipsă de date → CUSIP-ul lipsește din dicționar (NaN)."""
    split_factor, avg_price = {}, {}
    if close_adj is None or close_adj.empty:
        return split_factor, avg_price
    lo, hi = pd.Timestamp(prev_period), pd.Timestamp(curr_period)
    raw = raw_closes(close_adj, splits)
    win = (raw.index > lo) & (raw.index <= hi)
    for cusip, sym in cusip_to_symbol.items():
        if not sym or sym not in raw.columns:
            continue
        px = raw.loc[win, sym].dropna()
        px = px[px > 0]
        if len(px):
            avg_price[cusip] = float(px.mean())
        if splits is not None and not splits.empty and sym in splits.columns:
            sp = splits.loc[(splits.index > lo) & (splits.index <= hi), sym]
            sp = sp[sp > 0]
            if len(sp):
                split_factor[cusip] = float(sp.prod())
    return split_factor, avg_price
