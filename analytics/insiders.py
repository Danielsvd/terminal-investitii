"""Tranzacțiile insiderilor (SEC Form 4): parsare, clasificare pe coduri, indicatori pe 90 de zile.

Ce este Form 4: raportul unui insider (director, ofițer, acționar > 10%) despre o schimbare a deținerii, depus în
2 zile lucrătoare. E semnalul SEC cel mai proaspăt.

Reguli (propunerea din 25.11.2026, §6.5, aprobată):
  - Semnalul folosește doar codurile P (cumpărare pe piață) și S (vânzare). Vânzările cu bifa 10b5-1 (sau cu nota
    „10b5-1”, înainte de bifa din 2023) sunt „prin plan”, separat de cele discreționare.
  - M (exercitare), F (impozit reținut), A (acordare), G (cadou) și restul apar în tabel, dar nu intră în semnal.
  - Grup = cel puțin 3 insideri distincți cu cumpărări P în fereastră. Cumpărările CEO/CFO sunt marcate.
  - Fără verdict „vânzare masivă”: textul descrie cifrele.
  - Form 4/A înlocuiește raportul inițial al aceluiași insider pentru aceeași dată a raportului (proxy: SEC nu leagă
    explicit amendamentul de original).
  - Un raport depus în comun (ex. fondul și administratorul lui) e un singur insider.
  - Lipsă = None, niciodată 0.

Doar tabelul non-derivat intră în indicatori; derivatele (opțiuni, RSU) se numără separat.
Funcții pure: fără Streamlit, fără rețea.
"""
import logging
import re
import xml.etree.ElementTree as ET

import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_DAYS = 90
GROUP_MIN_BUYERS = 3
SIGNAL_CODES = ("P", "S")
CODE_LABELS = {
    "P": "Cumpărare pe piață", "S": "Vânzare", "M": "Exercitare opțiuni/RSU", "F": "Impozit reținut",
    "A": "Acordare", "G": "Cadou", "D": "Predare către emitent", "C": "Conversie", "X": "Exercitare opțiuni",
    "J": "Altă tranzacție", "W": "Moștenire", "I": "Discreționară (plan)",
}
_CEO_CFO_RE = re.compile(r"\b(CEO|CFO|CHIEF EXECUTIVE|CHIEF FINANCIAL|PRINCIPAL EXECUTIVE|PRINCIPAL FINANCIAL)\b", re.I)
_PLAN_RE = re.compile(r"10b5-?1", re.I)

TX_COLUMNS = ["Data_tranz", "Data_depunere", "Insider", "Functie", "CEO_CFO", "Cod", "Tip", "Plan_10b5_1",
              "Actiuni", "Pret", "Valoare_USD", "Detinere_dupa", "Directa", "Accession", "Forma", "Cheie_insider"]


class ParseForm4Error(ValueError):
    """XML-ul nu e un Form 4 recognoscibil."""


# --------------------------------------------------------------------------- parsare

def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _child(el, name):
    if el is None:
        return None
    return next((c for c in el if _local(c.tag) == name), None)


def _children(el, name):
    return [] if el is None else [c for c in el if _local(c.tag) == name]


def _text(el, *path):
    for name in path:
        el = _child(el, name)
        if el is None:
            return None
    t = (el.text or "").strip()
    return t or None


def _value(el, *path):
    """Câmpurile Form 4 au forma <x><value>…</value><footnoteId/></x>; unele au textul direct."""
    v = _text(el, *path, "value")
    return v if v is not None else _text(el, *path)


def _num(s):
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return None


def _flag(s):
    return str(s or "").strip().lower() in ("1", "true", "y", "yes")


def _footnote_ids(el):
    if el is None:
        return set()
    return {f.get("id") for f in el.iter() if _local(f.tag) == "footnoteId" and f.get("id")}


def parse_form4(xml_bytes, accession=None, filing_date=None):
    """Un Form 4 (sau 4/A) → dict {forma, data_raport, emitent_cik, simbol, insider, functie, ceo_cfo,
    cheie_insider, tranzactii: [dict], derivate: int}. Ridică ParseForm4Error dacă rădăcina nu e ownershipDocument."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ParseForm4Error(f"XML invalid: {exc}") from exc
    if _local(root.tag) != "ownershipDocument":
        raise ParseForm4Error(f"rădăcina e {_local(root.tag)!r}, nu ownershipDocument")

    form = _text(root, "documentType") or "4"
    footnotes = {f.get("id"): (f.text or "") for f in root.iter() if _local(f.tag) == "footnote"}
    doc_plan = _flag(_text(root, "aff10b5One"))

    owners, ciks, roles, ceo_cfo = [], [], [], False
    for ro in _children(root, "reportingOwner"):
        owners.append(_text(ro, "reportingOwnerId", "rptOwnerName") or "?")
        ciks.append(_text(ro, "reportingOwnerId", "rptOwnerCik") or owners[-1])
        rel = _child(ro, "reportingOwnerRelationship")
        r = []
        title = _text(rel, "officerTitle")
        if _flag(_text(rel, "isOfficer")) or title:
            r.append(title or "Ofițer")
            ceo_cfo = ceo_cfo or bool(title and _CEO_CFO_RE.search(title))
        if _flag(_text(rel, "isDirector")):
            r.append("Director")
        if _flag(_text(rel, "isTenPercentOwner")):
            r.append("Acționar > 10%")
        if _flag(_text(rel, "isOther")):
            r.append(_text(rel, "otherText") or "Altul")
        roles.extend(x for x in r if x not in roles)

    txs = []
    for tx in _children(_child(root, "nonDerivativeTable"), "nonDerivativeTransaction"):
        code = _text(tx, "transactionCoding", "transactionCode")
        shares = _num(_value(tx, "transactionAmounts", "transactionShares"))
        price = _num(_value(tx, "transactionAmounts", "transactionPricePerShare"))
        notes = " ".join(footnotes.get(i, "") for i in _footnote_ids(tx))
        txs.append({
            "date": _value(tx, "transactionDate"),
            "code": code,
            "acq_disp": _value(tx, "transactionAmounts", "transactionAcquiredDisposedCode"),
            "shares": shares,
            "price": price if price is not None and price > 0 else None,
            "after": _num(_value(tx, "postTransactionAmounts", "sharesOwnedFollowingTransaction")),
            "direct": _value(tx, "ownershipNature", "directOrIndirectOwnership"),
            "plan": doc_plan or bool(_PLAN_RE.search(notes)),
        })
    derivs = len(_children(_child(root, "derivativeTable"), "derivativeTransaction"))
    return {
        "forma": form, "data_raport": _text(root, "periodOfReport"),
        "emitent_cik": _text(root, "issuer", "issuerCik"), "simbol": _text(root, "issuer", "issuerTradingSymbol"),
        "insider": " / ".join(owners) or "?", "functie": ", ".join(roles) or "N/A", "ceo_cfo": ceo_cfo,
        "cheie_insider": "|".join(sorted(ciks)) or None, "tranzactii": txs, "derivate": derivs,
        "accession": accession, "data_depunere": filing_date,
    }


# --------------------------------------------------------------------------- tabel

def transactions_frame(docs):
    """Lista de Form 4 parsate → DataFrame cu TX_COLUMNS, cele mai noi primele. Un 4/A înlocuiește rapoartele 4
    ale aceluiași insider cu aceeași dată a raportului (depuse înaintea lui)."""
    docs = [d for d in docs if d]
    drop = set()
    for a in docs:
        if str(a["forma"]).upper() != "4/A":
            continue
        for d in docs:
            if (d is not a and str(d["forma"]).upper() == "4" and d["cheie_insider"] == a["cheie_insider"]
                    and d["data_raport"] == a["data_raport"] and (d["data_depunere"] or "") <= (a["data_depunere"] or "")):
                drop.add(id(d))
    rows = []
    for d in docs:
        if id(d) in drop:
            continue
        for t in d["tranzactii"]:
            value = t["shares"] * t["price"] if t["shares"] is not None and t["price"] is not None else None
            rows.append({
                "Data_tranz": t["date"], "Data_depunere": d["data_depunere"], "Insider": d["insider"],
                "Functie": d["functie"], "CEO_CFO": d["ceo_cfo"], "Cod": t["code"],
                "Tip": CODE_LABELS.get(t["code"], t["code"] or "N/A"), "Plan_10b5_1": t["plan"],
                "Actiuni": t["shares"], "Pret": t["price"], "Valoare_USD": value, "Detinere_dupa": t["after"],
                "Directa": t["direct"], "Accession": d["accession"], "Forma": d["forma"],
                "Cheie_insider": d["cheie_insider"],
            })
    df = pd.DataFrame(rows, columns=TX_COLUMNS)
    if df.empty:
        return df
    df["Data_tranz"] = pd.to_datetime(df["Data_tranz"], errors="coerce")
    df["Data_depunere"] = pd.to_datetime(df["Data_depunere"], errors="coerce")
    return df.sort_values(["Data_tranz", "Data_depunere"], ascending=False, na_position="last").reset_index(drop=True)


# --------------------------------------------------------------------------- indicatori

def _sum_or_none(s):
    s = pd.Series(s).dropna()
    return float(s.sum()) if len(s) else None


def window_summary(df, asof, days=WINDOW_DAYS, group_min=GROUP_MIN_BUYERS):
    """Indicatorii pe ultimele `days` zile până la `asof` (după data tranzacției):
    cumpărători distincți P, valoarea P, grup, cumpărări CEO/CFO, cumpărarea ca % din deținerea anterioară (mediană),
    vânzări discreționare și prin plan (număr și valoare), restul codurilor numărate. Valorile fără preț nu intră
    în sume, dar se numără în `fara_pret`."""
    asof = pd.Timestamp(asof).normalize()
    start = asof - pd.Timedelta(days=days)
    out = {"start": start, "end": asof, "zile": days, "tranzactii": 0, "cumparatori": 0, "valoare_P": None,
           "nr_P": 0, "grup": False, "ceo_cfo_P": False, "pct_din_detinere_P": None,
           "nr_S_discretionar": 0, "valoare_S_discretionar": None, "nr_S_plan": 0, "valoare_S_plan": None,
           "alte_coduri": {}, "fara_pret": 0}
    if df is None or df.empty:
        return out
    w = df[(df["Data_tranz"] > start) & (df["Data_tranz"] <= asof)]
    out["tranzactii"] = int(len(w))
    if w.empty:
        return out
    p = w[w["Cod"] == "P"]
    s = w[w["Cod"] == "S"]
    s_plan, s_disc = s[s["Plan_10b5_1"].astype(bool)], s[~s["Plan_10b5_1"].astype(bool)]
    out["nr_P"], out["cumparatori"] = int(len(p)), int(p["Cheie_insider"].nunique())
    out["valoare_P"] = _sum_or_none(p["Valoare_USD"])
    out["grup"] = out["cumparatori"] >= group_min
    out["ceo_cfo_P"] = bool(p["CEO_CFO"].any()) if len(p) else False
    before = p["Detinere_dupa"] - p["Actiuni"]
    pct = (p["Actiuni"] / before)[before > 0]
    out["pct_din_detinere_P"] = float(pct.median()) if len(pct) else None
    out["nr_S_discretionar"], out["valoare_S_discretionar"] = int(len(s_disc)), _sum_or_none(s_disc["Valoare_USD"])
    out["nr_S_plan"], out["valoare_S_plan"] = int(len(s_plan)), _sum_or_none(s_plan["Valoare_USD"])
    other = w[~w["Cod"].isin(SIGNAL_CODES)]
    out["alte_coduri"] = {str(k): int(v) for k, v in other["Cod"].fillna("?").value_counts().items()}
    sig = w[w["Cod"].isin(SIGNAL_CODES)]
    out["fara_pret"] = int(sig["Valoare_USD"].isna().sum())
    return out


def describe(summary, fmt_usd=lambda v: f"{v:,.0f} $"):
    """Textul descriptiv al cardului, fără verdict."""
    def money(v):
        return "N/A" if v is None else fmt_usd(v)
    parts = [f"Cumpărări pe piață: {summary['nr_P']} ({summary['cumparatori']} insideri, {money(summary['valoare_P'])})",
             f"vânzări discreționare: {summary['nr_S_discretionar']} ({money(summary['valoare_S_discretionar'])})",
             f"vânzări prin plan 10b5-1: {summary['nr_S_plan']} ({money(summary['valoare_S_plan'])})"]
    text = "; ".join(parts) + "."
    if summary["grup"]:
        text += f" Grup: ≥ {GROUP_MIN_BUYERS} insideri au cumpărat."
    if summary["ceo_cfo_P"]:
        text += " Între cumpărători: CEO/CFO."
    return text


# --------------------------------------------------------------------------- depuneri

def raw_xml_name(primary_document):
    """Documentul principal al unui Form 4 în lista SEC e versiunea randată (ex. 'xslF345X05/form4.xml');
    XML-ul brut e același nume fără folderul 'xsl…/'."""
    name = str(primary_document or "")
    return name.split("/", 1)[1] if name.lower().startswith("xsl") and "/" in name else name


def cik_for_ticker(company_tickers, symbol):
    """CIK-ul din `company_tickers.json` al SEC pentru un simbol (BRK-B ↔ BRK.B), sau None."""
    sym = str(symbol or "").upper().replace("-", ".")
    for row in (company_tickers or {}).values():
        if str(row.get("ticker", "")).upper().replace("-", ".") == sym:
            return str(row.get("cik_str"))
    return None
