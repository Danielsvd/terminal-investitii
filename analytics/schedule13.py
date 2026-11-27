"""Deținerile peste 5% (SEC Schedule 13D / 13G) ale unui emitent: parsare, cine deține acum, semnale.

Ce sunt:
  - 13D: cine trece de 5% dintr-o clasă de acțiuni cu drept de vot și poate urmări influențarea companiei
    (activist). Depus în 5 zile lucrătoare de la prag (din 2025; înainte 10 zile).
  - 13G: aceeași deținere, dar pasivă (fonduri index, instituții, investitori fără intenție de control). Termene mai
    lungi (45 de zile după trimestru la instituții).

Reguli (propunerea din 25.11.2026, §6.6, aprobată: lista pe companie, 13D = activist, 13G = pasiv) și alegerile de
implementare de mai jos (de confirmat de Daniel):
  - Un depunător (CIK-ul din antet; altfel numele) = un rând; contează **ultima** lui depunere. Tipul (13D/13G)
    vine din ultima depunere: un fond care trece de la 13G la 13D e marcat „13G → 13D”.
  - O depunere în grup (fondul + administratorul + persoanele care îl controlează) raportează aceleași acțiuni de
    mai multe ori: procentul depunerii = **maximul** pe persoane, nu suma.
  - Procent < 5 în ultima depunere = a ieșit sub prag (rândul rămâne, marcat „sub 5%”).
  - Depunerile vechi (SC 13D / SC 13G, text sau HTML, dinainte de 18.12.2024) nu au XML: se arată cu numele din
    antet și procentul „N/A”, niciodată 0.
  - Lipsă = None → „N/A”.

Format XML (din 18.12.2024, `primary_doc.xml`): spațiile de nume `schedule13D` și `schedule13g` diferă și ca nume
de elemente (13D: dateOfEvent, issuerCIK, percentOfClass, aggregateAmountOwned; 13G:
eventDateRequiresFilingThisStatement, issuerCik, classPercent, reportingPersonBeneficiallyOwnedAggregateNumberOfShares).
Parsarea caută după numele local și acceptă ambele variante.

Funcții pure: fără Streamlit, fără rețea.
"""
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

THRESHOLD_PCT = 5.0
FORMS_XML = ("SCHEDULE 13D", "SCHEDULE 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A")
FORMS_OLD = ("SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A")
FORMS = FORMS_XML + FORMS_OLD
KIND_LABELS = {"13D": "13D (activist)", "13G": "13G (pasiv)"}

_PCT_TAGS = ("percentOfClass", "classPercent")
_SHARES_TAGS = ("aggregateAmountOwned", "reportingPersonBeneficiallyOwnedAggregateNumberOfShares")
_EVENT_TAGS = ("dateOfEvent", "eventDateRequiresFilingThisStatement")
_ISSUER_CIK_TAGS = ("issuerCIK", "issuerCik")
_CUSIP_TAGS = ("issuerCUSIP", "issuerCusip", "cusipNumber")
_PERSON_TAGS = ("reportingPersonInfo", "coverPageHeaderReportingPersonDetails")
_PERSON_CIK_TAGS = ("reportingPersonCIK", "reportingPersonCik")

HOLDER_COLUMNS = ["Cheie", "Detinator", "CIK_depunator", "Tip", "Trecere", "Procent", "Actiuni", "Sub_prag",
                  "Data_eveniment", "Data_depunere", "Forma", "Persoane", "Depuneri", "Accession", "Format"]


class ParseScheduleError(ValueError):
    """Documentul nu e un Schedule 13D/13G recognoscibil."""


# --------------------------------------------------------------------------- utilitare

def form_kind(form):
    """'13D' sau '13G' din tipul formularului (SC 13D/A, SCHEDULE 13G …); None pentru altceva."""
    f = str(form or "").upper()
    if "13D" in f:
        return "13D"
    if "13G" in f:
        return "13G"
    return None


def is_amendment(form):
    return str(form or "").strip().upper().endswith("/A")


def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _first(root, names):
    """Textul primului element (oriunde în arbore) cu unul dintre numele locale date; None dacă lipsește."""
    for el in root.iter():
        if _local(el.tag) in names:
            t = (el.text or "").strip()
            if t:
                return t
    return None


def _num(s):
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except ValueError:
        return None


def _date(s):
    """MM/DD/YYYY (formatul SEC din XML) sau YYYY-MM-DD → Timestamp; altfel None."""
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y%m%d"):
        try:
            return pd.Timestamp(datetime.strptime(str(s).strip(), fmt))
        except ValueError:
            continue
    return None


def _cik(s):
    s = str(s or "").strip()
    return str(int(s)) if s.isdigit() else None


# --------------------------------------------------------------------------- parsare

def parse_schedule_xml(body, accession=None, filing_date=None, form=None):
    """`primary_doc.xml` al unui SCHEDULE 13D/13G (format din 18.12.2024) → dicționar.

    Chei: accession, form, kind ('13D'/'13G'), amendment (bool), amendment_no, filing_date, event_date,
    issuer_cik, cusip, class_title, filer_cik, persons [{name, cik, percent, shares, type}], format='XML'."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ParseScheduleError(f"XML invalid: {exc}") from exc
    sub = _first(root, ("submissionType",)) or form
    kind = form_kind(sub)
    if kind is None:
        raise ParseScheduleError(f"nu e un Schedule 13D/13G (tip: {sub!r})")
    persons = []
    for el in root.iter():
        if _local(el.tag) not in _PERSON_TAGS:
            continue
        name = _first(el, ("reportingPersonName",))
        pct = _num(_first(el, _PCT_TAGS))
        shares = _num(_first(el, _SHARES_TAGS))
        if name is None and pct is None and shares is None:
            continue
        persons.append({"name": name, "cik": _cik(_first(el, _PERSON_CIK_TAGS)), "percent": pct,
                        "shares": shares, "type": _first(el, ("typeOfReportingPerson",))})
    if not persons:
        raise ParseScheduleError("nicio persoană raportoare")
    header = next((el for el in root.iter() if _local(el.tag) == "headerData"), root)
    return {
        "accession": accession, "form": sub, "kind": kind, "amendment": is_amendment(sub),
        "amendment_no": _num(_first(root, ("amendmentNo",))),
        "filing_date": _date(filing_date), "event_date": _date(_first(root, _EVENT_TAGS)),
        "issuer_cik": _cik(_first(root, _ISSUER_CIK_TAGS)), "cusip": _first(root, _CUSIP_TAGS),
        "class_title": _first(root, ("securitiesClassTitle",)),
        "filer_cik": _cik(_first(header, ("cik",))), "persons": persons, "format": "XML",
    }


_HDR_SECTION_RE = re.compile(r"^(SUBJECT COMPANY|FILED BY|FILER|REPORTING-OWNER)\s*:", re.M)
_HDR_NAME_RE = re.compile(r"COMPANY CONFORMED NAME:\s*(.+)")
_HDR_CIK_RE = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)")
_HDR_TYPE_RE = re.compile(r"CONFORMED SUBMISSION TYPE:\s*(.+)")
_HDR_PERIOD_RE = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})")


def parse_sgml_header(text, accession=None, filing_date=None, form=None):
    """Antetul SGML al unei depuneri (`<accession>.hdr.sgml`) → același dicționar, fără procente.

    Pentru depunerile vechi SC 13D / SC 13G (text sau HTML): numele și CIK-ul depunătorului („FILED BY”) și ale
    emitentului („SUBJECT COMPANY”). Procentul și acțiunile rămân None („N/A”)."""
    if isinstance(text, bytes):
        text = text.decode("latin-1", errors="replace")
    text = str(text or "")
    sub = (_HDR_TYPE_RE.search(text).group(1).strip() if _HDR_TYPE_RE.search(text) else None) or form
    kind = form_kind(sub)
    if kind is None:
        raise ParseScheduleError(f"antet fără Schedule 13D/13G (tip: {sub!r})")
    sections = {}
    marks = list(_HDR_SECTION_RE.finditer(text))
    for i, m in enumerate(marks):
        chunk = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        name, cik = _HDR_NAME_RE.search(chunk), _HDR_CIK_RE.search(chunk)
        sections.setdefault(m.group(1), (name.group(1).strip() if name else None, _cik(cik.group(1)) if cik else None))
    filer_name, filer_cik = sections.get("FILED BY") or sections.get("FILER") or (None, None)
    _subj_name, subj_cik = sections.get("SUBJECT COMPANY", (None, None))
    if filer_name is None and filer_cik is None:
        raise ParseScheduleError("antet fără „FILED BY”")
    period = _HDR_PERIOD_RE.search(text)
    return {
        "accession": accession, "form": sub, "kind": kind, "amendment": is_amendment(sub), "amendment_no": None,
        "filing_date": _date(filing_date), "event_date": _date(period.group(1)) if period else None,
        "issuer_cik": subj_cik, "cusip": None, "class_title": None, "filer_cik": filer_cik,
        "persons": [{"name": filer_name, "cik": filer_cik, "percent": None, "shares": None, "type": None}],
        "format": "text",
    }


# --------------------------------------------------------------------------- deținătorii actuali

def filing_summary(doc):
    """Procentul unei depuneri = maximul pe persoanele raportoare (o depunere în grup repetă aceleași acțiuni);
    numele = persoana cu procentul maxim (la egalitate, prima). Fără procente: prima persoană, procent None."""
    persons = doc.get("persons") or []
    with_pct = [p for p in persons if p.get("percent") is not None]
    top = max(with_pct, key=lambda p: p["percent"]) if with_pct else (persons[0] if persons else {})
    return {"name": top.get("name"), "percent": top.get("percent") if with_pct else None,
            "shares": top.get("shares") if with_pct else None, "n_persons": len(persons)}


def holder_key(doc):
    """Cheia depunătorului: CIK-ul din antet; altfel CIK-ul sau numele primei persoane (normalizat)."""
    if doc.get("filer_cik"):
        return f"CIK{doc['filer_cik']}"
    persons = doc.get("persons") or [{}]
    if persons[0].get("cik"):
        return f"CIK{persons[0]['cik']}"
    name = re.sub(r"[^A-Z0-9]", "", str(persons[0].get("name") or "").upper())
    return f"N:{name}" if name else None


def current_holders(docs, issuer_cik=None, threshold=THRESHOLD_PCT):
    """Ultima depunere a fiecărui depunător → tabelul `HOLDER_COLUMNS`, cei peste prag întâi, după procent.

    issuer_cik: dacă e dat, depunerile despre alt emitent (ex. compania e ea însăși depunător) se exclud; cele fără
    CIK de emitent rămân. Ordinea depunerilor aceluiași depunător: data depunerii, apoi accession."""
    issuer = _cik(issuer_cik) if issuer_cik is not None else None
    docs = [d for d in (docs or []) if not issuer or not d.get("issuer_cik") or d["issuer_cik"] == issuer]
    groups = {}
    for d in docs:
        k = holder_key(d)
        if k is not None:
            groups.setdefault(k, []).append(d)
    rows = []
    for k, ds in groups.items():
        ds = sorted(ds, key=lambda d: (d.get("filing_date") or pd.Timestamp.min, str(d.get("accession") or "")))
        last = ds[-1]
        s = filing_summary(last)
        kinds = [d["kind"] for d in ds]
        switch = None
        if len(set(kinds)) > 1:
            prev = next((x for x in reversed(kinds[:-1]) if x != last["kind"]), None)
            switch = f"{prev} → {last['kind']}" if prev else None
        pct = s["percent"]
        rows.append({
            "Cheie": k, "Detinator": s["name"], "CIK_depunator": last.get("filer_cik"), "Tip": last["kind"],
            "Trecere": switch, "Procent": pct, "Actiuni": s["shares"],
            "Sub_prag": (pct < threshold) if pct is not None else None,
            "Data_eveniment": last.get("event_date"), "Data_depunere": last.get("filing_date"),
            "Forma": last.get("form"), "Persoane": s["n_persons"], "Depuneri": len(ds),
            "Accession": last.get("accession"), "Format": last.get("format"),
        })
    out = pd.DataFrame(rows, columns=HOLDER_COLUMNS)
    if out.empty:
        return out
    out["_sub"] = out["Sub_prag"].map(lambda v: 1 if v is True else 0)
    out["_p"] = out["Procent"].astype(float).fillna(-1.0)
    out = out.sort_values(["_sub", "_p", "Data_depunere"], ascending=[True, False, False])
    return out.drop(columns=["_sub", "_p"]).reset_index(drop=True)


def holders_summary(holders, today, recent_days=90):
    """Cifrele cardului: câți depunători sunt peste prag (procent cunoscut ≥ 5% sau procent N/A), câți 13D,
    13D noi sau trecute din 13G în ultimele `recent_days` zile (după data depunerii), ultima depunere."""
    h = holders if holders is not None else pd.DataFrame(columns=HOLDER_COLUMNS)
    above = h[h["Sub_prag"].map(lambda v: v is not True)] if not h.empty else h
    start = pd.Timestamp(today) - pd.Timedelta(days=recent_days)
    d13 = above[above["Tip"].eq("13D")] if not above.empty else above
    recent_d = d13[d13["Data_depunere"].map(lambda d: d is not None and not pd.isna(d) and d > start)] \
        if not d13.empty else d13
    new_d = recent_d[recent_d["Forma"].map(lambda f: not is_amendment(f)) | recent_d["Trecere"].notna()] \
        if not recent_d.empty else recent_d
    dates = h["Data_depunere"].dropna() if not h.empty else pd.Series(dtype="datetime64[ns]")
    return {
        "peste_prag": int(len(above)),
        "fara_procent": int(above["Procent"].isna().sum()) if not above.empty else 0,
        "activisti": int(len(d13)),
        "pasivi": int(above["Tip"].eq("13G").sum()) if not above.empty else 0,
        "sub_prag": int(h["Sub_prag"].map(lambda v: v is True).sum()) if not h.empty else 0,
        "d13_noi": list(new_d["Detinator"]) if not new_d.empty else [],
        "ultima_depunere": dates.max() if len(dates) else None,
        "start": start,
    }
