"""Schedule 13D / 13G de la SEC EDGAR pentru un emitent: lista depunerilor, documentele parsate.

Depunerile despre o companie apar în `submissions` ale companiei (ca „subject company”), chiar dacă le depune
altcineva; fișierele se citesc din folderul depunerii, pe CIK-ul companiei (ca la Form 4).
  - XML (SCHEDULE 13D/13G, din 18.12.2024): `primary_doc.xml` → procente și acțiuni.
  - Vechi (SC 13D/13G): doar antetul `<accession>.hdr.sgml` → numele depunătorului; procentul rămâne N/A.

Rețeaua trece doar prin `sec_edgar.EdgarClient`. Fără Streamlit. Cache-ul (TTL 12 h) îl pune aplicația; jobul de
pe server (faza 2c) va folosi aceleași funcții.
"""
import logging
from datetime import date, timedelta
from urllib.parse import quote

from analytics import insiders, schedule13
from data import sec_edgar

logger = logging.getLogger(__name__)

YEARS = 3               # fereastra: depunerile din ultimii 3 ani (13G-urile pasive se reactualizează cel puțin anual)
MAX_FILINGS = 60        # plafon pe cerere: ~8 s la 8 cereri/s


def schedule_filings(submissions, since, older=()):
    """Depunerile 13D/13G cu data depunerii ≥ `since` (ISO), cele mai noi primele, fără dubluri.
    `older`: paginile vechi deja descărcate (`page_filings` pe fiecare)."""
    rows = sec_edgar.recent_filings(submissions, forms=schedule13.FORMS)
    for page in older:
        rows += sec_edgar.page_filings(page, forms=schedule13.FORMS)
    seen, out = set(), []
    for f in sorted(rows, key=lambda r: (r["filing_date"] or "", r["accession"]), reverse=True):
        if (f["filing_date"] or "") >= since and f["accession"] not in seen:
            seen.add(f["accession"])
            out.append(f)
    return out


def older_pages_needed(submissions, since):
    """Numele paginilor vechi din `filings.files` care acoperă perioada de după `since` (au `filingTo` ≥ since)."""
    files = ((submissions or {}).get("filings") or {}).get("files") or []
    return [f["name"] for f in files if f.get("name") and (f.get("filingTo") or "") >= since]


def header_name(accession):
    return f"{accession}.hdr.sgml"


def load_schedules(client, cik, years=YEARS, today=None, max_filings=MAX_FILINGS):
    """Depunerile 13D/13G despre emitent din ultimii `years` ani, parsate.
    Întoarce (docs, probleme, trunchiat). O depunere care nu se descarcă sau nu se parsează intră la probleme."""
    today = today or date.today()
    since = (today - timedelta(days=365 * years)).isoformat()
    subs = client.get_json(sec_edgar.submissions_url(cik))
    older = []
    for name in older_pages_needed(subs, since):
        try:
            older.append(client.get_json(f"{sec_edgar.DATA_BASE}/submissions/{name}"))
        except sec_edgar.EdgarError as exc:
            logger.warning("13D/G pagina %s: %s", name, exc)
    filings = schedule_filings(subs, since, older)
    truncated = len(filings) > max_filings
    docs, problems = [], []
    for f in filings[:max_filings]:
        acc, form = f["accession"], f["form"]
        try:
            if form in schedule13.FORMS_XML:
                name = insiders.raw_xml_name(f["primary_document"]) or "primary_doc.xml"
                body = client.get(sec_edgar.filing_file_url(cik, acc, quote(name)))
                docs.append(schedule13.parse_schedule_xml(body, accession=acc, filing_date=f["filing_date"], form=form))
            else:
                body = client.get(sec_edgar.filing_file_url(cik, acc, header_name(acc)))
                docs.append(schedule13.parse_sgml_header(body, accession=acc, filing_date=f["filing_date"], form=form))
        except (sec_edgar.EdgarError, schedule13.ParseScheduleError) as exc:
            logger.warning("13D/G %s: %s", acc, exc)
            problems.append(f"{acc} ({form}): {exc}")
    return docs, problems, truncated
