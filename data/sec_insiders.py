"""Form 4 de la SEC EDGAR pentru un emitent: simbol → CIK, lista depunerilor recente, XML-urile brute parsate.

Rețeaua trece doar prin `sec_edgar.EdgarClient` (ritm ≤ 8 cereri/s, User-Agent cu e-mail). Fără Streamlit.
Cache-ul (TTL 12 h) îl pune aplicația; jobul de pe server (faza 2c) va folosi aceleași funcții.
"""
import logging
from datetime import date, timedelta
from urllib.parse import quote

from analytics import insiders
from data import sec_edgar

logger = logging.getLogger(__name__)

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FORMS_4 = ("4", "4/A")
MAX_FILINGS = 80        # plafon pe cerere: ~10 s la 8 cereri/s; emitenții mari depun 20–60 de Form 4 pe trimestru


def ticker_cik(client, symbol):
    """CIK-ul emitentului pentru simbolul american, din lista SEC; None dacă nu există."""
    return insiders.cik_for_ticker(client.get_json(COMPANY_TICKERS_URL), symbol)


def form4_filings(submissions, since, max_filings=MAX_FILINGS):
    """Depunerile Form 4 / 4/A cu data depunerii ≥ `since` (ISO), cele mai noi primele, cel mult `max_filings`.
    Întoarce (listă, trunchiat: bool)."""
    rows = [f for f in sec_edgar.recent_filings(submissions, forms=FORMS_4) if (f["filing_date"] or "") >= since]
    return rows[:max_filings], len(rows) > max_filings


def load_form4(client, cik, days=insiders.WINDOW_DAYS + 5, today=None, max_filings=MAX_FILINGS):
    """Form 4-urile emitentului depuse în ultimele `days` zile, parsate. Întoarce (docs, probleme, trunchiat).
    O depunere care nu se descarcă sau nu se parsează intră la probleme, nu oprește restul."""
    today = today or date.today()
    since = (today - timedelta(days=days)).isoformat()
    subs = client.get_json(sec_edgar.submissions_url(cik))
    filings, truncated = form4_filings(subs, since, max_filings)
    docs, problems = [], []
    for f in filings:
        name = insiders.raw_xml_name(f["primary_document"])
        try:
            body = client.get(sec_edgar.filing_file_url(cik, f["accession"], quote(name)))
            docs.append(insiders.parse_form4(body, accession=f["accession"], filing_date=f["filing_date"]))
        except (sec_edgar.EdgarError, insiders.ParseForm4Error) as exc:
            logger.warning("Form 4 %s: %s", f["accession"], exc)
            problems.append(f"{f['accession']}: {exc}")
    return docs, problems, truncated
