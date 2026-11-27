"""Client SEC EDGAR: cereri cu User-Agent obligatoriu, limită de ritm, reîncercări; adrese și liste de depuneri.

Reguli SEC (fair access): fiecare cerere trimite un User-Agent cu nume și e-mail de contact; cel mult 10 cereri pe
secundă de la aceeași adresă IP. Clientul se oprește la 8/s, ca să rămână loc. La 429/503 așteaptă și reîncearcă.

Folosit de jobul de pe server (`tools/sec/`) și, la cerere, de aplicație (Form 4, XBRL — fazele 2–3).
Fără Streamlit. Rețeaua trece doar prin `EdgarClient`, ca testele să poată da o sesiune falsă.
"""
import json
import logging
import time

import requests

logger = logging.getLogger(__name__)

DATA_BASE = "https://data.sec.gov"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
MAX_RPS = 8                 # SEC permite 10/s; păstrăm o marjă
TIMEOUT_S = 30
RETRY_STATUSES = (429, 500, 502, 503, 504)
FORMS_13F = ("13F-HR", "13F-HR/A")


class EdgarError(RuntimeError):
    """Cerere eșuată după reîncercări sau răspuns neașteptat de la EDGAR."""


def cik10(cik):
    """CIK cu 10 cifre, cu zerouri în față (formatul din adresele data.sec.gov)."""
    s = str(cik).strip()
    if not s.isdigit():
        raise ValueError(f"CIK invalid: {cik!r}")
    return s.zfill(10)


def accession_nodash(accession):
    return str(accession).replace("-", "")


def submissions_url(cik):
    return f"{DATA_BASE}/submissions/CIK{cik10(cik)}.json"


def filing_index_url(cik, accession):
    """Indexul JSON al unei depuneri: lista fișierelor din folderul ei."""
    return f"{ARCHIVES_BASE}/{int(cik10(cik))}/{accession_nodash(accession)}/index.json"


def filing_file_url(cik, accession, name):
    return f"{ARCHIVES_BASE}/{int(cik10(cik))}/{accession_nodash(accession)}/{name}"


class EdgarClient:
    """Cereri GET către EDGAR, în ritm controlat. `session`, `sleep` și `clock` se pot înlocui în teste."""

    def __init__(self, user_agent, max_rps=MAX_RPS, session=None, sleep=time.sleep, clock=time.monotonic,
                 retries=3, backoff_s=2.0, timeout_s=TIMEOUT_S):
        ua = (user_agent or "").strip()
        if "@" not in ua:
            raise ValueError("SEC cere un User-Agent cu e-mail de contact (ex. 'Terminal Investitii nume@domeniu.ro').")
        self.user_agent = ua
        self.min_interval = 1.0 / max_rps
        self.session = session or requests.Session()
        self.sleep = sleep
        self.clock = clock
        self.retries = retries
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self._last = None

    def _throttle(self):
        now = self.clock()
        if self._last is not None:
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                self.sleep(wait)
                now = self.clock()
        self._last = now

    def get(self, url):
        """Corpul răspunsului (bytes). Ridică EdgarError după `retries` reîncercări sau la 404."""
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        last_err = None
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, headers=headers, timeout=self.timeout_s)
            except requests.RequestException as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning("EDGAR %s: %s (încercarea %d)", url, last_err, attempt + 1)
            else:
                if resp.status_code == 200:
                    return resp.content
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRY_STATUSES:
                    break
                logger.warning("EDGAR %s: %s (încercarea %d)", url, last_err, attempt + 1)
            if attempt < self.retries:
                self.sleep(self.backoff_s * (2 ** attempt))
        raise EdgarError(f"{url}: {last_err}")

    def get_json(self, url):
        body = self.get(url)
        try:
            return json.loads(body)
        except ValueError as exc:
            raise EdgarError(f"{url}: JSON invalid ({exc})") from exc


def recent_filings(submissions, forms=FORMS_13F):
    """Depunerile din `filings.recent` (format pe coloane) ca listă de dicționare, cele mai noi primele.

    Chei: accession, form, filing_date, report_date, primary_document. `recent` acoperă cel puțin ultimul an sau
    ultimele 1.000 de depuneri; paginile mai vechi sunt în `filings.files` (vezi `older_pages`)."""
    rec = ((submissions or {}).get("filings") or {}).get("recent") or {}
    return _columnar(rec, forms)


def older_pages(submissions):
    """Numele fișierelor JSON cu depuneri mai vechi (ex. 'CIK0001067983-submissions-001.json')."""
    files = ((submissions or {}).get("filings") or {}).get("files") or []
    return [f.get("name") for f in files if f.get("name")]


def page_filings(page, forms=FORMS_13F):
    """Aceeași extragere pentru o pagină veche (are coloanele direct la rădăcină)."""
    return _columnar(page or {}, forms)


def _columnar(rec, forms):
    acc = rec.get("accessionNumber") or []
    out = []
    for i, a in enumerate(acc):
        form = _at(rec, "form", i)
        if forms and form not in forms:
            continue
        out.append({
            "accession": a,
            "form": form,
            "filing_date": _at(rec, "filingDate", i),
            "report_date": _at(rec, "reportDate", i) or None,
            "primary_document": _at(rec, "primaryDocument", i),
        })
    out.sort(key=lambda r: (r["filing_date"] or "", r["accession"]), reverse=True)
    return out


def _at(rec, key, i):
    col = rec.get(key) or []
    return col[i] if i < len(col) else None


def info_table_candidates(index_json):
    """Din indexul unei depuneri 13F: (coperta, [fișiere XML candidate pentru tabelul de informații]).

    Coperta e `primary_doc.xml`. Tabelul de informații are nume liber (ex. 'infotable.xml', '50240.xml'), deci se
    întorc toate celelalte XML-uri; cel corect se recunoaște după rădăcina `informationTable` (`analytics.sec_13f`)."""
    items = ((index_json or {}).get("directory") or {}).get("item") or []
    names = [it.get("name", "") for it in items]
    xmls = [n for n in names if n.lower().endswith(".xml")]
    cover = next((n for n in xmls if n.lower() == "primary_doc.xml"), None)
    others = [n for n in xmls if n.lower() != "primary_doc.xml"]
    return cover, others
