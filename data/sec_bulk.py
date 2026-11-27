"""Arhivele SEC în bloc (`companyfacts.zip`, `submissions.zip`) pentru testul istoric (Faza 3c, decizia E6).

Se descarcă o singură dată în `tools/cache/sec/` (pe MacBook; mediul din cloud nu ajunge la SEC). Din arhive se
citesc doar CIK-urile cerute, fără dezarhivare totală; `companyfacts` se reduce cu `sec_xbrl.slim(..., toți
indicatorii)` și se păstrează pe CIK în cache (pickle). Fără dependențe noi (requests e deja în requirements).
**[de verificat la prima descărcare]** mărimea arhivelor (presupun ~1–2 GB `companyfacts.zip`, ~1 GB `submissions.zip`).
"""
import json
import logging
import pickle
import zipfile
from pathlib import Path

from analytics import xbrl_facts as xf
from data import sec_edgar, sec_xbrl

logger = logging.getLogger(__name__)

COMPANYFACTS_ZIP_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
SUBMISSIONS_ZIP_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
CHUNK = 1 << 20


def member_name(cik):
    """Numele fișierului din arhive: CIK0000320193.json."""
    return f"CIK{sec_edgar.cik10(cik)}.json"


def download_zip(url, path, user_agent, session=None, refresh=False):
    """Descarcă arhiva în `path` (în bucăți, prin `path.part`, redenumit la final). Nu descarcă din nou dacă există,
    decât cu `refresh`. Ridică `sec_edgar.EdgarError` la eșec. Întoarce calea."""
    import requests

    path = Path(path)
    if path.exists() and not refresh:
        return path
    ua = (user_agent or "").strip()
    if "@" not in ua:
        raise ValueError("SEC cere un User-Agent cu e-mail de contact.")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    sess = session or requests.Session()
    try:
        with sess.get(url, headers={"User-Agent": ua}, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                raise sec_edgar.EdgarError(f"{url}: HTTP {resp.status_code}")
            total = 0
            with open(part, "wb") as fh:
                for chunk in resp.iter_content(CHUNK):
                    if chunk:
                        fh.write(chunk)
                        total += len(chunk)
    except requests.RequestException as exc:
        raise sec_edgar.EdgarError(f"{url}: {type(exc).__name__}: {exc}") from exc
    part.replace(path)
    logger.info("descărcat %s: %.0f MB", path.name, total / 1e6)
    return path


def annual_filed_dates(cf):
    """Datele de depunere (ISO) ale rapoartelor 10-K / 10-K/A care apar în faptele JSON-ului, crescător."""
    dates = set()
    for concepts in ((cf or {}).get("facts") or {}).values():
        for node in concepts.values():
            for rows in (node.get("units") or {}).values():
                for r in rows:
                    if r.get("form") in xf.ANNUAL_FORMS and r.get("filed"):
                        dates.add(r["filed"])
    return sorted(dates)


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


class BulkSec:
    """Citire din `companyfacts.zip` și `submissions.zip`, cu cache pe CIK. `facts_zip` / `subs_zip` = căi."""

    def __init__(self, facts_zip, subs_zip=None, cache_dir=None, indicators=None):
        self._facts = zipfile.ZipFile(facts_zip) if facts_zip else None
        self._subs = zipfile.ZipFile(subs_zip) if subs_zip else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.indicators = tuple(xf.CONCEPTS) if indicators is None else tuple(indicators)
        self._names = set(self._facts.namelist()) if self._facts else set()
        self._sub_names = set(self._subs.namelist()) if self._subs else set()

    def close(self):
        for z in (self._facts, self._subs):
            if z is not None:
                z.close()

    def _raw_facts(self, cik):
        name = member_name(cik)
        if name not in self._names:
            return None
        with self._facts.open(name) as fh:
            return json.load(fh)

    def facts_cache_path(self, cik, symbol=None):
        return self.cache_dir / "facts" / f"{sec_edgar.cik10(cik)}_{symbol or ''}.pkl" if self.cache_dir else None

    def facts(self, cik, symbol=None):
        """`companyfacts` redus (toți indicatorii) pentru CIK, cu istoria CIK-urilor vechi din
        `sec_xbrl.PREDECESSORS` lipită (doar dacă numele se potrivește). None dacă CIK-ul nu e în arhivă."""
        path = self.facts_cache_path(cik, symbol)
        if path is not None and path.exists():
            return load_pickle(path)
        raw = self._raw_facts(cik)
        if raw is None:
            return None
        cf = sec_xbrl.slim(raw, self.indicators)
        used = []
        for old_cik, word in sec_xbrl.PREDECESSORS.get(str(symbol or "").upper(), []):
            old = self._raw_facts(old_cik)
            if old is None or word not in str(old.get("entityName") or "").upper():
                continue
            cf = sec_xbrl.merge_facts(cf, sec_xbrl.slim(old, self.indicators))
            used.append(old_cik)
        cf["predecessors"] = used
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump(cf, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return cf

    def submission(self, cik):
        """{sic, name, tickers} din `submissions.zip`, sau None."""
        name = member_name(cik)
        if self._subs is None or name not in self._sub_names:
            return None
        with self._subs.open(name) as fh:
            d = json.load(fh)
        return {"sic": d.get("sic"), "name": d.get("name"), "tickers": d.get("tickers") or []}


__all__ = ["COMPANYFACTS_ZIP_URL", "SUBMISSIONS_ZIP_URL", "member_name", "download_zip", "annual_filed_dates",
           "load_pickle", "BulkSec"]
