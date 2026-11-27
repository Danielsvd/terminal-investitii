"""Descărcarea situațiilor XBRL de la SEC (`companyfacts`) pentru un emitent. Faza 3 SEC.

Rețeaua trece doar prin `sec_edgar.EdgarClient` (User-Agent cu e-mail, ≤ 8 cereri/s, reîncercări). Fără Streamlit:
cache-ul (TTL 24 h) îl pune aplicația. `slim()` păstrează doar etichetele folosite (`xbrl_facts.CONCEPTS`), ca JSON-ul
mare al companiilor vechi (câțiva MB) să nu stea întreg în memorie.
"""
from analytics import xbrl_facts
from data import sec_edgar, sec_insiders


def companyfacts_url(cik):
    return f"{sec_edgar.DATA_BASE}/api/xbrl/companyfacts/CIK{sec_edgar.cik10(cik)}.json"


def slim(cf):
    """Copie a JSON-ului `companyfacts` cu doar etichetele din `xbrl_facts.CONCEPTS` și marcajul taxonomiilor prezente
    (pentru `filer_kind`). Restul câmpurilor de identificare (cik, entityName) se păstrează."""
    facts = (cf or {}).get("facts") or {}
    keep = {}
    for _, (_, _, concepts) in xbrl_facts.CONCEPTS.items():
        for tax, name in concepts:
            node = (facts.get(tax) or {}).get(name)
            if node is not None:
                keep.setdefault(tax, {})[name] = {"units": node.get("units") or {}}
    if "ifrs-full" in facts:
        keep.setdefault("ifrs-full", {})        # marcaj: emitent IFRS (20-F), fără date încărcate
    if "us-gaap" in facts:
        keep.setdefault("us-gaap", {})
    return {"cik": cf.get("cik"), "entityName": cf.get("entityName"), "facts": keep}


def load_companyfacts(client, cik):
    """JSON-ul `companyfacts` redus (`slim`). Ridică `sec_edgar.EdgarError` (ex. 404 = emitent fără XBRL)."""
    return slim(client.get_json(companyfacts_url(cik)))


def load_for_symbol(client, symbol):
    """(cik, companyfacts redus) pentru simbolul american; (None, None) dacă simbolul nu e în lista SEC."""
    cik = sec_insiders.ticker_cik(client, symbol)
    if cik is None:
        return None, None
    return cik, load_companyfacts(client, cik)
