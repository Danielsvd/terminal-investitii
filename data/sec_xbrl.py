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


# Reorganizări care au mutat emitentul pe un CIK nou (societate-mamă nouă): istoria rămâne pe CIK-ul vechi.
# simbol → [(CIK vechi, cuvânt obligatoriu în entityName)] — fără potrivirea numelui, CIK-ul vechi NU se lipește.
# [de verificat cu proba de pe server] XOM: CIK nou 2115436 din 2026, doar 10-Q (presupunere: holding nou).
PREDECESSORS = {
    "XOM": [("34088", "EXXON")],
    "GOOGL": [("1288776", "GOOGLE")],
    "GOOG": [("1288776", "GOOGLE")],
    "AVGO": [("1649338", "BROADCOM"), ("1441634", "AVAGO")],
}


def merge_facts(current, older):
    """Lipește valorile din JSON-ul `older` (CIK-ul vechi) înaintea celor din `current`, pe aceeași etichetă și unitate.
    Perioadele comune se rezolvă mai departe după data depunerii (ultima versiune câștigă)."""
    out = {"cik": current.get("cik"), "entityName": current.get("entityName"), "facts": {}}
    for src in (older, current):
        for tax, concepts in ((src or {}).get("facts") or {}).items():
            dst = out["facts"].setdefault(tax, {})
            for name, node in concepts.items():
                units = dst.setdefault(name, {"units": {}})["units"]
                for unit, rows in (node.get("units") or {}).items():
                    units.setdefault(unit, []).extend(rows)
    return out


def load_for_symbol(client, symbol):
    """(cik, companyfacts redus) pentru simbolul american; (None, None) dacă simbolul nu e în lista SEC.
    La reorganizările din `PREDECESSORS`, istoria CIK-ului vechi se lipește (doar dacă numele se potrivește);
    `cf["predecessors"]` = CIK-urile lipite."""
    cik = sec_insiders.ticker_cik(client, symbol)
    if cik is None:
        return None, None
    return cik, load_with_predecessors(client, cik, symbol)


def load_with_predecessors(client, cik, symbol):
    """`companyfacts` redus al CIK-ului, cu istoria CIK-urilor vechi din `PREDECESSORS` lipită (vezi load_for_symbol)."""
    cf = load_companyfacts(client, cik)
    used = []
    for old_cik, word in PREDECESSORS.get(str(symbol).upper(), []):
        try:
            old = client.get_json(companyfacts_url(old_cik))
        except sec_edgar.EdgarError:
            continue
        if word not in str(old.get("entityName") or "").upper():
            continue
        cf = merge_facts(cf, slim(old))
        used.append(old_cik)
    cf["predecessors"] = used
    return cf
