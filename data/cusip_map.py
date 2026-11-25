"""CUSIP → simbol bursier: OpenFIGI (Bloomberg, gratuit) și foaia `CUSIP_map` (cache persistent).

De ce: rapoartele 13F au doar CUSIP și numele emitentului, nu simbolul. Fără traducere nu se poate lega o poziție de
Analiză companie, de portofoliu sau de prețurile Yahoo. Micro Alpha arăta Meta ca „FB” (simbol vechi): luăm simbolul
curent de la OpenFIGI și îl salvăm cu data, ca să poată fi reverificat.

Limite OpenFIGI (documentația lor): fără cheie 25 de cereri/minut și 10 coduri/cerere; cu cheie 25 de cereri la 6 s
și 100 de coduri/cerere. Cheia se trimite în antetul `X-OPENFIGI-APIKEY`.

Fără Streamlit. Rețeaua trece prin `session`, ca testele să poată da o sesiune falsă.
"""
import logging
import time
from datetime import date

import requests

logger = logging.getLogger(__name__)

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
MAP_SHEET = "CUSIP_map"
MAP_COLUMNS = ["CUSIP", "Symbol", "Emitent", "Sursa", "Data"]
TIMEOUT_S = 30
# Bursele americane în codurile OpenFIGI; „US” e codul compozit, preferat.
US_EXCH = ("US", "UN", "UW", "UQ", "UA", "UR", "UP", "UV", "UF")


def batch_limits(api_key):
    """(coduri pe cerere, pauză între cereri în secunde) după cum avem sau nu cheie."""
    return (100, 6.0 / 25 + 0.05) if api_key else (10, 60.0 / 25 + 0.1)


def yahoo_symbol(ticker):
    """Simbolul OpenFIGI în forma Yahoo: 'BRK/B' → 'BRK-B'."""
    t = (ticker or "").strip().upper()
    return t.replace("/", "-").replace(" ", "-") or None


def pick_listing(data):
    """Din rezultatele OpenFIGI pentru un CUSIP: listarea compozită „US” (acțiune), altfel prima listare americană
    cu simbol, altfel None. Întoarce (simbol Yahoo, nume)."""
    data = data or []
    ranked = sorted(
        (d for d in data if d.get("ticker")),
        key=lambda d: (d.get("exchCode") != "US", d.get("exchCode") not in US_EXCH,
                       (d.get("marketSector") or "") != "Equity"),
    )
    for d in ranked:
        if d.get("exchCode") in US_EXCH:
            return yahoo_symbol(d.get("ticker")), d.get("name")
    return None, None


def openfigi_lookup(cusips, api_key=None, session=None, sleep=time.sleep):
    """Traduce o listă de CUSIP-uri. Întoarce (dict CUSIP → (simbol, nume)) și lista celor eșuate (fără răspuns,
    fără listare americană sau eroare de rețea). Un cod negăsit nu primește simbol inventat.

    Codurile care încep cu o literă sunt CINS (emitenți din afara SUA/Canadei, ex. Chubb H1467J104): dacă nu se găsesc
    ca ID_CUSIP, se reîncearcă o dată ca ID_CINS."""
    cusips = [c for c in dict.fromkeys(str(c).strip().upper() for c in cusips if c) if c]
    if not cusips:
        return {}, []
    session = session or requests.Session()
    state = {"first": True}
    found, failed = _lookup(cusips, "ID_CUSIP", api_key, session, sleep, state)
    cins = [c for c in failed if c[:1].isalpha()]
    if cins:
        more, still = _lookup(cins, "ID_CINS", api_key, session, sleep, state)
        found.update(more)
        failed = [c for c in failed if c not in more]
    return found, failed


def _lookup(codes, id_type, api_key, session, sleep, state):
    size, pause = batch_limits(api_key)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key
    found, failed = {}, []
    for i in range(0, len(codes), size):
        chunk = codes[i:i + size]
        if not state["first"]:
            sleep(pause)
        state["first"] = False
        payload = [{"idType": id_type, "idValue": c} for c in chunk]
        results = None
        for attempt in range(2):
            try:
                resp = session.post(OPENFIGI_URL, json=payload, headers=headers, timeout=TIMEOUT_S)
            except requests.RequestException as exc:
                logger.warning("OpenFIGI: %s (încercarea %d)", exc, attempt + 1)
                resp = None
            if resp is not None and resp.status_code == 200:
                try:
                    results = resp.json()
                except ValueError as exc:
                    logger.warning("OpenFIGI: JSON invalid (%s)", exc)
                break
            if resp is not None and resp.status_code not in (429, 500, 502, 503, 504):
                logger.warning("OpenFIGI: HTTP %s", resp.status_code)
                break
            sleep(6.0 if api_key else 60.0)
        if not isinstance(results, list) or len(results) != len(chunk):
            failed.extend(chunk)
            continue
        for c, r in zip(chunk, results):
            sym, name = pick_listing((r or {}).get("data"))
            if sym:
                found[c] = (sym, name)
            else:
                failed.append(c)
    return found, failed


def parse_map(values):
    """Foaia `CUSIP_map` (get_all_values) → dict CUSIP → simbol. Rândurile fără simbol se ignoră."""
    if not values:
        return {}
    head = [h.strip() for h in values[0]]
    try:
        ic, isym = head.index("CUSIP"), head.index("Symbol")
    except ValueError:
        logger.warning("CUSIP_map: antet neașteptat %s", head)
        return {}
    out = {}
    for row in values[1:]:
        c = row[ic].strip().upper() if len(row) > ic else ""
        s = row[isym].strip().upper() if len(row) > isym else ""
        if c and s:
            out[c] = s
    return out


def map_rows(found, source="openfigi", today=None):
    """Rândurile de adăugat în `CUSIP_map` pentru traducerile noi, în ordinea `MAP_COLUMNS`."""
    d = (today or date.today()).isoformat()
    return [[c, sym, name or "", source, d] for c, (sym, name) in sorted(found.items())]
