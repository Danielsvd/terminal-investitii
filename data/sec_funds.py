"""Administratorii urmăriți și încărcarea rapoartelor lor 13F din EDGAR (ultimele trimestre, cu amendamente).

Lista implicită: cei 20 de la Micro Alpha + Discovery Capital (decizia lui Daniel, 25.11.2026). CIK-urile marcate
„verificat” au fost confirmate pe paginile EDGAR ale depunerilor; celelalte sunt de confirmat: pagina compară numele de
la SEC cu cel așteptat și semnalează diferența. Cei patru „Larg” (cantitativi / multi-strategie) nu se descarcă
automat: au mii de poziții, nu reflectă convingere și sunt excluși din consens.

Fără Streamlit. Rețeaua trece prin `EdgarClient`.
"""
import logging
from urllib.parse import quote

import pandas as pd

from analytics import sec_13f
from data import sec_edgar

logger = logging.getLogger(__name__)

CONCENTRAT, LARG = "Concentrat", "Larg"

# CIK, fondul (cum apare la SEC), administratorul, stilul, clasa implicită, verificat pe EDGAR (25.11.2026)
MANAGERS = [
    ("1067983", "Berkshire Hathaway", "Warren Buffett / Greg Abel", "Value", CONCENTRAT, True),
    # ARK și Soros: „Larg” prin decizia lui Daniel (25.11.2026) — 181, respectiv 234 de poziții, primele 10 sub 50%.
    ("1697748", "ARK Investment Management", "Cathie Wood", "Growth", LARG, True),
    ("1709323", "Himalaya Capital Management", "Li Lu", "Deep value", CONCENTRAT, True),
    # Pershing: din 2025 rapoartele 13F se depun de Pershing Square Holdco, L.P. (CIK 2026053); vechiul CIK 1336528
    # (Capital Management) a raportat ultima dată 31.03.2026 (captura din 25.11.2026).
    ("2026053", "Pershing Square Holdco", "Bill Ackman", "Activist", CONCENTRAT, True),
    ("1536411", "Duquesne Family Office", "Stanley Druckenmiller", "Macro", CONCENTRAT, True),
    ("1656456", "Appaloosa LP", "David Tepper", "Distressed", CONCENTRAT, True),
    ("1040273", "Third Point", "Dan Loeb", "Activist", CONCENTRAT, False),
    ("1061768", "Baupost Group", "Seth Klarman", "Deep value", CONCENTRAT, False),
    ("1167483", "Tiger Global Management", "Chase Coleman", "Tech growth", CONCENTRAT, False),
    # Greenlight: CIK 1079114 a raportat ultima dată 31.12.2023; rapoartele curente se depun de entitatea de
    # administrare DME Capital Management, LP (CIK 1489933) — confirmat pe test1, 25.11.2026: 3,6 mld. $, 45 poziții.
    ("1489933", "Greenlight Capital (DME Capital Management)", "David Einhorn", "Value", CONCENTRAT, True),
    ("921669", "Icahn Carl C", "Carl Icahn", "Activist", CONCENTRAT, True),
    ("949509", "Oaktree Capital Management", "Howard Marks", "Distressed/Credit", CONCENTRAT, False),
    ("1029160", "Soros Fund Management", "George Soros", "Macro", LARG, True),
    ("1135730", "Coatue Management", "Philippe Laffont", "Tech growth", CONCENTRAT, True),
    ("1569205", "Fundsmith LLP", "Terry Smith", "Quality growth", CONCENTRAT, True),
    ("1389507", "Discovery Capital Management", "Rob Citrone", "Macro/Event", CONCENTRAT, True),
    ("1350694", "Bridgewater Associates", "Ray Dalio", "Macro", LARG, False),
    ("1423053", "Citadel Advisors", "Ken Griffin", "Quant/Multi-strategy", LARG, False),
    ("1037389", "Renaissance Technologies", "Jim Simons (estate)", "Quant", LARG, False),
    ("1603466", "Point72 Asset Management", "Steve Cohen", "Multi-strategy", LARG, False),
]
MANAGER_COLUMNS = ["CIK", "Fond", "Administrator", "Stil", "Clasa", "Verificat"]


def managers_frame():
    return pd.DataFrame(MANAGERS, columns=MANAGER_COLUMNS)


def fetch_filing(client, cik, accession, filing_date):
    """O depunere 13F: (copertă, tabel cu valoarea în dolari, probleme). Validarea se face pe unitatea raportată,
    apoi valoarea se convertește în dolari după data depunerii."""
    idx = client.get_json(sec_edgar.filing_index_url(cik, accession))
    cover_name, candidates = sec_edgar.info_table_candidates(idx)
    if cover_name is None:
        raise sec_13f.Parse13FError(f"{accession}: lipsește primary_doc.xml")
    cover = sec_13f.parse_cover(client.get(sec_edgar.filing_file_url(cik, accession, quote(cover_name))))
    info = None
    for name in candidates:
        body = client.get(sec_edgar.filing_file_url(cik, accession, quote(name)))
        if sec_13f.is_info_table(body):
            info = sec_13f.parse_info_table(body)
            break
    if info is None:
        info = pd.DataFrame(columns=sec_13f.INFO_COLUMNS)
        if (cover.get("entry_total") or 0) > 0:
            return cover, info, [f"{accession}: tabelul de informații nu a fost găsit"]
    problems = [f"{accession}: {p}" for p in sec_13f.validate(cover, info)]
    mult = sec_13f.value_multiplier(filing_date)
    if mult == 1.0 and sec_13f.detect_thousands(info, mult):
        mult = 1000.0
        cover["unit_note"] = (f"{accession}: valorile sunt raportate în mii de dolari (preț implicit median sub 1 $), "
                              "deși regula SEC cere dolari din 2023; convertite × 1000")
    info = info.copy()
    info["Valoare_raportata"] = info["Valoare_raportata"] * mult
    return cover, info, problems


def latest_quarters(filings, n=2):
    """Grupează depunerile 13F pe perioadă: pentru fiecare dintre ultimele `n` perioade, raportul de bază (ultimul
    13F-HR) și amendamentele depuse după el, în ordine cronologică."""
    by_period = {}
    for f in filings:
        p = f.get("report_date")
        if p:
            by_period.setdefault(p, []).append(f)
    out = []
    for period in sorted(by_period, reverse=True)[:n]:
        group = sorted(by_period[period], key=lambda f: (f["filing_date"] or "", f["accession"]))
        bases = [f for f in group if f["form"] == "13F-HR"]
        if not bases:
            continue
        base = bases[-1]
        amends = [f for f in group if f["form"] == "13F-HR/A" and (f["filing_date"] or "") >= (base["filing_date"] or "")]
        out.append({"period": period, "base": base, "amendments": amends})
    return out


def load_quarter(client, cik, q):
    """Un trimestru: raportul de bază cu amendamentele aplicate. Întoarce dict cu perioada, data depunerii, numele,
    tabelul (dolari) și problemele. Un raport de bază invalid lasă tabelul gol (nu se analizează date greșite)."""
    base = q["base"]
    cover, info, problems = fetch_filing(client, cik, base["accession"], base["filing_date"])
    notes = [cover["unit_note"]] if cover.get("unit_note") else []
    if problems:
        info = pd.DataFrame(columns=sec_13f.INFO_COLUMNS)
    last_filed = base["filing_date"]
    for am in q["amendments"]:
        try:
            acover, ainfo, aprob = fetch_filing(client, cik, am["accession"], am["filing_date"])
        except (sec_edgar.EdgarError, sec_13f.Parse13FError) as exc:
            problems.append(f"{am['accession']}: {exc}")
            continue
        if aprob:
            problems.extend(aprob)
            continue
        kind = acover.get("amendment_type")
        if kind not in ("RESTATEMENT", "NEW HOLDINGS"):
            problems.append(f"{am['accession']}: amendament fără tip ({kind!r}), ignorat")
            continue
        if acover.get("unit_note"):
            notes.append(acover["unit_note"])
        info = sec_13f.apply_amendment(info, ainfo, kind)
        last_filed = am["filing_date"]
    return {
        "period": q["period"], "filing_date": base["filing_date"], "last_filing_date": last_filed,
        "accession": base["accession"], "filer_name": cover.get("filer_name"), "info": info, "problems": problems,
        "notes": notes,
    }


def load_manager(client, cik, n=2):
    """Ultimele `n` trimestre ale unui administrator. Întoarce (numele de la SEC, listă de trimestre, probleme)."""
    sub = client.get_json(sec_edgar.submissions_url(cik))
    name = sub.get("name")
    filings = sec_edgar.recent_filings(sub)
    quarters, problems = [], []
    for q in latest_quarters(filings, n):
        try:
            quarters.append(load_quarter(client, cik, q))
        except (sec_edgar.EdgarError, sec_13f.Parse13FError) as exc:
            problems.append(f"{q['period']}: {exc}")
    if not filings:
        problems.append("nicio depunere 13F în lista recentă")
    return name, quarters, problems
