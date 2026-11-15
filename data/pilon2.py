"""Foile Pilon II din `portofoliu_db`: deținerile fondurilor de pensii (din rapoartele ASF Anexa 12).

- `Pilon2_detineri`: un rând pe (Date, Fond, ISIN), doar acțiuni. Scris de jobul de pe server
  (`tools/pilon2/run.py`), citit de aplicație. `Cantitate_derivata` = Valoare_RON / Pret_folosit (rapoartele
  nu conțin numărul de acțiuni), rotunjită; `Abatere_intreg` = cât de departe era câtul de un număr întreg.
- `ISIN_map`: ISIN → simbol Yahoo. Completată automat de job (Sursa „yahoo” / „negasit”) și corectabilă
  manual (Sursa „manual”).

Funcții pure: fără Streamlit, fără rețea. O celulă goală rămâne None, niciodată 0.
"""
import pandas as pd

from data.portfolio_sheet import cell_number

HOLDINGS_SHEET = "Pilon2_detineri"
MAP_SHEET = "ISIN_map"
HOLDINGS_COLUMNS = ["Date", "Fond", "ISIN", "Symbol", "Denumire", "Clasa", "Valoare_RON", "Pondere",
                    "Cantitate_derivata", "Pret_folosit", "Abatere_intreg", "Data_publicare"]
MAP_COLUMNS = ["ISIN", "Symbol", "Denumire", "Piata", "Sursa"]
_NUMERIC = ("Valoare_RON", "Pondere", "Cantitate_derivata", "Pret_folosit", "Abatere_intreg")


def _header_index(values, columns):
    header = [str(h).strip().lower() for h in values[0]]
    try:
        return [header.index(c.lower()) for c in columns]
    except ValueError:
        return None


def parse_isin_map(values):
    """Rândurile foii `ISIN_map` → {ISIN: {Symbol, Denumire, Piata, Sursa}}. Simbol gol rămâne ""."""
    if not values:
        return {}
    idx = _header_index(values, MAP_COLUMNS)
    if idx is None:
        return {}
    out = {}
    for row in values[1:]:
        cell = [str(row[i]).strip() if i < len(row) and row[i] is not None else "" for i in idx]
        isin = cell[0].upper()
        if isin:
            out[isin] = {"Symbol": cell[1].upper(), "Denumire": cell[2], "Piata": cell[3], "Sursa": cell[4]}
    return out


def parse_holdings(values):
    """Rândurile foii `Pilon2_detineri` → (DataFrame, probleme). Date și Data_publicare ca datetime."""
    empty = pd.DataFrame(columns=HOLDINGS_COLUMNS)
    if not values:
        return empty, []
    idx = _header_index(values, HOLDINGS_COLUMNS)
    if idx is None:
        return empty, [f"Foaia {HOLDINGS_SHEET} nu are antetul așteptat: {', '.join(HOLDINGS_COLUMNS)}."]
    rows, bad = [], 0
    for row in values[1:]:
        raw = {c: (row[i] if i < len(row) else None) for c, i in zip(HOLDINGS_COLUMNS, idx)}
        rec = {c: str(raw[c] or "").strip() for c in ("Fond", "ISIN", "Symbol", "Denumire", "Clasa")}
        rec["ISIN"], rec["Symbol"] = rec["ISIN"].upper(), rec["Symbol"].upper()
        rec["Date"] = pd.to_datetime(str(raw["Date"] or "").strip(), format="%Y-%m-%d", errors="coerce")
        rec["Data_publicare"] = pd.to_datetime(str(raw["Data_publicare"] or "").strip(), format="%Y-%m-%d",
                                               errors="coerce")
        for c in _NUMERIC:
            rec[c] = cell_number(raw[c])
        if pd.isna(rec["Date"]) or not rec["Fond"] or not rec["ISIN"]:
            if any(str(x).strip() for x in row):
                bad += 1
            continue
        rows.append(rec)
    problems = [f"{HOLDINGS_SHEET}: {bad} rânduri invalide sărite (lipsă Date AAAA-LL-ZZ, Fond sau ISIN)."] if bad else []
    if not rows:
        return empty, problems
    df = pd.DataFrame(rows)[HOLDINGS_COLUMNS].drop_duplicates(["Date", "Fond", "ISIN"], keep="last")
    for c in _NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["Date", "Fond", "ISIN"]).reset_index(drop=True), problems
