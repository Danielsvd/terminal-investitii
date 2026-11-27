"""Jobul SEC de pe server (Faza 2c): ce simboluri urmărește, ce rânduri intră în jurnal, ce alerte pleacă pe Telegram.

Decizia lui Daniel (27.11.2026):
  - Simbolurile: acțiunile americane deținute acum (registrul `Tranzactii`) + cele din `watchlist`.
  - Alerte: (1) cumpărare pe piață (cod P) a unui insider ≥ 50.000 $ într-un raport Form 4; (2) un administrator
    concentrat (13F) deschide sau închide poziția într-unul din simboluri. Fără alertă pentru grup sau 13D.
  - Jurnal: foaia `SEC_insideri_jurnal` (toate tranzacțiile Form 4 citite, deduplicare pe accession); starea
    alertelor 13F într-un JSON local, pe server.
  - Prima rulare (fără stare) doar înregistrează: nu trimite alerte pentru ce exista deja.

Funcții pure: fără rețea, fără Sheets, fără Streamlit.
"""
import math

import pandas as pd

from analytics import sec_13f, sec_view

JOURNAL_SHEET = "SEC_insideri_jurnal"
JOURNAL_COLUMNS = ["Data_tranz", "Data_depunere", "Symbol", "CIK_emitent", "Insider", "Functie", "Cod", "Plan_10b5_1",
                   "Actiuni", "Pret", "Valoare_USD", "Detinere_dupa", "Accession"]
P_ALERT_MIN_USD = 50_000.0
TELEGRAM_MAX = 3500


def is_us_symbol(sym):
    """Simbol american: fără sufix de bursă (.RO, .DE), fără cotații speciale (^, =), fără cripto (-USD)."""
    s = str(sym or "").strip().upper()
    return bool(s) and "." not in s and "=" not in s and "^" not in s and not s.endswith("-USD") and " " not in s


def watched_symbols(held, watchlist):
    """Simbolurile urmărite: deținute (listă) ∪ watchlist (listă), doar americane, sortate, fără dubluri."""
    out = {str(s).strip().upper() for s in list(held or []) + list(watchlist or []) if is_us_symbol(s)}
    return sorted(out)


def known_accessions(values):
    """Accession-urile deja în jurnal (get_all_values al foii: antet + rânduri)."""
    if not values:
        return set()
    head = [str(h).strip() for h in values[0]]
    if "Accession" not in head:
        return set()
    i = head.index("Accession")
    return {str(r[i]).strip() for r in values[1:] if len(r) > i and str(r[i]).strip()}


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, pd.Timestamp):
        return "" if pd.isna(v) else v.strftime("%Y-%m-%d")
    if isinstance(v, bool):
        return "DA" if v else "NU"
    return v


def journal_rows(tx, symbol, cik):
    """Tranzacțiile (`insiders.transactions_frame`) → rândurile foii, în ordinea JOURNAL_COLUMNS. Lipsă = celulă goală."""
    rows = []
    if tx is None or tx.empty:
        return rows
    for _, r in tx.iterrows():
        rows.append([_cell(r["Data_tranz"]), _cell(r["Data_depunere"]), symbol, str(cik), _cell(r["Insider"]),
                     _cell(r["Functie"]), _cell(r["Cod"]), _cell(bool(r["Plan_10b5_1"])), _cell(r["Actiuni"]),
                     _cell(r["Pret"]), _cell(r["Valoare_USD"]), _cell(r["Detinere_dupa"]), _cell(r["Accession"])])
    return rows


def p_buy_alerts(tx, symbol, min_usd=P_ALERT_MIN_USD):
    """Cumpărările pe piață (P) ≥ `min_usd` pe raport și insider. Valoarea = suma tranzacțiilor P cu preț din raport
    (cele fără preț nu se adună: lipsa nu devine 0). Întoarce listă de dicționare, cele mai mari primele."""
    if tx is None or tx.empty:
        return []
    p = tx[tx["Cod"].eq("P")]
    if p.empty:
        return []
    out = []
    for (acc, ins), g in p.groupby(["Accession", "Insider"], sort=False):
        val = pd.to_numeric(g["Valoare_USD"], errors="coerce").sum(min_count=1)
        if pd.isna(val) or val < min_usd:
            continue
        sh = pd.to_numeric(g["Actiuni"], errors="coerce").sum(min_count=1)
        out.append({"symbol": symbol, "insider": ins, "functie": g["Functie"].iloc[0], "valoare": float(val),
                    "actiuni": None if pd.isna(sh) else float(sh), "data": g["Data_tranz"].max(),
                    "accession": acc})
    return sorted(out, key=lambda a: -a["valoare"])


def fund_move_events(symbols, cmap, holdings, moves, fund_of, period_of):
    """Intrările (Nouă) și ieșirile (Închisă) ale administratorilor concentrați pe simbolurile urmărite.

    holdings / moves: panourile din `sec_view.holdings_panel` / `moves_panel`, DOAR cu administratorii concentrați.
    Clasele de acțiuni se adună pe emitent (`sec_view.issuer_moves`). Întoarce listă de dicționare cu `key` unic
    (CIK|perioadă|emitent|acțiune), pentru deduplicare între rulări."""
    im = sec_view.issuer_moves(moves)
    if im.empty:
        return []
    im = im[im["Actiune"].isin([sec_13f.ACT_NEW, sec_13f.ACT_CLOSED])]
    out = []
    for sym in symbols:
        keys = sec_view.keys_for_symbol(sym, cmap, holdings, moves)
        for _, r in im[im["CUSIP6"].isin(keys)].iterrows():
            cik, period = str(r["CIK"]), period_of.get(str(r["CIK"]))
            out.append({"symbol": sym, "cik": cik, "fond": fund_of.get(cik, cik), "actiune": r["Actiune"],
                        "perioada": period, "key": f"{cik}|{period}|{r['CUSIP6']}|{r['Actiune']}"})
    return sorted(out, key=lambda e: (e["symbol"], e["fond"]))


def new_events(events, seen):
    """Evenimentele care nu sunt în `seen` (mulțimea cheilor deja alertate), fără dubluri."""
    out, keys = [], set(seen or ())
    for e in events:
        if e["key"] not in keys:
            keys.add(e["key"])
            out.append(e)
    return out


def _usd(v):
    if v is None:
        return "N/A"
    if abs(v) >= 1e6:
        num = f"{v / 1e6:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{num} mil. $"           # sufixul se adaugă după conversie: punctul din „mil.” nu devine virgulă
    return f"{v:,.0f} $".replace(",", ".")


def alert_message(today, p_alerts, fund_events, n_symbols, n_rows, problems, seed=False):
    """Mesajul Telegram al rulării (text simplu). None dacă nu e nimic de spus (fără alerte, fără probleme, nu e
    rularea inițială): jobul zilnic nu trimite „nimic nou”."""
    if not p_alerts and not fund_events and not problems and not seed:
        return None
    lines = [f"SEC — {today:%d.%m.%Y} ({n_symbols} simboluri SUA urmărite)"]
    if seed:
        lines.append(f"Rulare inițială: jurnalul are {n_rows} tranzacții; alertele pornesc de la următoarea rulare.")
    for a in p_alerts:
        d = a["data"]
        ds = d.strftime("%d.%m.%Y") if isinstance(d, pd.Timestamp) and not pd.isna(d) else "N/A"
        sh = f"{a['actiuni']:,.0f}".replace(",", ".") if a["actiuni"] is not None else "N/A"
        lines.append(f"CUMPĂRARE INSIDER {a['symbol']}: {a['insider']} ({a['functie'] or 'N/A'}) — {sh} acț., "
                     f"{_usd(a['valoare'])}, {ds}")
    for e in fund_events:
        verb = "a DESCHIS" if e["actiune"] == sec_13f.ACT_NEW else "a ÎNCHIS"
        lines.append(f"13F {e['symbol']}: {e['fond']} {verb} poziția (trimestrul {e['perioada'] or 'N/A'})")
    if problems:
        lines.append(f"Probleme: {len(problems)}")
        lines.extend(f"  {p}" for p in problems[:8])
    lines.append("Netestat: context, nu verdict.")
    text = "\n".join(lines)
    return text if len(text) <= TELEGRAM_MAX else text[:TELEGRAM_MAX - 20] + "\n… (trunchiat)"
