"""Fișierele „Analiză companii” ale lui Daniel (un Google Sheet per emitent BVB). Doar citire.

Funcții pure: primesc valorile foilor (listă de rânduri, ca `values_batch_get`) și întorc tabele.
Sursa sunt foile „Date Q” (cifre CUMULATE: T1, S1, 9 luni) și „Date Anuale”, rândurile cu cifre
brute. Indicatorii calculați în fișiere nu se folosesc: aplicația îi recalculează, iar fluxurile
(profit, cifră de afaceri, EBITDA) se aduc la 12 luni (TTM) din cumulate:
    TTM(cumulat k luni, anul Y) = An(Y−1) − Cumulat_k(Y−1) + Cumulat_k(Y).
Fișierele nu se modifică niciodată de aici.
"""
import re

import numpy as np
import pandas as pd

from data.bvb_sheet import normalize_label, sheet_number

# Rândurile cu cifre brute: eticheta din fișier (normalizată) -> cheia internă
RAW_ROWS = {
    "datorii totale": "liab",
    "cifra de afaceri": "rev",
    "profit net": "ni",
    "profit distribuit": "div_total",
    "ebitda": "ebitda",
    "rezultat operational": "ebit",
    "venituri": "income_total",
    "cheltuieli": "expense_total",
    "capital propriu (activ net)": "equity",
    "conturi la banci": "cash",
    "datorii termen scurt": "liab_st",
    "datorii termen lung": "liab_lt",
    "active totale": "assets",
    "active imobilizate": "assets_fixed",
    "active circulante (curente)": "assets_cur",
    "stocuri": "inventory",
    "creante": "receivables",
    "pret curent actiune": "price_manual",
    "nr. de actiuni emise": "shares",
}
FIELD_LABELS = {
    "liab": "Datorii totale", "rev": "Cifră de afaceri", "ni": "Profit net", "div_total": "Profit distribuit",
    "ebitda": "EBITDA", "ebit": "Rezultat operațional", "income_total": "Venituri", "expense_total": "Cheltuieli",
    "equity": "Capital propriu", "cash": "Conturi la bănci", "liab_st": "Datorii termen scurt",
    "liab_lt": "Datorii termen lung", "assets": "Active totale", "assets_fixed": "Active imobilizate",
    "assets_cur": "Active circulante", "inventory": "Stocuri", "receivables": "Creanțe",
    "price_manual": "Preț curent acțiune", "shares": "Nr. de acțiuni emise",
    "bilant": "Bilanț (capital + datorii față de active)", "active": "Active (imobilizate + circulante față de total)",
}
FLOWS = ("rev", "ni", "ebitda", "ebit", "income_total", "expense_total")
BALANCE_CHECKED = ("equity", "assets", "liab", "liab_st", "liab_lt", "inventory", "receivables",
                   "assets_fixed", "assets_cur", "cash")
# Băncile au alt șablon (depozite, active interbancare): nu se citesc cu acesta.
BANK_MARKERS = ("datorii la clienti (depozite)", "active interbancare", "depozite clienti",
                "venit bancar net", "credite si avansuri acordate clientelei")
# Termenul legal maxim de publicare: luni cumulate -> (lună, zi). Anualul: 30 aprilie, anul următor.
PUBLISH_DEADLINE = {3: (5, 15), 6: (9, 30), 9: (11, 15), 12: (4, 30)}
SHEET_Q, SHEET_A = "Date Q", "Date Anuale"


def ticker_from_title(title):
    """Simbolul BVB din titlul fișierului („DN Agrar" -> „DN"). None pentru fișiere care nu sunt emitenți."""
    title = str(title or "").strip()
    if not title or title[0] in "0Zz" and "-" in title[:3]:
        return None                                  # „0-Dashboard", „Z-Gol"
    token = title.split()[0].upper()
    return token if re.fullmatch(r"[A-Z0-9]{1,6}", token) else None


def parse_period(label):
    """„Q1 2021" -> (2021, 3); „Q3 2021" -> (2021, 9); „2021" -> (2021, 12); altceva -> None."""
    label = str(label or "").strip()
    m = re.fullmatch(r"Q([123])\s+(\d{4})", label)
    if m:
        return int(m.group(2)), 3 * int(m.group(1))
    m = re.fullmatch(r"(\d{4})", label)
    if m:
        return int(m.group(1)), 12
    return None


def period_label(year, months):
    return f"{year}" if months == 12 else {3: "T1", 6: "S1", 9: "9 luni"}[months] + f" {year}"


def is_bank_template(values):
    labels = {normalize_label(r[0]) for r in values or [] if r}
    return any(m in labels for m in BANK_MARKERS)


def parse_sheet(values):
    """{(an, luni): {cheie: valoare sau None}} și lista câmpurilor lipsă din foaie.

    Se ia prima apariție a fiecărei etichete (sub rândul 47 sunt comparațiile între perioade)."""
    if not values:
        return {}, sorted(set(RAW_ROWS.values()))
    header = values[0]
    cols = {c: parse_period(h) for c, h in enumerate(header) if c > 0 and parse_period(h)}
    out, seen = {}, set()
    for row in values[1:]:
        if not row:
            continue
        key = RAW_ROWS.get(normalize_label(row[0]))
        if not key or key in seen:
            continue
        seen.add(key)
        for c, per in cols.items():
            out.setdefault(per, {})[key] = sheet_number(row[c]) if c < len(row) else None
    return out, sorted(set(RAW_ROWS.values()) - seen)


def parse_company(q_values, a_values):
    """Tabelul perioadelor (cumulate și anuale) al unui emitent, sortat cronologic.

    O perioadă există doar dacă are active totale > 0 și cifră de afaceri (coloanele pregătite
    pentru viitor au 0 sau sunt goale). Coloana `published` = termenul legal maxim de publicare."""
    rows, missing = [], {}
    for name, values in ((SHEET_Q, q_values), (SHEET_A, a_values)):
        data, miss = parse_sheet(values)
        if miss:
            missing[name] = miss
        for (y, m), vals in data.items():
            rows.append({"year": y, "months": m, **{k: vals.get(k) for k in RAW_ROWS.values()}})
    df = pd.DataFrame(rows, columns=["year", "months"] + list(RAW_ROWS.values()))
    if df.empty:
        return df, missing
    df[list(RAW_ROWS.values())] = df[list(RAW_ROWS.values())].astype(float)
    df = df[(df["assets"].fillna(0) > 0) & df["rev"].notna()].copy()
    if df.empty:
        return df.reset_index(drop=True), missing
    df["period_end"] = pd.to_datetime(dict(year=df.year, month=df.months, day=1)) + pd.offsets.MonthEnd(0)
    df["published"] = [pd.Timestamp(y + 1, *PUBLISH_DEADLINE[12]) if m == 12 else pd.Timestamp(y, *PUBLISH_DEADLINE[m])
                       for y, m in zip(df.year, df.months)]
    df = df.drop_duplicates(["year", "months"], keep="first")
    return df.sort_values(["year", "months"]).reset_index(drop=True), missing


def validate(df):
    """Verificări de consistență. Întoarce (tabel curățat, listă de probleme).

    Valorile care pică sunt puse pe NaN în tabelul curățat (nu se corectează și nu se estimează):
      - EBITDA mai mic decât rezultatul operațional (EBITDA = EBIT + amortizări); EBITDA 0 = necompletat;
      - valori izolate de solduri, de peste 8× față de ambii vecini (de regulă lei vs mii de lei);
      - datorii totale sau pe termen scurt egale cu 0 (necompletate).
    Doar semnalate (rămân în calcul): bilanțul care nu se închide (±10%), suma activelor (±10%),
    numerarul izolat, cifra de afaceri cumulată care scade în cursul anului (reclasificări / retratări
    între rapoarte, cifre confirmate de Daniel).
    """
    df = df.copy().reset_index(drop=True)
    issues = []

    def flag(i, key, msg, excluded):
        issues.append({"year": int(df.at[i, "year"]), "months": int(df.at[i, "months"]), "field": key,
                       "message": msg, "excluded": excluded})

    # Datoriile totale și cele pe termen scurt nu pot fi 0 la o companie care funcționează: 0 = necompletat.
    for key in ("liab", "liab_st"):
        for i in df.index[df[key] == 0]:
            flag(i, key, f"{FIELD_LABELS[key]} = 0 (necompletat în fișier)", True)
            df.at[i, key] = np.nan

    for i in df.index:
        e, b = df.at[i, "ebitda"], df.at[i, "ebit"]
        if pd.notna(e) and e == 0:
            df.at[i, "ebitda"] = np.nan
        elif pd.notna(e) and pd.notna(b) and e < b - abs(b) * 0.005:
            flag(i, "ebitda", f"EBITDA {e:,.0f} < rezultat operațional {b:,.0f}", True)
            df.at[i, "ebitda"] = np.nan
        a, eq, li = df.at[i, "assets"], df.at[i, "equity"], df.at[i, "liab"]
        if all(pd.notna(x) for x in (a, eq, li)) and a > 0 and abs(eq + li - a) / a > 0.10:
            flag(i, "bilant", f"capital {eq:,.0f} + datorii {li:,.0f} ≠ active {a:,.0f} "
                              f"(abatere {abs(eq + li - a) / a:.0%})", False)
        fx, cu = df.at[i, "assets_fixed"], df.at[i, "assets_cur"]
        if all(pd.notna(x) for x in (a, fx, cu)) and a > 0 and abs(fx + cu - a) / a > 0.10:
            flag(i, "active", f"imobilizate {fx:,.0f} + circulante {cu:,.0f} ≠ active totale {a:,.0f}", False)

    for _, g in df.groupby("year"):
        prev, prev_i = None, None
        for i in g.sort_values("months").index:
            r = df.at[i, "rev"]
            if prev is not None and pd.notna(r) and r < prev * 0.99:
                # Cifre verificate de Daniel cu rapoartele (13 oct.): o scădere a cumulatului apare la
                # reclasificări sau retratări între rapoarte. Rămân în calcul, doar semnalate.
                flag(i, "rev", f"cifra de afaceri cumulată scade: {prev:,.0f} → {r:,.0f} "
                               "(trimestrul implicit e negativ: reclasificare sau retratare între rapoarte)", False)
            if pd.notna(r):
                prev, prev_i = r, i

    for key in BALANCE_CHECKED:
        s = df[key].astype(float)
        last_ok = np.nan
        for pos in range(len(df)):
            v = s.iloc[pos]
            if not (pd.notna(v) and v > 0):
                continue
            nxt = s.iloc[pos + 1] if pos + 1 < len(df) else np.nan
            rp = v / last_ok if pd.notna(last_ok) and last_ok > 0 else np.nan
            rn = v / nxt if pd.notna(nxt) and nxt > 0 else np.nan
            if pd.notna(rp) and pd.notna(rn):
                outlier = (rp > 8 and rn > 8) or (rp < 1 / 8 and rn < 1 / 8)
            elif pd.notna(rp) and pos == len(df) - 1:
                outlier = rp > 20 or rp < 1 / 20
            else:
                outlier = False
            if outlier:
                i = df.index[pos]
                excluded = key != "cash"
                flag(i, key, f"valoare izolată {v:,.0f} (anterior {last_ok:,.0f}, următor {nxt:,.0f})", excluded)
                if excluded:
                    df.at[i, key] = np.nan
            else:
                last_ok = v
    return df, issues


def add_ttm(df):
    """Coloanele `<flux>_ttm`: anualul rămâne anual; cumulatul k din anul Y devine
    An(Y−1) − Cumulat_k(Y−1) + Cumulat_k(Y). Lipsește oricare termen -> NaN."""
    df = df.copy()
    pos = {(int(y), int(m)): i for i, y, m in zip(df.index, df.year, df.months)}
    for f in FLOWS:
        vals = []
        for i, y, m in zip(df.index, df.year, df.months):
            if m == 12:
                vals.append(df.at[i, f])
                continue
            a_prev, c_prev = pos.get((y - 1, 12)), pos.get((y - 1, m))
            if a_prev is None or c_prev is None:
                vals.append(np.nan)
                continue
            parts = (df.at[a_prev, f], df.at[c_prev, f], df.at[i, f])
            vals.append(parts[0] - parts[1] + parts[2] if all(pd.notna(p) for p in parts) else np.nan)
        df[f + "_ttm"] = vals
    return df


def load_company(q_values, a_values):
    """Parsare + validare + TTM. Întoarce (tabel, probleme, câmpuri lipsă)."""
    df, missing = parse_company(q_values, a_values)
    if df.empty:
        return df, [], missing
    clean, issues = validate(df)
    return add_ttm(clean), issues, missing
