"""Deținerile fondurilor de pensii Pilon II la BVB: panou pe (fond, dată, simbol), fluxuri, tabele.

Intrare: DataFrame-ul din `data.pilon2.parse_holdings` (doar acțiuni). Cantitățile sunt **derivate**
(valoare / preț de închidere la data raportului), deci un proxy. Rapoartele apar cu ~3,5 luni întârziere:
fluxurile descriu ce au făcut fondurile atunci, nu acum.

Reguli:
  - Un fond care a publicat raportul la o dată și nu are simbolul în el deține 0 acțiuni la acea dată
    (rapoartele sunt complete: jobul le respinge pe cele în care suma acțiunilor nu iese).
  - O cantitate necunoscută (fără preț) rămâne NaN, iar fluxurile care depind de ea sunt NaN — niciodată 0.
  - Fluxul în lei = Δ acțiuni × prețul de la data raportului (scoate efectul de preț din variația valorii).

Funcții pure: fără Streamlit, fără rețea.
"""
import numpy as np
import pandas as pd

# Ordinea fixă a fondurilor (după active, 2026) și culorile lor: paleta categorială validată (dataviz,
# validate_palette.js: ALL PASS pe #0E1117 și pe alb). Culoarea urmează fondul, nu rangul.
FUND_ORDER = ["NN", "AZT Viitorul Tău", "Metropolitan Life", "Vital", "Aripi", "BCR", "BT Pilon 2"]
FUND_COLORS = dict(zip(FUND_ORDER, ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9"]))
OTHER_COLOR = "#8B949E"
MIN_OBS = 20            # sub ~20 de observații: fără verdict (regula proiectului)


def _pos(info, key):
    try:
        v = float((info or {}).get(key))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def shares_outstanding(info):
    """(număr de acțiuni, sursa) din `info` Yahoo; (None, None) dacă lipsește.

    Ordinea: `sharesOutstanding`, `impliedSharesOutstanding`, apoi capitalizare / preț (proxy). La BVB, Yahoo trimite
    uneori `info` fără `sharesOutstanding` (16.11.2026, TLV: 23,59% la o încărcare, N/A la următoarea)."""
    for key in ("sharesOutstanding", "impliedSharesOutstanding"):
        v = _pos(info, key)
        if v:
            return v, key
    cap = _pos(info, "marketCap")
    px = _pos(info, "currentPrice") or _pos(info, "regularMarketPrice")
    if cap and px:
        return cap / px, "capitalizare / preț (proxy)"
    return None, None


def fund_color(name):
    return FUND_COLORS.get(name, OTHER_COLOR)


def fund_sort_key(name):
    return (FUND_ORDER.index(name) if name in FUND_ORDER else len(FUND_ORDER), str(name))


def build_panel(df):
    """Panoul (Fond, Date, Symbol) pentru acțiunile BVB, cu 0 unde fondul a raportat și nu deține simbolul.

    Coloane: Fond, Date, Symbol, Detinut, Cantitate, Valoare, Pondere, Pret (al rândului), Pret_data
    (prețul simbolului la acea dată, din orice fond), dQ (față de raportul anterior al fondului), Flux_RON."""
    cols = ["Fond", "Date", "Symbol", "Detinut", "Cantitate", "Valoare", "Pondere", "Pret", "Pret_data", "dQ", "Flux_RON"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    bvb = df[df["Symbol"].fillna("").str.endswith(".RO")]
    if bvb.empty:
        return pd.DataFrame(columns=cols)
    agg = (bvb.groupby(["Fond", "Date", "Symbol"], as_index=False)
              .agg(Cantitate=("Cantitate_derivata", lambda s: s.sum(min_count=len(s))),
                   Valoare=("Valoare_RON", lambda s: s.sum(min_count=1)),
                   Pondere=("Pondere", lambda s: s.sum(min_count=1)),
                   Pret=("Pret_folosit", "max")))
    agg["Detinut"] = True
    pairs = df[["Fond", "Date"]].drop_duplicates()
    syms = pd.DataFrame({"Symbol": sorted(bvb["Symbol"].unique())})
    grid = pairs.merge(syms, how="cross").merge(agg, on=["Fond", "Date", "Symbol"], how="left")
    absent = grid["Detinut"].isna()
    grid["Detinut"] = ~absent
    for c in ("Cantitate", "Valoare", "Pondere"):
        grid.loc[absent, c] = 0.0
    grid["Pret_data"] = grid.groupby(["Symbol", "Date"])["Pret"].transform("max")
    grid = grid.sort_values(["Symbol", "Fond", "Date"]).reset_index(drop=True)
    grid["dQ"] = grid.groupby(["Symbol", "Fond"])["Cantitate"].diff()
    grid["Flux_RON"] = np.where(grid["dQ"] == 0, 0.0, grid["dQ"] * grid["Pret_data"])   # neschimbat = 0 lei
    return grid[cols]


def company_panel(panel, symbol):
    """Rândurile simbolului; fondurile care nu l-au deținut niciodată sunt scoase."""
    p = panel[panel["Symbol"] == str(symbol).upper()]
    ever = p.groupby("Fond")["Detinut"].transform("any")
    return p[ever].reset_index(drop=True)


def _at_month(g, when):
    """Rândul fondului din luna lui `when` (rapoartele sunt la sfârșit de lună), sau None."""
    m = g[(g["Date"].dt.year == when.year) & (g["Date"].dt.month == when.month)]
    return None if m.empty else m.iloc[-1]


def _label(q, prev):
    if pd.isna(q) or pd.isna(prev):
        return "N/A"
    if prev > 0 and q == 0:
        return "ieșit complet"
    if prev == 0 and q > 0:
        return "nou"
    if q > prev:
        return "intrare"
    if q < prev:
        return "ieșire"
    return "neschimbat"


def fund_table(cpanel, shares_out=None):
    """Un rând pe fond, la ultimul lui raport: acțiuni, % din capital, valoare, Δ 1/3/12 luni, etichetă.

    Fondurile care nu dețin simbolul nici acum, nici la raportul anterior nu apar."""
    rows = []
    for fond, g in cpanel.groupby("Fond"):
        g = g.sort_values("Date")
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        q, d = last["Cantitate"], last["Date"]
        q_prev = prev["Cantitate"] if prev is not None else np.nan
        if q == 0 and (prev is None or q_prev == 0):
            continue
        r3 = _at_month(g, d - pd.DateOffset(months=3))
        r12 = _at_month(g, d - pd.DateOffset(months=12))
        rows.append({
            "Fond": fond, "Data raport": d, "Acțiuni": q,
            "% din capital": (q / shares_out * 100) if (shares_out and pd.notna(q)) else np.nan,
            "Valoare (lei)": last["Valoare"], "Pondere în fond (%)": last["Pondere"],
            "Δ față de raportul anterior": q - q_prev,
            "Δ 3 luni": (q - r3["Cantitate"]) if r3 is not None else np.nan,
            "Δ 12 luni": (q - r12["Cantitate"]) if r12 is not None else np.nan,
            "Etichetă": _label(q, q_prev),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Fond", key=lambda s: s.map(fund_sort_key)).reset_index(drop=True)


def totals(cpanel):
    """Pe dată: acțiuni totale, valoare totală, câte fonduri dețin, câte au raportat.

    Fondurile raportează la date diferite: totalul unei date cuprinde doar fondurile cu raport la acea dată
    (coloana „Fonduri raportate” o spune)."""
    if cpanel.empty:
        return pd.DataFrame(columns=["Date", "Acțiuni", "Valoare", "Fonduri care dețin", "Fonduri raportate", "Flux_RON", "dQ"])
    g = cpanel.groupby("Date")
    out = pd.DataFrame({
        "Acțiuni": g["Cantitate"].apply(lambda s: s.sum(min_count=len(s))),
        "Valoare": g["Valoare"].sum(min_count=1),
        "Fonduri care dețin": g["Cantitate"].apply(lambda s: int((s > 0).sum())),
        "Fonduri raportate": g["Fond"].nunique(),
        "dQ": g["dQ"].apply(lambda s: s.sum(min_count=len(s)) if s.notna().any() else np.nan),
        "Flux_RON": g["Flux_RON"].apply(lambda s: s.sum(min_count=len(s)) if s.notna().any() else np.nan),
    }).reset_index()
    return out.sort_values("Date").reset_index(drop=True)


def implied_cost(cpanel, since=None):
    """Prețul mediu al cumpărărilor nete (proxy): Σ(dQ>0 × preț) / Σ dQ>0. Întoarce (preț sau None, nr. observații)."""
    p = cpanel[(cpanel["dQ"] > 0) & cpanel["Pret_data"].notna()]
    if since is not None:
        p = p[p["Date"] >= since]
    if p.empty:
        return None, 0
    return float((p["dQ"] * p["Pret_data"]).sum() / p["dQ"].sum()), int(len(p))


def days_to_liquidate(total_shares, avg_daily_volume):
    """Zile de bursă necesare ca să se vândă toată deținerea la volumul mediu zilnic (proxy de risc)."""
    if total_shares is None or avg_daily_volume is None or pd.isna(total_shares) or not avg_daily_volume > 0:
        return None
    return float(total_shares / avg_daily_volume)


def reference_date(panel, lookback=3):
    """Cea mai recentă dată raportată de cele mai multe fonduri (dintre ultimele `lookback` date)."""
    if panel.empty:
        return None
    counts = panel.groupby("Date")["Fond"].nunique().sort_index().tail(lookback)
    return counts[counts == counts.max()].index.max()


def market_flows(panel, when=None):
    """Fluxurile nete pe simbol la o dată (toate fondurile cu raport atunci): Flux_RON, dQ, cumpărători, vânzători."""
    when = when if when is not None else reference_date(panel)
    p = panel[(panel["Date"] == when) & panel["dQ"].notna()]
    if p.empty:
        return pd.DataFrame(columns=["Symbol", "Flux_RON", "dQ", "Cumpărători", "Vânzători", "Valoare"])
    g = p.groupby("Symbol")
    out = pd.DataFrame({
        "Flux_RON": g["Flux_RON"].sum(min_count=1),
        "dQ": g["dQ"].sum(min_count=1),
        "Cumpărători": g["dQ"].apply(lambda s: int((s > 0).sum())),
        "Vânzători": g["dQ"].apply(lambda s: int((s < 0).sum())),
        "Valoare": g["Valoare"].sum(min_count=1),
    }).reset_index()
    return out.sort_values("Flux_RON", ascending=False, na_position="last").reset_index(drop=True)


def flow_heatmap(panel, months=12, top=25):
    """Matrice simbol × dată cu fluxul net în lei, pentru cele mai mari `top` dețineri (la ultima dată)."""
    if panel.empty:
        return pd.DataFrame()
    dates = sorted(panel["Date"].unique())[-months:]
    last = reference_date(panel)
    size = panel[panel["Date"] == last].groupby("Symbol")["Valoare"].sum().sort_values(ascending=False)
    syms = list(size.head(top).index)
    p = panel[panel["Date"].isin(dates) & panel["Symbol"].isin(syms)]
    mat = p.pivot_table(index="Symbol", columns="Date", values="Flux_RON", aggfunc=lambda s: s.sum(min_count=1))
    return mat.reindex(index=syms)


def fund_equity_totals(df):
    """Valoarea totală a acțiunilor (toate, inclusiv străine) pe fond și dată, în lei."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Fond", "Date", "Valoare"])
    return (df.groupby(["Fond", "Date"], as_index=False)["Valoare_RON"].sum(min_count=1)
              .rename(columns={"Valoare_RON": "Valoare"}))
