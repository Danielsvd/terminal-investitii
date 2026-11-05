"""Rezumatul Zilei, etapa B — „Portofoliul meu azi”: câștigul sau pierderea pozițiilor deschise în ultima ședință.
Funcții pure (fără Streamlit, fără rețea).

Reguli:
- pozițiile = cele deschise din registrul `Tranzactii` (ca în „3. Portofoliu”), fără obligațiuni și titluri de stat
  (nu au preț zilnic de piață);
- fiecare poziție se raportează la ultima ședință a pieței ei (BVB, SUA, Europa); un simbol fără cotație în acea zi
  nu intră în total (variația lui ar fi din altă zi) și e listat separat;
- P/L în moneda poziției = cantitatea de acum × (închiderea ședinței − închiderea precedentă), pe prețuri
  ajustate doar la splituri (Close din Yahoo cu auto_adjust=False): în ziua ex-dividend scăderea prețului apare
  ca pierdere, iar dividendul vine separat, în numerar — ca la broker;
- P/L în lei = P/L în moneda poziției × ultimul curs BNR: doar efectul prețului, fără efectul cursului din ziua
  respectivă (BNR fixează cursul o dată pe zi, la ~13:00, înainte de închiderea Wall Street);
- contribuția = P/L-ul poziției în lei ÷ valoarea din ziua precedentă a tuturor pozițiilor incluse (puncte procentuale);
  suma contribuțiilor = variația portofoliului;
- σ = aceeași regulă ca în restul Rezumatului (`briefing.daily_move`); peste 2σ = mișcare neobișnuită (netestat);
- dacă registrul are o tranzacție cu simbolul în ziua ședinței sau după ziua precedentă, cantitatea s-a schimbat
  în ziua respectivă și P/L-ul e aproximativ (marcat).
"""
import numpy as np
import pandas as pd

from analytics import briefing
from data.universes import market_of

SIGMA_ALERT = 2.0          # mișcare neobișnuită, ca în restul Rezumatului (netestat)
TOP_CONTRIB = 3            # câte contribuții pozitive / negative se arată
MARKET_LABEL = {"bvb": "BVB", "us": "SUA", "eu": "Europa"}
MARKET_ORDER = ("bvb", "us", "eu")
PENCE_CODES = ("GBX", "GBP_PENCE", "GBp")   # cotații în pence: cursul GBP ÷ 100

COLUMNS = ["Symbol", "Market", "Currency", "Quantity", "Session", "Date", "Close", "Prev", "Pct", "Sigma",
           "PnL", "PnL_RON", "Value_RON", "PrevValue_RON", "Contribution", "Status", "Approx"]

STATUS_OK = "ok"
STATUS_NO_PRICE = "fără preț"
STATUS_STALE = "fără cotație în ședință"
STATUS_NO_FX = "fără curs BNR"


def held_positions(positions, nominal_classes=("Obligatiuni",), eps=1e-9):
    """Pozițiile deschise din `ledger.build_positions`, fără clasele evaluate la nominal: Symbol, Class, Currency, Quantity."""
    if positions is None or len(positions) == 0:
        return pd.DataFrame(columns=["Symbol", "Class", "Currency", "Quantity"])
    held = positions[(positions["Quantity"] > eps) & ~positions["Class"].isin(nominal_classes)]
    return held[["Symbol", "Class", "Currency", "Quantity"]].reset_index(drop=True)


def last_trade_dates(ledger, trade_types=("OPEN", "BUY", "SELL", "BONUS", "SPLIT")):
    """{simbol: data ultimei tranzacții care schimbă cantitatea} (normalizată la zi)."""
    if ledger is None or len(ledger) == 0:
        return {}
    t = ledger[ledger["Type"].isin(trade_types) & ledger["Symbol"].astype(bool)].dropna(subset=["Date"])
    if t.empty:
        return {}
    return {s: pd.Timestamp(d).normalize() for s, d in t.groupby("Symbol")["Date"].max().items()}


def fx_from_bnr(history):
    """Ultimul curs BNR pe fiecare valută: (dict cod → lei pe unitate, dict cod → data cursului). RON = 1."""
    rates, dates = {"RON": 1.0}, {}
    if history is None or len(history) == 0:
        return rates, dates
    for code in history.columns:
        s = pd.to_numeric(history[code], errors="coerce").dropna()
        s = s[s > 0]
        if len(s):
            rates[str(code).upper()] = float(s.iloc[-1])
            dates[str(code).upper()] = s.index[-1]
    return rates, dates


def rate_for(currency, fx):
    """Lei pentru o unitate din moneda poziției; pence = GBP ÷ 100. None dacă lipsește cursul."""
    cur = str(currency or "").strip()
    if cur in PENCE_CODES:
        gbp = fx.get("GBP")
        return gbp / 100.0 if gbp else None
    r = fx.get(cur.upper())
    return float(r) if r else None


def _clean_series(s):
    s = pd.to_numeric(pd.Series(s, dtype="float64"), errors="coerce").dropna()
    s = s[s > 0]
    if len(s) and isinstance(s.index, pd.DatetimeIndex):
        idx = s.index.tz_localize(None) if s.index.tz is not None else s.index
        s = pd.Series(s.values, index=idx.normalize())
        s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def market_sessions(close, symbols, given=None):
    """Data ultimei ședințe pe piață. `given` = {piață: dată} (de ex. din universul întreg al Rezumatului, mai sigur);
    pentru piețele fără dată dată, `briefing.session_date` pe simbolurile deținute din acea piață."""
    out = {m: (pd.Timestamp(d).normalize() if d is not None else None) for m, d in (given or {}).items()}
    frame = close if close is not None else pd.DataFrame()
    for m in MARKET_ORDER:
        if out.get(m) is not None:
            continue
        cols = [s for s in symbols if market_of(s) == m and s in frame.columns]
        if cols:
            out[m] = briefing.session_date(frame[cols])
    return out


def portfolio_day(holdings, close, fx, sessions=None, trade_dates=None):
    """P/L-ul ultimei ședințe pe pozițiile deschise.

    holdings: DataFrame Symbol, Currency, Quantity (de la `held_positions`).
    close: DataFrame cu închiderile (o coloană pe simbol, index de date).
    fx: {cod: lei pe unitate} (de la `fx_from_bnr`).
    sessions: {piață: dată}; lipsă → `market_sessions`.
    trade_dates: {simbol: data ultimei tranzacții} (de la `last_trade_dates`).

    Întoarce (rows, totals): rows = DataFrame cu COLUMNS (Pct, Contribution ca fracții), sortat după contribuție;
    totals = dict value_ron (valoarea de azi a pozițiilor cu preț și curs, inclusiv cele fără cotație în ședință, la
    ultima lor cotație), n_valued, pnl_ron, prev_ron, pct, n, by_market {piață: dict value_ron, n_valued, pnl_ron,
    prev_ron, pct, n, held, session}, excluded
    (listă de (simbol, motiv)), alerts (simbolurile cu |σ| ≥ SIGMA_ALERT), top, bottom (cele mai mari contribuții
    pozitive / negative, câte TOP_CONTRIB).
    """
    holdings = holdings if holdings is not None else pd.DataFrame(columns=["Symbol", "Currency", "Quantity"])
    close = close if close is not None else pd.DataFrame()
    trade_dates = trade_dates or {}
    symbols = list(holdings["Symbol"])
    sessions = market_sessions(close, symbols, sessions)
    rows = []
    for rec in holdings.itertuples(index=False):
        sym, cur, qty = rec.Symbol, rec.Currency, float(rec.Quantity)
        mkt = market_of(sym)
        session = sessions.get(mkt)
        row = {c: None for c in COLUMNS}
        row.update({"Symbol": sym, "Market": mkt, "Currency": cur, "Quantity": qty, "Session": session,
                    "Status": STATUS_OK, "Approx": False})
        s = _clean_series(close[sym]) if sym in close.columns else pd.Series(dtype="float64")
        if session is not None:
            s = s[s.index <= session]
        if s.empty or session is None:
            row["Status"] = STATUS_NO_PRICE
            rows.append(row)
            continue
        m = briefing.daily_move(s)
        row.update({"Date": m["date"], "Close": m["last"], "Prev": m["prev"], "Pct": m["pct"], "Sigma": m["sigma"]})
        rate = rate_for(cur, fx)
        if m["date"] != session or m["prev"] is None:
            row["Status"] = STATUS_STALE
            if rate is not None:            # valoarea de azi la ultima cotație (intră în „Valoare azi”, nu în P/L)
                row["Value_RON"] = qty * m["last"] * rate
            rows.append(row)
            continue
        row["PnL"] = qty * (m["last"] - m["prev"])
        if rate is None:
            row["Status"] = STATUS_NO_FX
            rows.append(row)
            continue
        row.update({"PnL_RON": row["PnL"] * rate, "Value_RON": qty * m["last"] * rate, "PrevValue_RON": qty * m["prev"] * rate})
        prev_day = s.index[-2]
        last_trade = trade_dates.get(sym)
        row["Approx"] = bool(last_trade is not None and last_trade > prev_day)
        rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)
    ok = df[df["Status"] == STATUS_OK]
    prev_total = float(ok["PrevValue_RON"].sum()) if len(ok) else 0.0
    if prev_total > 0:
        df.loc[ok.index, "Contribution"] = ok["PnL_RON"] / prev_total
    pnl_total = float(ok["PnL_RON"].sum()) if len(ok) else None
    by_market = {}
    for mkt in MARKET_ORDER:
        part = ok[ok["Market"] == mkt]
        held_here = df[df["Market"] == mkt]
        if held_here.empty:
            continue
        p_prev = float(part["PrevValue_RON"].sum()) if len(part) else 0.0
        p_pnl = float(part["PnL_RON"].sum()) if len(part) else None
        val_here = pd.to_numeric(held_here["Value_RON"], errors="coerce").dropna()
        by_market[mkt] = {"pnl_ron": p_pnl, "prev_ron": p_prev if len(part) else None,
                          "value_ron": float(val_here.sum()) if len(val_here) else None, "n_valued": len(val_here),
                          "pct": (p_pnl / p_prev) if len(part) and p_prev > 0 else None,
                          "n": len(part), "held": len(held_here), "session": sessions.get(mkt)}
    ranked = df[df["Contribution"].notna()].sort_values("Contribution", ascending=False)
    top = ranked[ranked["Contribution"] > 0].head(TOP_CONTRIB)
    bottom = ranked[ranked["Contribution"] < 0].sort_values("Contribution").head(TOP_CONTRIB)
    sig = pd.to_numeric(df["Sigma"], errors="coerce")
    alerts = df[(df["Status"] == STATUS_OK) & (sig.abs() >= SIGMA_ALERT)]
    vals = pd.to_numeric(df["Value_RON"], errors="coerce").dropna()
    totals = {
        "value_ron": float(vals.sum()) if len(vals) else None, "n_valued": len(vals),
        "pnl_ron": pnl_total,
        "prev_ron": prev_total if len(ok) else None,
        "pct": (pnl_total / prev_total) if pnl_total is not None and prev_total > 0 else None,
        "n": len(ok), "held": len(df),
        "by_market": by_market,
        "excluded": [(r.Symbol, r.Status) for r in df[df["Status"] != STATUS_OK].itertuples(index=False)],
        "alerts": list(alerts["Symbol"]),
        "approx": list(df.loc[df["Approx"].astype(bool), "Symbol"]),
        "top": list(top["Symbol"]), "bottom": list(bottom["Symbol"]),
    }
    order = df["Contribution"].astype("float64").fillna(-np.inf)
    df = df.loc[order.sort_values(ascending=False).index].reset_index(drop=True)
    return df, totals
