"""Evaluarea pozițiilor și curba de valoare a portofoliului. Funcții pure."""
import numpy as np
import pandas as pd


def value_positions(positions, live_prices, closes):
    """Evaluează fiecare poziție la prețul curent.

    positions   : DataFrame cu `Symbol`, `Quantity`, `AvgPrice` (numerice).
    live_prices : dict simbol -> preț curent, sau None când prețul nu a putut fi citit.
    closes      : DataFrame cu închiderile zilnice, o coloană per simbol.

    Un preț live lipsă este înlocuit cu ultima închidere disponibilă. Dacă nu
    există nici aceasta, poziția rămâne fără preț (NaN) și este raportată în
    `notes['missing_price']`. Nu se folosește niciodată prețul 0: o poziție
    evaluată la 0 ar apărea ca pierdere de 100%.

    Întoarce (tabel, variație_zilnică_abs, variație_zilnică_pct, notes).
    """
    rows = []
    notes = {"missing_price": [], "price_from_close": []}
    daily_abs = 0.0

    for _, row in positions.iterrows():
        sym = row["Symbol"]
        qty = float(row["Quantity"])
        avg_p = float(row["AvgPrice"])

        series = closes[sym].dropna() if sym in closes.columns else pd.Series(dtype="float64")

        curr_p = live_prices.get(sym) if live_prices else None
        if curr_p is None or not np.isfinite(curr_p) or curr_p <= 0:
            if len(series):
                curr_p = float(series.iloc[-1])
                if sym not in notes["price_from_close"]:
                    notes["price_from_close"].append(sym)
            else:
                curr_p = np.nan
                if sym not in notes["missing_price"]:
                    notes["missing_price"].append(sym)

        prev_p = float(series.iloc[-2]) if len(series) >= 2 else np.nan

        mkt_val = qty * curr_p
        inv_val = qty * avg_p
        profit = mkt_val - inv_val
        profit_pct = (profit / inv_val * 100) if inv_val != 0 else np.nan

        if np.isfinite(curr_p) and np.isfinite(prev_p):
            daily_abs += (curr_p - prev_p) * qty

        rows.append({
            "Symbol": sym, "Quantity": qty, "AvgPrice": avg_p, "CurrentPrice": curr_p,
            "MarketValue": mkt_val, "Profit": profit, "Profit %": profit_pct,
        })

    table = pd.DataFrame(rows)
    total_now = float(np.nansum(table["MarketValue"])) if len(table) else 0.0
    base = total_now - daily_abs
    daily_pct = (daily_abs / base * 100) if base != 0 else 0.0
    return table, daily_abs, daily_pct, notes


def portfolio_curve(positions, closes):
    """Valoarea zilnică a pozițiilor curente, pe istoricul comun al simbolurilor.

    Curba începe la prima dată la care TOATE simbolurile deținute au preț. Nu se
    completează prețuri înapoi în timp (fără bfill): un preț inventat înainte de
    listare ar falsifica randamentele. Golurile din interior (sărbători diferite
    pe burse diferite) se completează cu ultimul preț cunoscut.

    Atenție: curba aplică deținerile de AZI pe trecut. Arată cum s-ar fi
    comportat portofoliul actual, nu performanța realizată efectiv.

    Întoarce (serie, notes) cu notes = {'no_history': [...], 'curve_start': data,
    'limiting_symbol': simbolul cu cel mai scurt istoric}.
    """
    notes = {"no_history": [], "curve_start": None, "limiting_symbol": None}
    held = positions.groupby("Symbol")["Quantity"].sum()
    held = held[held != 0]

    cols = [s for s in held.index if s in closes.columns and closes[s].notna().any()]
    notes["no_history"] = [s for s in held.index if s not in cols]
    if not cols:
        return pd.Series(dtype="float64"), notes

    px = closes[cols].sort_index()
    first_valid = px.apply(lambda c: c.first_valid_index())
    notes["limiting_symbol"] = first_valid.idxmax()

    px = px.ffill().dropna(how="any")
    if px.empty:
        return pd.Series(dtype="float64"), notes

    curve = px.mul(held[cols], axis=1).sum(axis=1)
    notes["curve_start"] = curve.index[0]
    return curve, notes
