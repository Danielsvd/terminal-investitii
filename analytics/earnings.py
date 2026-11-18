"""Tabelul „Raportări vs Așteptări” (EPS raportat față de estimarea analiștilor). Funcții pure.

Sursa e `yf.Ticker(...).earnings_history`. Când Yahoo nu are estimări ale analiștilor pentru un emitent
(frecvent la BVB, de exemplu TEL.RO), tabelul vine doar cu `epsActual`, fără `epsEstimate`,
`epsDifference` și `surprisePercent`. Coloanele lipsă nu se inventează: apar ca NaN („N/A” în interfață).
"""
import pandas as pd

# coloana yfinance → eticheta afișată, în ordinea afișării
EARNINGS_COLUMNS = {
    "epsEstimate": "Estimare",
    "epsActual": "Realizat",
    "epsDifference": "Diferență",
    "surprisePercent": "Surpriză %",
}


def earnings_table(df):
    """earnings_history → (tabel cu etichete în română, are_estimari).

    - `None`, tabel gol sau fără niciuna dintre cele patru coloane → (DataFrame gol, False).
    - Coloanele lipsă apar ca NaN, nu ca 0.
    - `are_estimari` e True doar dacă există coloana `epsEstimate` și are cel puțin o valoare.
    - Valorile nenumerice devin NaN.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=list(EARNINGS_COLUMNS.values())), False
    present = [c for c in EARNINGS_COLUMNS if c in df.columns]
    if not present:
        return pd.DataFrame(columns=list(EARNINGS_COLUMNS.values())), False
    out = pd.DataFrame(index=df.index)
    for col, label in EARNINGS_COLUMNS.items():
        if col in df.columns:
            out[label] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[label] = float("nan")
    has_estimates = "epsEstimate" in df.columns and out["Estimare"].notna().any()
    return out, bool(has_estimates)
