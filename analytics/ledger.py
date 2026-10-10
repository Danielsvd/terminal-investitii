"""Registrul de tranzacții (foaia `Tranzactii`): citire, dețineri la cost mediu, rezultate pe clase, XIRR.

Funcții pure: fără Streamlit, fără rețea. Toate sumele rămân în moneda rândului; conversia
valutară se face în alt strat.

Convenții ale foii (o tranzacție pe rând):
  Date, Symbol, Class, Type, Quantity, Price, Currency, Fees, Amount, Broker, ID, Notes
- `Amount` este efectul net în numerar, cu semn (cumpărare negativ, vânzare/dividend pozitiv),
  deja după comisioane. `Fees` e informativ și NU se scade a doua oară.
- `Quantity` e pozitivă la BUY/SELL/OPEN/BONUS și are semn la SPLIT și RIGHTS.
- O celulă goală înseamnă „lipsă” (NaN), niciodată 0.
"""
import math

import numpy as np
import pandas as pd

from data.helpers import smart_to_float

COLUMNS = ["Date", "Symbol", "Class", "Type", "Quantity", "Price", "Currency", "Fees", "Amount", "Broker", "ID", "Notes"]
NUMERIC = ["Quantity", "Price", "Fees", "Amount"]

TRADE_TYPES = {"OPEN", "BUY", "SELL", "BONUS", "SPLIT"}          # schimbă cantitatea deținută
INCOME_TYPES = {"DIV", "COUPON"}                                  # venit atribuit unui simbol
CASH_TYPES = {"DEPOSIT", "WITHDRAW", "FX", "TRANSFER", "INTEREST"}
KNOWN_TYPES = TRADE_TYPES | INCOME_TYPES | CASH_TYPES | {"TAX", "FEE", "RIGHTS"}
EXTERNAL_FLOW_TYPES = {"DEPOSIT", "WITHDRAW"}                     # bani intrați/ieșiți din portofoliu

QTY_EPS = 1e-9


def _cell_to_float(value):
    """Număr din foaie (format RO sau US); celula goală sau nenumerică dă NaN, nu 0."""
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or not any(ch.isdigit() for ch in text):
        return np.nan
    return smart_to_float(text)


def parse_ledger(values):
    """Transformă `worksheet.get_all_values()` (antet + rânduri) în registru tipizat.

    Întoarce (DataFrame sortat cronologic, listă de probleme). Rândurile complet goale se
    ignoră. Rândurile cu dată necitibilă sau tip necunoscut sunt scoase din registru și
    raportate: un rând greșit nu trebuie să intre tăcut în calcule.
    """
    problems = []
    if not values:
        return pd.DataFrame(columns=COLUMNS), ["Foaia e goală."]
    header = [str(h).strip() for h in values[0]]
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        return pd.DataFrame(columns=COLUMNS), [f"Coloane lipsă în antet: {', '.join(missing)}"]

    width = len(header)
    rows = [list(r) + [""] * (width - len(r)) for r in values[1:]]
    rows = [r[:width] for r in rows if any(str(c).strip() for c in r)]
    df = pd.DataFrame(rows, columns=header)[COLUMNS].copy()
    df["_row"] = range(2, 2 + len(df))                    # numărul rândului din foaie, pentru mesaje

    for col in ("Symbol", "Class", "Type", "Currency", "Broker", "ID", "Notes"):
        df[col] = df[col].astype(str).str.strip()
    df["Type"] = df["Type"].str.upper()
    df["Currency"] = df["Currency"].str.upper()
    for col in NUMERIC:
        df[col] = df[col].map(_cell_to_float)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")

    bad_date = df["Date"].isna()
    for row in df.loc[bad_date, "_row"]:
        problems.append(f"Rândul {row}: dată necitibilă.")
    bad_type = ~df["Type"].isin(KNOWN_TYPES)
    for row, typ in df.loc[bad_type & ~bad_date, ["_row", "Type"]].itertuples(index=False):
        problems.append(f"Rândul {row}: tip necunoscut „{typ}”.")
    df = df.loc[~(bad_date | bad_type)].copy()

    # OPEN fără simbol e sold de deschidere în numerar: are doar Amount, nu și cantitate.
    need_qty = df["Type"].isin(TRADE_TYPES) & df["Symbol"].ne("") & df["Quantity"].isna()
    for row, typ in df.loc[need_qty, ["_row", "Type"]].itertuples(index=False):
        problems.append(f"Rândul {row}: {typ} fără cantitate.")
    df = df.loc[~need_qty].copy()

    dup = df["ID"].ne("") & df["ID"].duplicated(keep="first")
    for row, oid in df.loc[dup, ["_row", "ID"]].itertuples(index=False):
        problems.append(f"Rândul {row}: ID duplicat „{oid}” (rând ignorat).")
    df = df.loc[~dup].copy()

    df = df.sort_values(["Date", "_row"], kind="stable").reset_index(drop=True)
    return df, problems


def build_positions(ledger):
    """Dețineri și rezultate pe simbol, la cost mediu ponderat.

    Regulile, în ordine cronologică:
    - OPEN  : cantitate + ; cost + cantitate × preț (sold de deschidere, fără numerar).
            OPEN fără simbol e numerar de deschidere și apare doar în `cash_balances`.
    - BUY   : cantitate + ; cost + (−Amount), adică prețul plătit cu tot cu comisioane.
    - BONUS : cantitate + ; costul rămâne (acțiuni gratuite: costul mediu scade).
    - SPLIT : cantitate ± ; costul rămâne (consolidare sau splitare).
    - SELL  : realizat += Amount − cost_mediu × cantitate ; costul scade proporțional.
    - DIV/COUPON → venit ; TAX → impozite ; FEE → comisioane (toate cu semnul din Amount).
    - RIGHTS și rândurile fără simbol nu intră în dețineri.

    Diferențe față de rapoartele brokerilor, toate intenționate:
    - XTB închide pe loturi: pe o vânzare parțială profitul poate diferi, pe o poziție
      închisă complet totalul e același.
    - Tradeville dă acțiunilor gratuite un cost egal cu valoarea nominală (convenție fiscală).
      Aici costul lor e 0, pentru că nu s-a plătit nimic: profitul se leagă astfel cu numerarul.
    - Impozitul reținut la sursă la vânzare e deja scăzut din `Amount`, deci micșorează realizatul.

    Întoarce (DataFrame cu o linie pe simbol, listă de probleme).
    """
    state, problems = {}, []
    for rec in ledger.itertuples(index=False):
        sym, typ = rec.Symbol, rec.Type
        if not sym or typ == "RIGHTS" or typ in CASH_TYPES:
            continue
        st = state.setdefault(sym, {"Symbol": sym, "Class": rec.Class, "Currency": rec.Currency, "Quantity": 0.0,
                                    "CostBasis": 0.0, "Realized": 0.0, "Income": 0.0, "Taxes": 0.0, "Fees": 0.0,
                                    "Bought": 0.0, "Sold": 0.0})
        amount = 0.0 if pd.isna(rec.Amount) else float(rec.Amount)
        qty = 0.0 if pd.isna(rec.Quantity) else float(rec.Quantity)

        if typ == "OPEN":
            if pd.isna(rec.Price):
                problems.append(f"{sym}: sold de deschidere fără preț (cost necunoscut, considerat lipsă).")
                st["CostBasis"] = np.nan
            else:
                st["CostBasis"] += qty * float(rec.Price)
                st["Bought"] += qty * float(rec.Price)
            st["Quantity"] += qty
        elif typ == "BUY":
            st["Quantity"] += qty
            st["CostBasis"] += -amount
            st["Bought"] += -amount
        elif typ in ("BONUS", "SPLIT"):
            st["Quantity"] += qty
        elif typ == "SELL":
            held = st["Quantity"]
            if qty > held + 1e-6:
                problems.append(f"{sym}: vânzare de {qty:g} la {rec.Date:%Y-%m-%d}, dar în registru sunt doar {held:g} "
                                "(lipsește istoric). Profitul realizat pe acest simbol nu e de încredere.")
                sold_cost = st["CostBasis"]
                st["Quantity"], st["CostBasis"] = 0.0, 0.0
            else:
                sold_cost = st["CostBasis"] * (qty / held) if held > QTY_EPS else 0.0
                st["Quantity"] = held - qty
                st["CostBasis"] -= sold_cost
            st["Realized"] += amount - sold_cost
            st["Sold"] += amount
            if abs(st["Quantity"]) < 1e-6:                 # poziție închisă: fără resturi de rotunjire
                st["Quantity"], st["CostBasis"] = 0.0, 0.0
        elif typ in INCOME_TYPES:
            st["Income"] += amount
        elif typ == "TAX":
            st["Taxes"] += amount
        elif typ == "FEE":
            st["Fees"] += amount

    table = pd.DataFrame(list(state.values()),
                         columns=["Symbol", "Class", "Currency", "Quantity", "CostBasis", "Realized", "Income",
                                  "Taxes", "Fees", "Bought", "Sold"])
    if table.empty:
        table["AvgCost"] = pd.Series(dtype="float64")
        return table, problems
    table["AvgCost"] = np.where(table["Quantity"] > QTY_EPS, table["CostBasis"] / table["Quantity"].where(table["Quantity"] > QTY_EPS), np.nan)
    return table.sort_values(["Class", "Symbol"]).reset_index(drop=True), problems


def class_summary(positions):
    """Totaluri pe (clasă, monedă): cost deschis, realizat, venit, impozite, comisioane.

    `Result` = realizat + venit + impozite + comisioane (impozitele și comisioanele au deja semn
    negativ). Nu include profitul nerealizat, care are nevoie de prețuri curente.
    """
    cols = ["CostBasis", "Realized", "Income", "Taxes", "Fees"]
    if positions.empty:
        return pd.DataFrame(columns=["Class", "Currency", "OpenPositions"] + cols + ["Result"])
    grouped = positions.groupby(["Class", "Currency"], as_index=False)
    out = grouped[cols].sum(min_count=1)
    out["OpenPositions"] = grouped["Quantity"].agg(lambda q: int((q > QTY_EPS).sum()))["Quantity"].values
    out["Result"] = out[["Realized", "Income", "Taxes", "Fees"]].sum(axis=1)
    return out[["Class", "Currency", "OpenPositions"] + cols + ["Result"]]


def cash_balances(ledger):
    """Numerar pe (broker, monedă) = suma `Amount` pe toate rândurile."""
    if ledger.empty:
        return pd.DataFrame(columns=["Broker", "Currency", "Cash"])
    out = ledger.groupby(["Broker", "Currency"], as_index=False)["Amount"].sum()
    return out.rename(columns={"Amount": "Cash"})


def class_cashflows(ledger, asset_class, currency):
    """Fluxurile de numerar ale unei clase într-o monedă, din perspectiva investitorului.

    Amount are deja semnul corect (cumpărare −, vânzare/dividend +, impozit −). Soldul de
    deschidere intră ca investiție la data lui: −cantitate × preț. Valoarea curentă a
    pozițiilor rămase se adaugă de către apelant, ca flux pozitiv la data evaluării.
    Întoarce listă de (dată, sumă), fără fluxuri nule.
    """
    part = ledger[(ledger["Class"] == asset_class) & (ledger["Currency"] == currency)]
    flows = []
    for rec in part.itertuples(index=False):
        if rec.Type == "OPEN":
            value = -(rec.Quantity * rec.Price) if not (pd.isna(rec.Quantity) or pd.isna(rec.Price)) else np.nan
        else:
            value = rec.Amount
        if pd.isna(value) or value == 0:
            continue
        flows.append((rec.Date, float(value)))
    return flows


def xirr(flows, lo=-0.9999, hi=100.0, tol=1e-9, max_iter=200):
    """Rata internă de rentabilitate anualizată pentru fluxuri la date neregulate.

    flows: listă de (dată, sumă); investițiile sunt negative, încasările pozitive.
    Rezolvă Σ sumă / (1 + r)^(zile/365) = 0 prin bisecție. Întoarce None când nu există o
    soluție unică de căutat: sub două fluxuri, toate de același semn, sau toate în aceeași zi.
    """
    flows = [(pd.Timestamp(d), float(a)) for d, a in flows if a is not None and not pd.isna(a) and a != 0]
    if len(flows) < 2:
        return None
    if not (any(a < 0 for _, a in flows) and any(a > 0 for _, a in flows)):
        return None
    t0 = min(d for d, _ in flows)
    years = [(d - t0).total_seconds() / 86400 / 365.0 for d, _ in flows]
    if max(years) == 0:
        return None
    amounts = [a for _, a in flows]

    def npv(rate):
        return sum(a / (1.0 + rate) ** y for a, y in zip(amounts, years))

    f_lo, f_hi = npv(lo), npv(hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0
