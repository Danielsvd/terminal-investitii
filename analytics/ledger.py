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


MIN_XIRR_DAYS = 90      # sub atât, randamentul anualizat nu se afișează


def flows_span_days(flows):
    """Câte zile acoperă fluxurile (de la primul la ultimul); 0 pentru mai puțin de două."""
    dates = [pd.Timestamp(d) for d, a in flows if a is not None and not pd.isna(a) and a != 0]
    if len(dates) < 2:
        return 0.0
    return (max(dates) - min(dates)).total_seconds() / 86400


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


# --- Evaluare la prețuri curente ---------------------------------------------

NOMINAL_CLASSES = ("Obligatiuni",)      # fără preț de piață în sursele aplicației: evaluate la nominal
BOND_NOMINAL = 100.0                    # titlurile de stat și obligațiunile din registru au nominal 100
PRICE_LIVE, PRICE_CLOSE, PRICE_NOMINAL, PRICE_MISSING = "live", "ultima închidere", "nominal (proxy)", "lipsă"


def _usable_price(mapping, symbol):
    price = (mapping or {}).get(symbol)
    return None if (price is None or pd.isna(price) or price <= 0) else float(price)


def value_open_positions(positions, prices, close_prices=None, nominal_classes=NOMINAL_CLASSES, nominal=BOND_NOMINAL):
    """Evaluează pozițiile deschise: valoare de piață și profit nerealizat.

    prices: dict simbol -> preț curent sau None. close_prices: dict simbol -> ultima închidere,
    folosită doar când prețul curent lipsește (sursa apare în `PriceSource`). Clasele din
    `nominal_classes` nu au preț de piață disponibil și se evaluează la nominal; e un proxy
    și e marcat ca atare. Fără niciun preț, valoarea rămâne NaN (nu 0: ar apărea ca pierdere de 100%).
    Întoarce doar pozițiile cu cantitate > 0, cu coloanele în plus
    Price, PriceSource, MarketValue, Unrealized, UnrealizedPct.
    """
    cols = list(positions.columns) + ["Price", "PriceSource", "MarketValue", "Unrealized", "UnrealizedPct"]
    held = positions[positions["Quantity"] > QTY_EPS].copy()
    if held.empty:
        return pd.DataFrame(columns=cols)
    price_list, source_list = [], []
    for rec in held.itertuples(index=False):
        if rec.Class in nominal_classes:
            price_list.append(float(nominal)); source_list.append(PRICE_NOMINAL)
            continue
        live, close = _usable_price(prices, rec.Symbol), _usable_price(close_prices, rec.Symbol)
        if live is not None:
            price_list.append(live); source_list.append(PRICE_LIVE)
        elif close is not None:
            price_list.append(close); source_list.append(PRICE_CLOSE)
        else:
            price_list.append(np.nan); source_list.append(PRICE_MISSING)
    held["Price"], held["PriceSource"] = price_list, source_list
    held["MarketValue"] = held["Quantity"] * held["Price"]
    held["Unrealized"] = held["MarketValue"] - held["CostBasis"]
    held["UnrealizedPct"] = np.where(held["CostBasis"] > 0, held["Unrealized"] / held["CostBasis"].where(held["CostBasis"] > 0) * 100, np.nan)
    return held[cols].reset_index(drop=True)


def class_results(ledger, positions, valued, asof):
    """Rezultatul pe (clasă, monedă): realizat, nerealizat, venit, impozite, total și XIRR.

    - `Total` = realizat + nerealizat + venit + impozite + comisioane.
    - `XIRR` folosește fluxurile clasei plus valoarea de piață a pozițiilor rămase, ca încasare
      la `asof`. Dacă vreunei poziții deschise îi lipsește prețul, MarketValue, Unrealized,
      Total și XIRR rămân lipsă pentru clasa respectivă: un total parțial ar induce în eroare.
    - `Flows` e numărul de fluxuri folosite; sub ~20 randamentul e orientativ. Dacă fluxurile
      acoperă mai puțin de `MIN_XIRR_DAYS` zile, XIRR rămâne lipsă.
    Clasele fără dețineri și fără rezultat (de ex. „Drepturi”) nu apar.
    """
    out_cols = ["Class", "Currency", "OpenPositions", "CostBasis", "MarketValue", "Unrealized", "Realized", "Income",
                "Taxes", "Fees", "Total", "XIRR", "Flows", "MissingPrices", "Proxy"]
    summary = class_summary(positions)
    rows = []
    for rec in summary.itertuples(index=False):
        part = valued[(valued["Class"] == rec.Class) & (valued["Currency"] == rec.Currency)]
        missing = sorted(part.loc[part["MarketValue"].isna(), "Symbol"])
        if missing:
            market = unrealized = np.nan
        else:
            market = float(part["MarketValue"].sum()) if not part.empty else 0.0
            unrealized = float(part["Unrealized"].sum()) if not part.empty else 0.0
        flows = class_cashflows(ledger, rec.Class, rec.Currency)
        rate = None
        if not missing:
            all_flows = flows + ([(pd.Timestamp(asof), market)] if market > 0 else [])
            # o tranzacție de câteva zile, anualizată, dă procente fără sens: sub MIN_XIRR_DAYS nu se afișează
            if flows_span_days(all_flows) >= MIN_XIRR_DAYS:
                rate = xirr(all_flows)
        total = np.nan if missing else rec.Realized + unrealized + rec.Income + rec.Taxes + rec.Fees
        if rec.OpenPositions == 0 and not flows:
            continue
        rows.append({"Class": rec.Class, "Currency": rec.Currency, "OpenPositions": rec.OpenPositions,
                     "CostBasis": rec.CostBasis, "MarketValue": market, "Unrealized": unrealized,
                     "Realized": rec.Realized, "Income": rec.Income, "Taxes": rec.Taxes, "Fees": rec.Fees,
                     "Total": total, "XIRR": np.nan if rate is None else rate * 100, "Flows": len(flows),
                     "MissingPrices": ", ".join(missing),
                     "Proxy": bool((part["PriceSource"] == PRICE_NOMINAL).any()) if not part.empty else False})
    return pd.DataFrame(rows, columns=out_cols)


def consolidate(results, cash, fx_rates, base="RON"):
    """Valoarea de azi a portofoliului în moneda de bază: poziții + numerar, la cursul curent.

    fx_rates: dict monedă -> câte unități din `base` face o unitate (ex. {"EUR": 5.1, "USD": 4.4});
    `base` are implicit cursul 1. Întoarce (tabel pe monedă, total, listă de monede fără curs).
    Dacă lipsește un curs sau o valoare de piață, totalul e None: nu se afișează un total parțial.
    """
    rates = dict(fx_rates or {})
    rates[base] = 1.0
    currencies = sorted(set(results["Currency"]) | set(cash["Currency"]))
    rows, missing, incomplete = [], [], False
    for cur in currencies:
        res = results[results["Currency"] == cur]
        positions_value = res["MarketValue"].sum(min_count=1) if not res.empty else 0.0
        if not res.empty and res["MarketValue"].isna().any():
            positions_value = np.nan
        cash_value = float(cash.loc[cash["Currency"] == cur, "Cash"].sum())
        rate = rates.get(cur)
        if rate is None or pd.isna(rate) or rate <= 0:
            rate = np.nan
            missing.append(cur)
        value_base = (positions_value + cash_value) * rate
        incomplete = incomplete or pd.isna(value_base)
        rows.append({"Currency": cur, "Positions": positions_value, "Cash": cash_value, "Rate": rate, "ValueBase": value_base})
    table = pd.DataFrame(rows, columns=["Currency", "Positions", "Cash", "Rate", "ValueBase"])
    total = None if (incomplete or table.empty) else float(table["ValueBase"].sum())
    return table, total, missing


# --- Portofoliul întreg în moneda de bază: depuneri, XIRR, curbă zilnică, TWR ----

def fx_on(fx_history, currency, date, base="RON"):
    """Cursul `currency`/`base` valabil la `date`: ultima cotație din acea zi sau dinainte.

    fx_history: DataFrame cu index de date și o coloană per monedă (unități de `base` pentru
    o unitate de monedă). Întoarce None dacă nu există nicio cotație până la acea dată
    (nu se ia un curs din viitor).
    """
    if currency == base:
        return 1.0
    if fx_history is None or currency not in getattr(fx_history, "columns", []):
        return None
    series = fx_history[currency].dropna()
    series = series.loc[:pd.Timestamp(date).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)]
    return None if series.empty else float(series.iloc[-1])


def external_flows(ledger, fx_history, base="RON"):
    """Banii intrați în portofoliu sau ieșiți din el, convertiți la cursul din ziua lor.

    Sunt externe: DEPOSIT și WITHDRAW (cu semnul din Amount) și soldurile de deschidere OPEN
    (numerarul, respectiv cantitate × preț pentru poziții). Schimburile valutare și
    transferurile între conturi sunt interne și nu intră aici.
    Întoarce (DataFrame cu Date, Currency, Amount, Rate, AmountBase; listă de probleme).
    Un flux fără curs disponibil rămâne cu AmountBase lipsă și e raportat.
    """
    rows, problems = [], []
    for rec in ledger.itertuples(index=False):
        if rec.Type in EXTERNAL_FLOW_TYPES:
            amount = rec.Amount
        elif rec.Type == "OPEN":
            amount = rec.Amount if not rec.Symbol else rec.Quantity * rec.Price
        else:
            continue
        if pd.isna(amount) or amount == 0:
            continue
        rate = fx_on(fx_history, rec.Currency, rec.Date, base)
        if rate is None:
            problems.append(f"{rec.Date:%Y-%m-%d}: fără curs {rec.Currency}/{base} pentru {rec.Type} de {amount:,.2f} {rec.Currency}.")
        rows.append({"Date": rec.Date, "Currency": rec.Currency, "Amount": float(amount),
                     "Rate": np.nan if rate is None else rate,
                     "AmountBase": np.nan if rate is None else float(amount) * rate})
    return pd.DataFrame(rows, columns=["Date", "Currency", "Amount", "Rate", "AmountBase"]), problems


def money_weighted_result(ext_flows, total_value_base, asof):
    """Rezultatul întregului portofoliu în moneda de bază.

    net_invested = depuneri − retrageri (la cursul din ziua lor); profit = valoarea de azi −
    net_invested; xirr = randament anualizat ponderat cu banii (fracție, nu procent).
    Dacă lipsește valoarea de azi sau cursul vreunui flux, toate trei sunt None.
    """
    empty = {"net_invested": None, "profit": None, "xirr": None}
    if total_value_base is None or pd.isna(total_value_base) or ext_flows.empty or ext_flows["AmountBase"].isna().any():
        return empty
    net = float(ext_flows["AmountBase"].sum())
    flows = [(d, -a) for d, a in zip(ext_flows["Date"], ext_flows["AmountBase"])] + [(pd.Timestamp(asof), float(total_value_base))]
    rate = xirr(flows) if flows_span_days(flows) >= MIN_XIRR_DAYS else None
    return {"net_invested": net, "profit": float(total_value_base) - net, "xirr": rate}


def daily_quantities(ledger, index):
    """Cantitatea deținută din fiecare simbol la sfârșitul fiecărei zile din `index`."""
    part = ledger[ledger["Type"].isin(TRADE_TYPES) & ledger["Symbol"].ne("") & ledger["Quantity"].notna()]
    if part.empty:
        return pd.DataFrame(index=index)
    signed = part["Quantity"].where(part["Type"] != "SELL", -part["Quantity"])
    delta = signed.groupby([part["Date"].dt.normalize(), part["Symbol"]]).sum().unstack(fill_value=0.0)
    full = delta.reindex(delta.index.union(index)).fillna(0.0).cumsum()
    out = full.reindex(index)
    return out.where(out.abs() > 1e-6, 0.0)


def daily_cash(ledger, index):
    """Numerarul pe monedă la sfârșitul fiecărei zile din `index` (suma cumulată a `Amount`)."""
    part = ledger[ledger["Amount"].notna()]
    if part.empty:
        return pd.DataFrame(index=index)
    delta = part.groupby([part["Date"].dt.normalize(), part["Currency"]])["Amount"].sum().unstack(fill_value=0.0)
    return delta.reindex(delta.index.union(index)).fillna(0.0).cumsum().reindex(index)


def unadjust_splits(close, splits):
    """Prețul efectiv istoric dintr-o serie ajustată la splituri.

    Yahoo dă închiderile ajustate la toate spliturile ulterioare; registrul ține cantitățile
    reale din acea zi. Fără dezajustare, o consolidare 1:10 ar umfla de 10 ori valoarea
    dinaintea ei. preț_real[t] = close[t] × produsul rapoartelor de split de după t.
    `splits`: serie cu raportul în ziua splitului (0 sau NaN în rest).
    """
    if splits is None or len(splits) == 0:
        return close
    ratios = splits.reindex(close.index).fillna(0.0)
    ratios = ratios.where(ratios > 0, 1.0)
    # factorul pentru ziua t = produsul rapoartelor din zilele strict ulterioare
    after = ratios.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    return close * after


def ledger_trade_prices(ledger, index):
    """Prețurile din tranzacții (OPEN/BUY/SELL), pe zi și simbol, duse înainte până la următoarea.

    Proxy pentru simbolurile fără istoric de piață: ultimul preț la care s-a tranzacționat.
    """
    part = ledger[ledger["Type"].isin({"OPEN", "BUY", "SELL"}) & ledger["Symbol"].ne("") & ledger["Price"].notna()]
    if part.empty:
        return pd.DataFrame(index=index)
    last = part.groupby([part["Date"].dt.normalize(), part["Symbol"]])["Price"].last().unstack()
    return last.reindex(last.index.union(index)).ffill().reindex(index)


def build_price_table(index, symbol_classes, market_closes, trade_prices, nominal_classes=NOMINAL_CLASSES, nominal=BOND_NOMINAL):
    """Preț zilnic per simbol, cu sursa lui.

    Ordinea: clasele evaluate la nominal primesc `nominal`; restul primesc închiderea de piață
    (dusă înainte peste zilele fără ședință, niciodată înapoi); unde piața nu are date, prețul
    din ultima tranzacție (proxy). Întoarce (prețuri, dict simbol -> "market" | "proxy" | "nominal" | "mixed").
    """
    prices, sources = pd.DataFrame(index=index), {}
    for sym, cls in symbol_classes.items():
        if cls in nominal_classes:
            prices[sym] = float(nominal); sources[sym] = "nominal"
            continue
        market = market_closes[sym].reindex(market_closes.index.union(index)).ffill().reindex(index) if (
            market_closes is not None and sym in market_closes.columns) else pd.Series(np.nan, index=index)
        proxy = trade_prices[sym] if sym in trade_prices.columns else pd.Series(np.nan, index=index)
        prices[sym] = market.combine_first(proxy)
        has_m, has_p = market.notna().any(), (market.isna() & proxy.notna()).any()
        sources[sym] = "mixed" if (has_m and has_p) else ("market" if has_m else ("proxy" if has_p else "missing"))
    return prices, sources


def portfolio_curve_base(quantities, prices, cash, symbol_currency, fx_history, base="RON"):
    """Valoarea zilnică a portofoliului în moneda de bază: poziții + numerar.

    O zi în care o poziție deținută nu are preț, sau lipsește cursul unei monede folosite,
    rămâne fără valoare (NaN): nu se afișează o valoare parțială. Întoarce (serie, simboluri
    care au lăsat zile fără valoare).
    """
    index = quantities.index
    def fx_series(cur):
        if cur == base:
            return pd.Series(1.0, index=index)
        if fx_history is None or cur not in fx_history.columns:
            return pd.Series(np.nan, index=index)
        s = fx_history[cur].dropna()
        return s.reindex(s.index.union(index)).ffill().reindex(index)
    total = pd.Series(0.0, index=index)
    blockers = []
    for sym in quantities.columns:
        qty = quantities[sym]
        held = qty.abs() > 1e-6
        if not held.any():
            continue
        price = prices[sym] if sym in prices.columns else pd.Series(np.nan, index=index)
        value = (qty * price * fx_series(symbol_currency.get(sym, base))).where(held, 0.0)
        if value[held].isna().any():
            blockers.append(sym)
        total = total + value
    for cur in cash.columns:
        value = cash[cur] * fx_series(cur)
        total = total + value.where(cash[cur].abs() > 1e-9, 0.0)
    return total, sorted(blockers)


def twr(values, flows):
    """Randament ponderat cu timpul din valorile zilnice și fluxurile externe.

    values: serie zilnică a valorii portofoliului; flows: serie cu banii intrați (+) sau ieșiți (−)
    în fiecare zi, în aceeași monedă. Fluxul e considerat făcut la începutul zilei:
    r_t = V_t / (V_{t−1} + F_t) − 1. Zilele fără valoare întrerup lanțul: se calculează doar pe
    ultimul șir continuu de valori. Întoarce (serie cumulată a randamentului, randament total)
    sau (serie goală, None) dacă nu există cel puțin două zile.
    """
    values = values.astype(float)
    valid = values.notna()
    if valid.sum() < 2:
        return pd.Series(dtype="float64"), None
    last_gap = np.flatnonzero(~valid.to_numpy())
    start = (last_gap[-1] + 1) if len(last_gap) else 0
    v = values.iloc[start:]
    if len(v) < 2:
        return pd.Series(dtype="float64"), None
    f = flows.reindex(v.index).fillna(0.0)
    base_value = v.shift(1) + f
    r = (v / base_value - 1.0).where(base_value > 0)
    r.iloc[0] = 0.0
    r = r.fillna(0.0)
    cumulative = (1.0 + r).cumprod() - 1.0
    return cumulative, float(cumulative.iloc[-1])


def flows_on_index(ext_flows, index):
    """Fluxurile externe (în moneda de bază) așezate pe zilele din `index`.

    Un flux dintr-o zi care nu e în index (weekend, sărbătoare) trece în prima zi următoare
    din index; cele de după ultima zi se ignoră. Întoarce o serie pe `index`, 0 unde nu e flux.
    """
    out = pd.Series(0.0, index=index)
    if ext_flows.empty or len(index) == 0:
        return out
    days = ext_flows["Date"].dt.normalize()
    pos = index.searchsorted(days.to_numpy(), side="left")
    for i, amount in zip(pos, ext_flows["AmountBase"]):
        if i < len(index) and not pd.isna(amount):
            out.iloc[i] += float(amount)
    return out
