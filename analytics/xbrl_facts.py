"""Situațiile financiare XBRL de la SEC (`companyfacts`) → serii curate: anual, trimestrial, TTM, point-in-time.

Faza 3 SEC (propunerea `claude/propunere_faza3_xbrl_2026-11-27.md`, deciziile D1–D10 acceptate de Daniel pe 27.11.2026).
Funcții pure: fără rețea, fără Streamlit. Aceeași bibliotecă servește aplicația și testul istoric.

Reguli (D2–D5):
  * Indicatorii au o listă de etichete US-GAAP în ordinea priorității (`CONCEPTS`). Pe fiecare perioadă se ia eticheta cu
    prioritatea cea mai mare care are valoare; seria poate combina etichete (ex. `SalesRevenueNet` până în 2017,
    `RevenueFromContractWithCustomer…` din 2018). Unde două etichete au valoare pentru aceeași perioadă și diferă cu
    peste `SEAM_TOL` (1%), perioada e marcată „conflict” și nu intră în CAGR.
  * Doar unitățile `USD`, `USD/shares`, `shares`, și doar rapoartele 10-K / 10-Q (și amendamentele lor). Altă monedă
    sau 20-F → nimic (emitenții străini rămân pe Yahoo, D7).
  * Perioada se deduce din `start`–`end`, NU din `fy`/`fp` (ele descriu raportul, nu valoarea): an = 350–380 de zile,
    trimestru = 80–100 de zile, cumulat 6 / 9 luni. T4 și trimestrele din fluxul de numerar (raportat doar cumulat în 10-Q)
    se obțin din diferențe de valori cumulate cu același început.
  * Point-in-time (D4): fiecare valoare are `filed`; devine cunoscută a doua zi lucrătoare (`available_on`). `asof=None`
    = ultima versiune (aplicația); `asof=data` = valoarea din cel mai recent raport disponibil la acea dată (testul).
  * Lipsă = None / NaN, niciodată 0.
"""
import math
from datetime import date, timedelta

import pandas as pd

TAXONOMIES = ("us-gaap", "dei")
FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A")
FOREIGN_FORMS = ("20-F", "20-F/A", "40-F", "40-F/A")
SEAM_TOL = 0.01                 # două etichete pentru aceeași perioadă: diferență > 1% → conflict
TTM_CHECK_TOL = 0.005           # TTM la încheierea anului vs. valoarea anuală
Q_DAYS = (80, 100)
H_DAYS = (170, 190)
M9_DAYS = (260, 285)
FY_DAYS = (350, 380)
FY_END_TOL_DAYS = 7             # bilanțul „de la încheierea anului”: cel mult o săptămână distanță

# indicator → (unitate, [(taxonomie, etichetă), ...] în ordinea priorității), tip ("duration" / "instant")
CONCEPTS = {
    "revenue": ("USD", "duration", [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
        ("us-gaap", "SalesRevenueGoodsNet"),
    ]),
    "net_income": ("USD", "duration", [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic"),
        ("us-gaap", "ProfitLoss"),
    ]),
    "eps_diluted": ("USD/shares", "duration", [
        ("us-gaap", "EarningsPerShareDiluted"),
    ]),
    "shares_diluted": ("shares", "duration", [
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ]),
    "cfo": ("USD", "duration", [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    ]),
    "capex": ("USD", "duration", [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "PaymentsToAcquireProductiveAssets"),
    ]),
    "sbc": ("USD", "duration", [
        ("us-gaap", "ShareBasedCompensation"),
        ("us-gaap", "AllocatedShareBasedCompensationExpense"),
    ]),
    "dps": ("USD/shares", "duration", [
        ("us-gaap", "CommonStockDividendsPerShareDeclared"),
        ("us-gaap", "CommonStockDividendsPerShareCashPaid"),
    ]),
    "equity": ("USD", "instant", [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ]),
    "cash": ("USD", "instant", [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    ]),
    "debt_lt": ("USD", "instant", [
        ("us-gaap", "LongTermDebt"),
        ("us-gaap", "LongTermDebtNoncurrent"),
    ]),
    "shares_out": ("shares", "instant", [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ]),
}
DURATION_INDICATORS = tuple(k for k, v in CONCEPTS.items() if v[1] == "duration")
INSTANT_INDICATORS = tuple(k for k, v in CONCEPTS.items() if v[1] == "instant")
# nu se adună pe trimestre: EPS-ul anual ≠ suma EPS-urilor trimestriale când variază acțiunile; acțiunile sunt medii.
# Dividendul pe acțiune SE adună (plățile din an), deci nu e aici.
NON_ADDITIVE = ("eps_diluted", "shares_diluted")

_COLS = ["prio", "concept", "start", "end", "days", "val", "form", "filed", "accn"]


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _day(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def available_on(filed):
    """Ziua de la care o valoare depusă la `filed` e cunoscută: următoarea zi lucrătoare (luni–vineri).
    `companyfacts` nu are ora depunerii; multe depuneri sunt după închiderea bursei → conservator."""
    d = _day(filed)
    if d is None:
        return None
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def filer_kind(cf):
    """"10-K" (emitent american, US-GAAP), "20-F" (emitent străin: IFRS sau formulare 20-F/40-F) sau None (fără date)."""
    facts = (cf or {}).get("facts") or {}
    if "ifrs-full" in facts and "us-gaap" not in facts:
        return "20-F"
    forms = set()
    for concepts in (facts.get("us-gaap") or {}).values():
        for rows in (concepts.get("units") or {}).values():
            forms.update(r.get("form") for r in rows[:50])
        if len(forms) > 3:
            break
    if forms & set(FORMS):
        return "10-K"
    if forms & set(FOREIGN_FORMS):
        return "20-F"
    return None


def raw_facts(cf, indicator, asof=None):
    """Toate valorile brute ale indicatorului (toate etichetele lui), din 10-K/10-Q, ca DataFrame cu coloanele
    `prio, concept, start, end, days, val, form, filed, accn`. Cu `asof`: doar valorile cunoscute la acea dată."""
    unit, kind, concepts = CONCEPTS[indicator]
    facts = (cf or {}).get("facts") or {}
    asof_d = _day(asof) if asof is not None else None
    rows = []
    for prio, (tax, name) in enumerate(concepts):
        node = (facts.get(tax) or {}).get(name) or {}
        for r in (node.get("units") or {}).get(unit) or []:
            if r.get("form") not in FORMS:
                continue
            val, end, filed = _num(r.get("val")), _day(r.get("end")), _day(r.get("filed"))
            if val is None or end is None or filed is None:
                continue
            if asof_d is not None and available_on(filed) > asof_d:
                continue
            start = _day(r.get("start"))
            if kind == "duration" and start is None:
                continue
            days = (end - start).days + 1 if start else None
            rows.append((prio, name, start, end, days, val, r.get("form"), filed, r.get("accn")))
    return pd.DataFrame(rows, columns=_COLS)


def _latest(df):
    """Pentru fiecare cheie de perioadă: rândul din raportul cel mai recent (ultima versiune, inclusiv retratări)."""
    return df.sort_values(["filed", "accn"]).groupby(["prio", "start", "end"], dropna=False, as_index=False).last()


def _resolve(df):
    """Eticheta cu prioritatea cea mai mare pe fiecare (start, end) + marcajul de conflict între etichete."""
    if df.empty:
        return df.assign(conflict=pd.Series(dtype=bool))
    latest = _latest(df)
    out = []
    for (_, _), g in latest.groupby(["start", "end"], dropna=False):
        g = g.sort_values("prio")
        best = g.iloc[0].copy()
        others = g.iloc[1:]["val"]
        conflict = False
        for v in others:
            den = max(abs(best["val"]), abs(v))
            if den > 0 and abs(best["val"] - v) / den > SEAM_TOL:
                conflict = True
        best["conflict"] = conflict
        out.append(best)
    return pd.DataFrame(out).reset_index(drop=True)


def _in(days, rng):
    return days is not None and not (isinstance(days, float) and math.isnan(days)) and rng[0] <= days <= rng[1]


def annual(cf, indicator, asof=None):
    """Valorile anuale (exercițiu financiar de 350–380 de zile) ale unui indicator de flux, indexate pe data încheierii.
    Coloane: `val, start, filed, accn, concept, conflict`. Gol dacă nu există."""
    if CONCEPTS[indicator][1] != "duration":
        raise ValueError(f"{indicator}: indicator de bilanț; folosește balance_at_year_end()")
    res = _resolve(raw_facts(cf, indicator, asof))
    if res.empty:
        return _empty()
    fy = res[res["days"].apply(lambda d: _in(d, FY_DAYS))]
    return _frame(fy)


def _empty():
    return pd.DataFrame(columns=["val", "start", "filed", "accn", "concept", "conflict"]).rename_axis("end")


def _frame(df):
    if df.empty:
        return _empty()
    out = df.set_index("end")[["val", "start", "filed", "accn", "concept", "conflict"]].sort_index()
    return out[~out.index.duplicated(keep="last")]


def fiscal_year_ends(cf, asof=None):
    """Datele de încheiere a exercițiilor financiare (din profitul net sau cifra de afaceri anuală)."""
    ends = set()
    for ind in ("net_income", "revenue"):
        ends.update(annual(cf, ind, asof).index)
    return sorted(ends)


def instants(cf, indicator, asof=None):
    """Valorile de bilanț (la o dată), indexate pe dată, ultima versiune cunoscută la `asof`."""
    if CONCEPTS[indicator][1] != "instant":
        raise ValueError(f"{indicator}: indicator de flux; folosește annual() / quarterly()")
    df = raw_facts(cf, indicator, asof)
    if df.empty:
        return _empty()
    df = df.assign(start=pd.NaT)
    return _frame(_resolve(df))


def balance_at_year_end(cf, indicator, asof=None):
    """Valoarea de bilanț la fiecare încheiere de exercițiu (cel mult `FY_END_TOL_DAYS` distanță), pe datele anuale."""
    inst = instants(cf, indicator, asof)
    ends = fiscal_year_ends(cf, asof)
    rows = {}
    for e in ends:
        if inst.empty:
            break
        dist = [abs((d - e).days) for d in inst.index]
        i = min(range(len(dist)), key=dist.__getitem__)
        if dist[i] <= FY_END_TOL_DAYS:
            rows[e] = inst.iloc[i]
    if not rows:
        return _empty()
    out = pd.DataFrame(rows).T
    out.index.name = "end"
    return out


def quarterly(cf, indicator, asof=None):
    """Trimestrele unui indicator de flux, indexate pe data încheierii. Coloane: `val, start, filed, derived, conflict`.

    Surse, în ordine: valoarea trimestrială raportată direct (80–100 de zile); altfel diferența dintre două valori
    cumulate cu același început (T2 = 6 luni − 3 luni, T3 = 9 luni − 6 luni, T4 = an − 9 luni). Fluxul de numerar
    apare în 10-Q doar cumulat, deci trimestrele lui sunt mereu derivate. Indicatorii pe acțiune nu se derivează."""
    if CONCEPTS[indicator][1] != "duration":
        raise ValueError(f"{indicator}: indicator de bilanț")
    res = _resolve(raw_facts(cf, indicator, asof))
    cols = ["val", "start", "filed", "derived", "conflict"]
    if res.empty:
        return pd.DataFrame(columns=cols).rename_axis("end")
    q = {}
    for _, r in res[res["days"].apply(lambda d: _in(d, Q_DAYS))].iterrows():
        q[r["end"]] = {"val": r["val"], "start": r["start"], "filed": r["filed"], "derived": False,
                       "conflict": bool(r["conflict"])}
    if indicator not in NON_ADDITIVE:
        for _, g in res.groupby("start"):
            cum = g.sort_values("end")
            cum = cum[cum["days"].apply(lambda d: _in(d, Q_DAYS) or _in(d, H_DAYS) or _in(d, M9_DAYS)
                                        or _in(d, FY_DAYS))]
            prev = None
            for _, r in cum.iterrows():
                if prev is not None:
                    gap = (r["end"] - prev["end"]).days
                    if Q_DAYS[0] <= gap <= Q_DAYS[1] and r["end"] not in q:
                        q[r["end"]] = {"val": r["val"] - prev["val"], "start": prev["end"] + timedelta(days=1),
                                       "filed": max(r["filed"], prev["filed"]), "derived": True,
                                       "conflict": bool(r["conflict"] or prev["conflict"])}
                prev = r
    if not q:
        return pd.DataFrame(columns=cols).rename_axis("end")
    out = pd.DataFrame.from_dict(q, orient="index")[cols].sort_index()
    out.index.name = "end"
    return out


def ttm(cf, indicator, asof=None):
    """Suma ultimelor 4 trimestre consecutive (TTM): {val, end, start, filed, check} sau None.
    `check` = diferența relativă față de valoarea anuală când TTM se încheie la sfârșitul exercițiului (altfel None)."""
    if indicator in NON_ADDITIVE:
        return None
    q = quarterly(cf, indicator, asof)
    if len(q) < 4:
        return None
    last4 = q.iloc[-4:]
    ends = list(last4.index)
    for a, b in zip(ends, ends[1:]):
        if not (Q_DAYS[0] <= (b - a).days <= Q_DAYS[1]):
            return None
    span = (ends[-1] - last4["start"].iloc[0]).days + 1
    if not _in(span, FY_DAYS) or last4["conflict"].any():
        return None
    val = float(last4["val"].sum())
    fy = annual(cf, indicator, asof)
    check = None
    if ends[-1] in fy.index:
        a = fy.loc[ends[-1], "val"]
        check = abs(val - a) / abs(a) if a else None
    return {"val": val, "end": ends[-1], "start": last4["start"].iloc[0], "filed": max(last4["filed"]), "check": check}


def split_factor_after(splits, when):
    """Produsul split-urilor de după `when` (2.0 = 2 la 1). `splits`: Series (data → raport), ca în yfinance."""
    if splits is None or len(splits) == 0:
        return 1.0
    s = pd.Series(splits).dropna()
    idx = pd.DatetimeIndex(s.index)
    s.index = idx.tz_localize(None) if idx.tz is not None else idx
    after = s[(s.index > pd.Timestamp(when)) & (s > 0)]
    return float(after.prod()) if len(after) else 1.0


def annual_table(cf, asof=None, splits=None):
    """Tabelul anual (un rând pe exercițiu, indexat pe data încheierii), în USD și pe acțiune.

    Coloane: revenue, net_income, cfo, capex, fcf, sbc, equity, cash, debt_lt, shares_diluted (ajustate cu split-urile
    de după exercițiu), eps (= profit net / acțiuni diluate ajustate), eps_reported (ajustat), eps_check (diferența
    relativă), revenue_ps, fcf_ps, bvps, dps (ajustat), filed (cel mai recent raport folosit pe rând), conflicts (lista
    indicatorilor cu etichete în conflict). Lipsă = NaN."""
    ends = fiscal_year_ends(cf, asof)
    if not ends:
        return pd.DataFrame()
    t = pd.DataFrame(index=pd.Index(ends, name="end"))
    filed = pd.Series(pd.NaT, index=t.index, dtype="object")
    conflicts = {e: [] for e in ends}

    def put(name, df):
        t[name] = [df.loc[e, "val"] if e in df.index else float("nan") for e in ends]
        for e in ends:
            if e in df.index:
                f = df.loc[e, "filed"]
                filed[e] = f if pd.isna(filed[e]) else max(filed[e], f)
                if bool(df.loc[e, "conflict"]):
                    conflicts[e].append(name)

    for ind in ("revenue", "net_income", "cfo", "capex", "sbc", "eps_diluted", "shares_diluted", "dps"):
        put(ind, annual(cf, ind, asof))
    for ind in ("equity", "cash", "debt_lt"):
        put(ind, balance_at_year_end(cf, ind, asof))
    # dividendul anual lipsă (unele companii îl raportează doar pe trimestru): suma celor 4 trimestre din exercițiu
    dps_q = None
    for e in ends:
        if not pd.isna(t.loc[e, "dps"]):
            continue
        dps_q = quarterly(cf, "dps", asof) if dps_q is None else dps_q
        inside = dps_q[(dps_q.index > e - timedelta(days=FY_DAYS[1])) & (dps_q.index <= e)]
        if len(inside) == 4 and not inside["conflict"].any():
            t.loc[e, "dps"] = float(inside["val"].sum())

    t["capex"] = t["capex"].abs()                       # ieșire de numerar, pozitivă (unele companii o raportează cu minus)
    t["fcf"] = t["cfo"] - t["capex"]
    factor = pd.Series([split_factor_after(splits, e) for e in ends], index=t.index)
    t["shares_diluted"] = t["shares_diluted"] * factor
    t["eps_reported"] = t.pop("eps_diluted") / factor
    t["dps"] = t["dps"] / factor
    sh = t["shares_diluted"].where(t["shares_diluted"] > 0)
    t["eps"] = t["net_income"] / sh
    t["eps_check"] = ((t["eps"] - t["eps_reported"]).abs() / t["eps_reported"].abs()).where(t["eps_reported"] != 0)
    t["revenue_ps"] = t["revenue"] / sh
    t["fcf_ps"] = t["fcf"] / sh
    t["bvps"] = t["equity"] / sh
    t["filed"] = filed
    t["conflicts"] = [conflicts[e] for e in ends]
    return t


def consistency(series):
    """(ani cu creștere față de anul anterior, ani comparați) pe o serie anuală ordonată; perechile cu lipsă se sar."""
    vals = [(_num(v)) for v in series]
    pairs = [(a, b) for a, b in zip(vals, vals[1:]) if a is not None and b is not None]
    return sum(1 for a, b in pairs if b > a), len(pairs)
