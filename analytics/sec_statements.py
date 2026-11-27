"""Adaptor SEC → situații anuale în formatul Yahoo, pentru testul istoric al scorului SUA (Faza 3c, decizia E4).

Din faptele XBRL „cunoscute la t” (`analytics.xbrl_facts`) se construiesc trei DataFrame-uri cu rândurile pe care le
citesc funcțiile aplicației (`analytics.fundamentals`: DCF, Piotroski, Altman, îndatorare) și coloanele = anii fiscali
(cel mai recent primul, ca în Yahoo). Astfel scorul se calculează cu ACELEAȘI funcții ca în aplicație.
Funcții pure: fără rețea, fără Streamlit.

Definiții (spuse în raportul testului):
  * Totalurile din `annual_table` (venit, profit, CFO, capex cu software capitalizat, acțiuni diluate ajustate cu
    split-urile de după depunere și corectate de scară, EPS raportat ajustat, capitaluri, numerar).
  * EBIT = profit înainte de impozit + cheltuieli cu dobânzile (definiția Yahoo). Fără una dintre ele, rândul EBIT
    lipsește și funcțiile aplicației trec la „Operating Income” (`OperatingIncomeLoss`), ca la Yahoo.
  * EBITDA = EBIT (sau, fără el, profitul din exploatare) + amortizare (`DepreciationDepletionAndAmortization`…).
  * Marja brută: `GrossProfit`; fără ea, venit − cost al vânzărilor (marcat `gross_profit_proxy`).
  * Datorii totale: `Liabilities`; fără ea, (datorii + capitaluri) − capitaluri inclusiv minoritarii (marcat).
  * Datoria totală = datoria curentă + cea pe termen lung; altfel `LongTermDebt` (include partea curentă) +
    împrumuturile pe termen scurt (`ShortTermBorrowings`, altfel `CommercialPaper`) etichetate. Datoria curentă =
    `DebtCurrent`; altfel partea curentă a datoriei pe termen lung (etichetată sau `LongTermDebt` − partea pe termen
    lung) + împrumuturile pe termen scurt etichetate. **[presupunere]** o linie de bilanț neetichetată nu există (orice
    linie de pe fața bilanțului se etichetează în XBRL); etichetele proprii ale companiei NU sunt citite → datoria poate
    fi subestimată la cei care le folosesc. Leasingurile financiare nu sunt incluse (Yahoo le include în „Total Debt”).
  * Numerar + plasamente pe termen scurt: doar când plasamentele sunt etichetate; altfel doar numerarul.
  * Acțiunile de la data bilanțului: `CommonStockSharesOutstanding`, ajustate cu split-urile de după depunere și
    corectate de scară față de acțiunile diluate (rapoarte vechi în mii).
  * Valorile la o cusătură de etichete în conflict se păstrează; anii lor sunt listați în `notes["conflicts"]`.
Lipsă = NaN, niciodată 0.

Incremental (decizia E6, corectată 27.11.2026): situațiile „cunoscute la t” se schimbă doar la datele la care apare
o valoare nouă sau schimbată dintre cele citite. `change_dates` le găsește (toate rapoartele 10-K; din 10-Q doar
bilanțurile care pot schimba un an fiscal și dividendele trimestriale când lipsesc cele anuale), `asof_versions`
calculează o versiune pe dată, `version_at` ia ultima versiune ≤ t. Rezultat identic cu apelul direct
(`statements_from_facts(cf, asof=t)`), verificat în teste.
"""
import bisect
import math

import pandas as pd

from analytics import xbrl_facts as xf

# indicatorii citiți de `statements_from_facts` (cei din `annual_table` + cei de bilanț și rezultat ai adaptorului)
SOURCE_INDICATORS = (
    "revenue", "net_income", "net_income_common", "cfo", "capex", "capex_software", "sbc", "eps_diluted",
    "shares_diluted", "dps", "shares_basic", "eps_basic", "profit_loss", "nci", "nci_redeemable", "nci_nonredeemable",
    "equity", "cash", "debt_lt",
) + tuple(xf.STATEMENT_CONCEPTS)

INCOME_ROWS = ("Total Revenue", "Net Income", "Net Income Common Stockholders", "Diluted Average Shares",
               "Diluted EPS", "Gross Profit", "Operating Income", "Pretax Income", "Tax Provision",
               "Interest Expense", "EBIT", "EBITDA")
BALANCE_ROWS = ("Total Assets", "Current Assets", "Current Liabilities", "Total Liabilities Net Minority Interest",
                "Stockholders Equity", "Retained Earnings", "Inventory", "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments", "Current Debt", "Long Term Debt", "Total Debt",
                "Ordinary Shares Number")
CASHFLOW_ROWS = ("Operating Cash Flow", "Capital Expenditure", "Free Cash Flow")


def _v(x):
    """float finit sau None."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _add(*xs):
    vals = [_v(x) for x in xs]
    return None if any(v is None for v in vals) else sum(vals)


def _sub(a, b):
    a, b = _v(a), _v(b)
    return None if a is None or b is None else a - b


def _frame(rows, names, ends):
    """DataFrame rânduri × ani (cel mai recent primul); rândurile fără nicio valoare se scot."""
    cols = [pd.Timestamp(e) for e in sorted(ends, reverse=True)]
    data = {n: [rows.get(n, {}).get(e) for e in sorted(ends, reverse=True)] for n in names}
    df = pd.DataFrame.from_dict(data, orient="index", columns=cols, dtype=float)
    return df.dropna(how="all")


def statements_from_facts(cf, asof=None, splits=None):
    """Situațiile anuale cunoscute la `asof`, în formatul Yahoo.

    Întoarce {"income", "balance", "cashflow": DataFrame, "notes": {marcaj: [ani]}} sau None (niciun an fiscal).
    `splits`: Series yfinance (data → raport), pentru a aduce acțiunile pe baza de azi (ca prețul Yahoo)."""
    t = xf.annual_table(cf, asof, splits)
    if t.empty:
        return None
    ends = list(t.index)

    def ann(ind):
        df = xf.annual(cf, ind, asof)
        return ({e: _v(df.loc[e, "val"]) for e in ends if e in df.index},
                {e for e in ends if e in df.index and bool(df.loc[e, "conflict"])})

    def bal(ind):
        df = xf.balance_at_year_end(cf, ind, asof, ends)
        return ({e: _v(df.loc[e, "val"]) for e in ends if e in df.index},
                {e: df.loc[e, "filed"] for e in ends if e in df.index},
                {e for e in ends if e in df.index and bool(df.loc[e, "conflict"])})

    notes = {"conflicts": [], "gross_profit_proxy": [], "liabilities_proxy": [], "ebitda_on_operating_income": [],
             "current_debt_derived": []}
    conflicted = set(e for e in ends if t.loc[e, "conflicts"])
    flows = {}
    for ind in ("operating_income", "gross_profit", "cost_of_revenue", "interest_expense", "pretax_income",
                "tax_provision", "dna"):
        flows[ind], c = ann(ind)
        conflicted |= c
    stock = {}
    for ind in ("assets", "assets_current", "liabilities_current", "liabilities", "liabilities_and_equity",
                "equity_total", "retained_earnings", "inventory", "st_investments", "debt_current", "ltd_current",
                "st_borrowings", "commercial_paper", "ltd_noncurrent", "shares_out_bs"):
        vals, filed, c = bal(ind)
        stock[ind] = (vals, filed)
        conflicted |= c

    inc, bs, cfl = {}, {}, {}
    col = lambda name: {e: _v(t.loc[e, name]) for e in ends}            # noqa: E731
    inc["Total Revenue"] = col("revenue")
    inc["Net Income"] = col("net_income")
    inc["Net Income Common Stockholders"] = col("net_income_common")
    inc["Diluted Average Shares"] = col("shares_diluted")
    inc["Diluted EPS"] = col("eps_reported")
    inc["Operating Income"] = flows["operating_income"]
    inc["Pretax Income"] = flows["pretax_income"]
    inc["Tax Provision"] = flows["tax_provision"]
    interest = {e: abs(v) for e, v in flows["interest_expense"].items() if v is not None}
    inc["Interest Expense"] = interest
    gp, ebit, ebitda = {}, {}, {}
    for e in ends:
        g = flows["gross_profit"].get(e)
        if g is None:
            g = _sub(inc["Total Revenue"].get(e), flows["cost_of_revenue"].get(e))
            if g is not None:
                notes["gross_profit_proxy"].append(e)
        gp[e] = g
        ebit[e] = _add(flows["pretax_income"].get(e), interest.get(e))
        base = ebit[e]
        if base is None and flows["operating_income"].get(e) is not None:
            base = flows["operating_income"][e]
        ebitda[e] = _add(base, flows["dna"].get(e))
        if ebitda[e] is not None and ebit[e] is None:
            notes["ebitda_on_operating_income"].append(e)
    inc["Gross Profit"], inc["EBIT"], inc["EBITDA"] = gp, ebit, ebitda

    s = {k: v[0] for k, v in stock.items()}
    bs["Total Assets"] = s["assets"]
    bs["Current Assets"] = s["assets_current"]
    bs["Current Liabilities"] = s["liabilities_current"]
    bs["Stockholders Equity"] = col("equity")
    bs["Retained Earnings"] = s["retained_earnings"]
    bs["Inventory"] = s["inventory"]
    bs["Cash And Cash Equivalents"] = col("cash")
    liab, cash_sti, cur_debt, lt_debt, tot_debt, sh_out = {}, {}, {}, {}, {}, {}
    debt_lt = col("debt_lt")
    for e in ends:
        li = s["liabilities"].get(e)
        if li is None:
            li = _sub(s["liabilities_and_equity"].get(e), s["equity_total"].get(e))
            if li is not None:
                notes["liabilities_proxy"].append(e)
        liab[e] = li
        sti = s["st_investments"].get(e)
        cash_sti[e] = _add(bs["Cash And Cash Equivalents"].get(e), sti) if sti is not None else None
        stb = s["st_borrowings"].get(e)
        short = stb if stb is not None else s["commercial_paper"].get(e)
        ltd_cur = s["ltd_current"].get(e)
        if ltd_cur is None and debt_lt.get(e) is not None and s["ltd_noncurrent"].get(e) is not None:
            d = debt_lt[e] - s["ltd_noncurrent"][e]
            ltd_cur = d if d >= 0 else None
        if s["debt_current"].get(e) is not None:
            cur_debt[e] = s["debt_current"][e]
        elif ltd_cur is not None or short is not None:
            cur_debt[e] = (ltd_cur or 0.0) + (short or 0.0)
            notes["current_debt_derived"].append(e)
        else:
            cur_debt[e] = None
        lt = s["ltd_noncurrent"].get(e)
        if lt is None and debt_lt.get(e) is not None and ltd_cur is not None and debt_lt[e] - ltd_cur >= 0:
            lt = debt_lt[e] - ltd_cur
        lt_debt[e] = lt
        # datoria curentă + pe termen lung; altfel LongTermDebt (cu partea curentă) + împrumuturile pe termen scurt
        tot = _add(cur_debt[e], lt)
        if tot is None and debt_lt.get(e) is not None:
            tot = debt_lt[e] + (short or 0.0)
        tot_debt[e] = tot
        # acțiunile de la data bilanțului: split-urile de după depunere + scara față de acțiunile diluate (ajustate)
        raw = s["shares_out_bs"].get(e)
        if raw is not None and raw > 0:
            adj = raw * xf.split_factor_after(splits, stock["shares_out_bs"][1].get(e))
            dil = inc["Diluted Average Shares"].get(e)
            if dil is not None and dil > 0:
                adj *= xf.scale_correction(adj, dil)
            sh_out[e] = adj
    bs["Total Liabilities Net Minority Interest"] = liab
    bs["Cash Cash Equivalents And Short Term Investments"] = cash_sti
    bs["Current Debt"], bs["Long Term Debt"], bs["Total Debt"] = cur_debt, lt_debt, tot_debt
    bs["Ordinary Shares Number"] = sh_out

    cfl["Operating Cash Flow"] = col("cfo")
    cfl["Capital Expenditure"] = {e: (-v if v is not None else None) for e, v in col("capex").items()}
    cfl["Free Cash Flow"] = col("fcf")

    notes["conflicts"] = sorted(conflicted)
    return {"income": _frame(inc, INCOME_ROWS, ends), "balance": _frame(bs, BALANCE_ROWS, ends),
            "cashflow": _frame(cfl, CASHFLOW_ROWS, ends), "notes": notes}


# --- Incremental: o versiune pe fiecare dată la care se poate schimba ceva -------------------------------------

def change_dates(cf, indicators=SOURCE_INDICATORS):
    """Datele (`available_on`) la care situațiile „cunoscute la t” se pot schimba, crescător.

    * toate depunerile 10-K / 10-K/A ale indicatorilor citiți;
    * din 10-Q: valorile de bilanț noi sau schimbate (retratări în coloanele comparative) — la indicatorii cu o
      singură etichetă doar cele de lângă o încheiere de exercițiu, la cei cu mai multe etichete toate (eticheta
      perioadei celei mai noi decide ancora, vezi `xf._resolve`);
    * din 10-Q: dividendele trimestriale noi sau schimbate, doar dacă vreun an nu are dividendul anual.
    Valorile de flux din 10-Q nu intră în anual (`xf.annual` citește doar 10-K)."""
    facts = (cf or {}).get("facts") or {}
    ends = xf.fiscal_year_ends(cf)
    dps_years = set(xf.annual(cf, "dps").index)
    use_q_dps = any(e not in dps_years for e in ends)
    tol = xf.FY_END_TOL_DAYS
    dates, tracked = set(), []
    for ind in indicators:
        unit, kind, concepts = xf.CONCEPTS[ind]
        multi = len(concepts) > 1
        for tax, name in concepts:
            for r in ((facts.get(tax) or {}).get(name) or {}).get("units", {}).get(unit) or []:
                form = r.get("form")
                if form not in xf.FORMS:
                    continue
                filed, end, val = xf._day(r.get("filed")), xf._day(r.get("end")), xf._num(r.get("val"))
                if filed is None or end is None or val is None:
                    continue
                if form in xf.ANNUAL_FORMS:
                    dates.add(xf.available_on(filed))
                if kind == "instant":
                    if multi or any(abs((end - e).days) <= tol for e in ends):
                        tracked.append(((name, None, end), filed, form, val))
                elif ind == "dps" and use_q_dps:
                    tracked.append(((name, xf._day(r.get("start")), end), filed, form, val))
    last = {}
    for key, filed, form, val in sorted(tracked, key=lambda x: (x[1], x[2] in xf.ANNUAL_FORMS)):
        if key not in last or last[key] != val:
            dates.add(xf.available_on(filed))
        last[key] = val
    return sorted(dates)


def asof_versions(cf, splits=None, dates=None):
    """[(dată, situații)] crescător: situațiile cunoscute de la fiecare dată din `change_dates` (sau `dates`)."""
    out = []
    for d in (change_dates(cf) if dates is None else dates):
        st = statements_from_facts(cf, asof=d, splits=splits)
        if st is not None:
            out.append((pd.Timestamp(d), st))
    return out


def version_at(versions, when):
    """Situațiile cunoscute la `when` (ultima versiune cu data ≤ `when`) sau None."""
    keys = [d for d, _ in versions]
    i = bisect.bisect_right(keys, pd.Timestamp(when)) - 1
    return versions[i][1] if i >= 0 else None


__all__ = ["statements_from_facts", "change_dates", "asof_versions", "version_at", "SOURCE_INDICATORS"]
