"""Indicatori fundamentali calculați din situațiile financiare. Fără Streamlit, fără rețea.

Intrările sunt DataFrame-urile yfinance (`income_stmt`, `balance_sheet`, `cashflow`,
anuale și trimestriale): indicatorii pe rânduri, datele de raportare pe coloane.
Ordinea coloanelor nu e presupusă: funcțiile sortează singure după dată.

Regula modulului: o intrare lipsă dă None, niciodată 0 sau altă valoare plauzibilă.
Funcțiile care pot eșua din mai multe motive întorc și motivul, ca interfața să-l afișeze.
"""
import math

import pandas as pd

# --- Nume de rânduri (yfinance le schimbă între companii și versiuni) --------

CFO_ROWS = ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
            "Total Cash From Operating Activities")
CAPEX_ROWS = ("Capital Expenditure", "Capital Expenditures", "Purchase Of PPE")
TOTAL_DEBT_ROWS = ("Total Debt",)
LONG_DEBT_ROWS = ("Long Term Debt", "Long Term Debt And Capital Lease Obligation")
SHORT_DEBT_ROWS = ("Current Debt", "Current Debt And Capital Lease Obligation")
CASH_ROWS = ("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
DILUTED_SHARES_ROWS = ("Diluted Average Shares",)
BASIC_SHARES_ROWS = ("Ordinary Shares Number", "Share Issued", "Basic Average Shares")
INTEREST_ROWS = ("Interest Expense", "Interest Expense Non Operating")
TAX_ROWS = ("Tax Provision",)
PRETAX_ROWS = ("Pretax Income",)

# Limite metodologice (instrucțiunile proiectului)
MAX_TERMINAL_GROWTH = 0.03      # g terminal ≤ 3%
MIN_SPREAD_STABLE = 0.02        # sub 2 pp între r și g, valoarea terminală e instabilă
MAX_TAX_RATE = 0.35             # cotă efectivă de impozit peste 35% = an atipic; se plafonează
DEBT_SPREAD_PROXY = 0.015       # costul datoriei când dobânda nu e raportată: rf + 1,5 pp (proxy)
MAX_DEBT_SPREAD = 0.10          # cost al datoriei peste rf + 10 pp = dată nerealistă, se folosește proxy-ul
DEFAULT_ERP = 0.05              # prima de risc a pieței de acțiuni; ipoteză, afișată în interfață

# Simboluri BVB din sectorul financiar. Yahoo nu trimite `sector` pentru multe simboluri .RO,
# deci excluderea după sector nu le-ar prinde. Listă fixă: se revizuiește la listări noi.
BVB_FINANCIALS = frozenset({
    "TLV", "BRD", "PBK",                              # bănci
    "FP", "EVER", "LION", "TRANSI", "LONG", "INFINITY",  # fonduri / foste SIF-uri
    "BRK", "BVB",                                     # broker, operator de piață
})


# --- Citire sigură -----------------------------------------------------------

def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sorted_desc(df):
    """Coloanele ordonate de la cea mai recentă raportare la cea mai veche."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    try:
        cols = pd.to_datetime(df.columns)
    except (TypeError, ValueError):
        return df
    order = cols.argsort()[::-1]
    out = df.iloc[:, order]
    return out


def stmt_value(df, names, col=0):
    """Prima valoare disponibilă dintre rândurile `names`, pe coloana `col`
    (0 = cea mai recentă raportare). None dacă rândul sau valoarea lipsește."""
    df = _sorted_desc(df)
    if df is None or col >= df.shape[1]:
        return None
    for name in names:
        if name in df.index:
            try:
                value = float(df.loc[name].iloc[col])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def stmt_date(df, col=0):
    """Data raportării de pe coloana `col` (0 = cea mai recentă) sau None."""
    df = _sorted_desc(df)
    if df is None or col >= df.shape[1]:
        return None
    try:
        return pd.Timestamp(df.columns[col])
    except (TypeError, ValueError):
        return None


def ttm_sum(quarterly_df, names):
    """Suma ultimelor 4 trimestre (TTM) pentru primul rând găsit din `names`.

    Întoarce None dacă nu există 4 trimestre consecutive, toate cu valoare.
    Consecutive = cele 4 date de raportare acoperă 8–10 luni între prima și ultima;
    așa sunt respinse raportările semestriale (frecvente în UE și la BVB) și golurile.
    """
    df = _sorted_desc(quarterly_df)
    if df is None or df.shape[1] < 4:
        return None
    try:
        dates = pd.to_datetime(df.columns[:4])
    except (TypeError, ValueError):
        return None
    span_days = (dates[0] - dates[3]).days
    if not 240 <= span_days <= 310:
        return None
    for name in names:
        if name in df.index:
            values = pd.to_numeric(df.loc[name].iloc[:4], errors="coerce")
            if values.notna().all():
                return float(values.sum())
    return None


# --- Free cash flow ----------------------------------------------------------

def free_cash_flow(cfo, capex):
    """FCF = flux de numerar din exploatare − |cheltuieli de capital|.

    yfinance raportează capex negativ, alte surse pozitiv: modulul face formula
    independentă de semn. Dacă una dintre intrări lipsește, rezultatul e None
    (un capex lipsă NU se consideră zero).
    """
    if not _is_num(cfo) or not _is_num(capex):
        return None
    return cfo - abs(capex)


def fcf_ttm(quarterly_cashflow):
    """FCF pe ultimele 4 trimestre. None dacă lipsesc trimestre."""
    return free_cash_flow(ttm_sum(quarterly_cashflow, CFO_ROWS), ttm_sum(quarterly_cashflow, CAPEX_ROWS))


def fcf_history(annual_cashflow):
    """FCF anual, de la cel mai vechi an la cel mai recent (Series indexată pe data raportării).

    Anii fără CFO sau fără capex sunt omiși, nu completați.
    """
    df = _sorted_desc(annual_cashflow)
    if df is None:
        return pd.Series(dtype=float)
    out = {}
    for col in range(df.shape[1]):
        value = free_cash_flow(stmt_value(df, CFO_ROWS, col), stmt_value(df, CAPEX_ROWS, col))
        if value is not None:
            out[pd.Timestamp(df.columns[col])] = value
    return pd.Series(out, dtype=float).sort_index()


def fcf_cagr(history):
    """Rata anuală compusă a FCF între primul și ultimul an din `history`.

    None dacă sunt mai puțin de 2 ani sau dacă unul dintre capete e ≤ 0
    (CAGR nu are sens peste o schimbare de semn).
    """
    if history is None or len(history) < 2:
        return None
    first, last = float(history.iloc[0]), float(history.iloc[-1])
    years = (history.index[-1] - history.index[0]).days / 365.25
    if first <= 0 or last <= 0 or years <= 0:
        return None
    return (last / first) ** (1 / years) - 1


def fcf_average(history, years=3):
    """Media FCF pe ultimii `years` ani fiscali (FCF normalizat). None sub 2 ani de date.

    Folosit când FCF-ul curent e deformat de un vârf de investiții sau de un an atipic:
    un singur an slab, proiectat la nesfârșit, subevaluează compania (și invers).
    """
    if history is None or len(history) < 2:
        return None
    return float(history.iloc[-years:].mean())


def fcf_distortion_warning(cfo, capex, fcf, average):
    """Mesaj când FCF-ul curent nu e o bază bună de proiecție, altfel None.

    Două semnale: capex-ul consumă peste 75% din fluxul din exploatare (ciclu de
    investiții), sau FCF-ul curent e sub jumătate / peste dublul mediei multianuale.
    """
    if _is_num(cfo) and _is_num(capex) and cfo > 0 and abs(capex) / cfo > 0.75:
        return (f"Capex-ul consumă {abs(capex) / cfo * 100:.0f}% din fluxul de numerar din exploatare: "
                "compania e într-un ciclu de investiții, iar FCF-ul curent subestimează capacitatea ei normală.")
    if _is_num(fcf) and _is_num(average) and average > 0 and fcf > 0:
        ratio = fcf / average
        if ratio < 0.5 or ratio > 2.0:
            return (f"FCF-ul curent este {ratio * 100:.0f}% din media ultimilor ani: "
                    "un an atipic proiectat la nesfârșit deformează valoarea.")
    return None


# --- Datorie netă și acțiuni -------------------------------------------------

def total_debt(balance, col=0):
    """Datoria totală purtătoare de dobândă. Dacă rândul `Total Debt` lipsește,
    se adună datoria pe termen lung și cea curentă, doar când ambele există."""
    value = stmt_value(balance, TOTAL_DEBT_ROWS, col)
    if value is not None:
        return value
    long_d = stmt_value(balance, LONG_DEBT_ROWS, col)
    short_d = stmt_value(balance, SHORT_DEBT_ROWS, col)
    if long_d is not None and short_d is not None:
        return long_d + short_d
    return None


def net_debt(balance, col=0):
    """Datorie netă = datorie totală − numerar, echivalente și plasamente pe termen scurt.

    Negativă când compania are numerar net. None dacă lipsește datoria sau numerarul.
    """
    debt = total_debt(balance, col)
    cash = stmt_value(balance, CASH_ROWS, col)
    if debt is None or cash is None:
        return None
    return debt - cash


def diluted_shares(income, balance=None):
    """(număr de acțiuni, este_diluat).

    Preferă media diluată din contul de profit și pierdere. Dacă lipsește, folosește
    numărul de acțiuni din bilanț și întoarce este_diluat=False, ca interfața să spună
    că valoarea pe acțiune nu include diluarea. (None, False) dacă nu există niciunul.
    """
    value = stmt_value(income, DILUTED_SHARES_ROWS)
    if value is not None and value > 0:
        return value, True
    for source in (balance, income):
        value = stmt_value(source, BASIC_SHARES_ROWS)
        if value is not None and value > 0:
            return value, False
    return None, False


# --- Sectoare la care DCF pe FCF nu se aplică --------------------------------

def dcf_not_applicable_reason(sector, industry, symbol=None):
    """Motivul pentru care DCF pe FCF nu se aplică, sau None dacă se aplică.

    Bănci, asigurători și restul sectorului financiar: datoria e materie primă, nu
    finanțare, iar „fluxul de numerar din exploatare" nu măsoară ce măsoară la o companie
    industrială. REIT-uri: se evaluează pe FFO. Pentru simbolurile .RO fără sector în
    Yahoo se folosește lista `BVB_FINANCIALS`.
    """
    sector_l = (sector or "").strip().lower()
    industry_l = (industry or "").strip().lower()
    if sector_l in ("financial services", "financial", "financials"):
        return "Sector financiar: DCF pe free cash flow nu se aplică. Folosește P/BV raportat la ROE."
    if "reit" in industry_l:
        return "REIT: DCF pe free cash flow nu se aplică. Se evaluează pe FFO."
    if symbol:
        sym = str(symbol).upper()
        if sym.endswith(".RO") and sym[:-3] in BVB_FINANCIALS:
            return "Emitent financiar BVB: DCF pe free cash flow nu se aplică. Folosește P/BV raportat la ROE."
    return None


# --- Costul capitalului ------------------------------------------------------

def cost_of_equity(risk_free, beta, erp=DEFAULT_ERP):
    """CAPM: ke = rf + β × ERP. Toate ca fracții (0,04 = 4%). None dacă lipsește rf sau β."""
    if not _is_num(risk_free) or not _is_num(beta) or not _is_num(erp):
        return None
    return risk_free + beta * erp


def effective_tax_rate(income, col=0):
    """Impozit / profit înainte de impozitare, plafonat la [0, 35%].
    None dacă lipsesc datele sau profitul brut e ≤ 0."""
    tax = stmt_value(income, TAX_ROWS, col)
    pretax = stmt_value(income, PRETAX_ROWS, col)
    if tax is None or pretax is None or pretax <= 0:
        return None
    return min(max(tax / pretax, 0.0), MAX_TAX_RATE)


def wacc(risk_free, beta, market_cap, debt, interest_expense=None, tax_rate=None, erp=DEFAULT_ERP):
    """Costul mediu ponderat al capitalului. Întoarce un dict sau None.

    WACC = E/(D+E) × ke + D/(D+E) × kd × (1 − t)
      ke = rf + β × ERP (CAPM)
      kd = |cheltuieli cu dobânzile| / datorie; dacă dobânda lipsește sau raportul e nerealist: rf + 1,5 pp (proxy)
      t  = cota efectivă de impozit; dacă lipsește: 0 (fără scut fiscal, variantă prudentă)

    None dacă lipsesc rf, β sau capitalizarea: fără ele nu există cost al capitalului
    propriu, iar o valoare presupusă ar arăta ca una calculată. `notes` listează fiecare
    aproximare folosită.
    """
    ke = cost_of_equity(risk_free, beta, erp)
    if ke is None or not _is_num(market_cap) or market_cap <= 0:
        return None
    notes = []
    if not _is_num(debt) or debt <= 0:
        if not _is_num(debt):
            notes.append("Datoria lipsește din bilanț: WACC = costul capitalului propriu.")
        return {"wacc": ke, "ke": ke, "kd": None, "tax": None, "w_e": 1.0, "w_d": 0.0,
                "rf": risk_free, "beta": beta, "erp": erp, "notes": notes}

    kd = None
    if _is_num(interest_expense) and interest_expense != 0:
        implied = abs(interest_expense) / debt
        # Plauzibil = între jumătate din rata fără risc și rf + 10 pp. În afara intervalului,
        # „dobânda" raportată include de regulă alte costuri financiare (actualizarea
        # provizioanelor, leasing, diferențe de curs) sau datoria s-a schimbat mult în an.
        if 0.5 * risk_free <= implied <= risk_free + MAX_DEBT_SPREAD:
            kd = implied
        else:
            notes.append(f"Dobânda raportată / datorie = {implied * 100:.1f}%, nerealist "
                         "(include probabil alte costuri financiare): costul datoriei = rf + 1,5 pp (proxy).")
    else:
        notes.append("Cheltuiala cu dobânzile nu e raportată: costul datoriei = rf + 1,5 pp (proxy).")
    if kd is None:
        kd = risk_free + DEBT_SPREAD_PROXY
    if _is_num(tax_rate):
        tax = min(max(tax_rate, 0.0), MAX_TAX_RATE)
    else:
        tax = 0.0
        notes.append("Cota de impozit lipsește: fără scut fiscal (t = 0).")
    w_e = market_cap / (market_cap + debt)
    w_d = 1.0 - w_e
    value = w_e * ke + w_d * kd * (1.0 - tax)
    return {"wacc": value, "ke": ke, "kd": kd, "tax": tax, "w_e": w_e, "w_d": w_d,
            "rf": risk_free, "beta": beta, "erp": erp, "notes": notes}


# --- DCF ---------------------------------------------------------------------

def growth_path(growth, terminal_growth, years=5):
    """Ratele de creștere pe anii 1..years: pornesc de la `growth` și scad liniar
    până la `terminal_growth` în ultimul an. Un singur an = rata inițială."""
    if years < 1:
        return []
    if years == 1:
        return [growth]
    step = (terminal_growth - growth) / (years - 1)
    return [growth + step * i for i in range(years)]


def dcf_fcf(fcf0, growth, discount_rate, terminal_growth, net_debt_value, shares, years=5):
    """DCF pe free cash flow, în două etape. Toate ratele sunt fracții (0,09 = 9%).

    1. FCF proiectat `years` ani; creșterea scade liniar de la `growth` la `terminal_growth`.
    2. Valoare terminală Gordon: TV = FCF_n × (1 + g) / (r − g).
    3. Valoarea întreprinderii = Σ FCF_t / (1 + r)^t + TV / (1 + r)^n.
    4. Valoarea capitalului = valoarea întreprinderii − datoria netă.
    5. Valoare pe acțiune = valoarea capitalului / număr de acțiuni (diluat).

    Întoarce un dict cu `per_share` (None când modelul nu se poate aplica), `reason`
    (de ce), `warnings` și componentele calculului. Nu aruncă excepții pe date lipsă.
    """
    result = {"per_share": None, "reason": None, "warnings": [], "enterprise_value": None,
              "equity_value": None, "pv_fcf": None, "pv_terminal": None, "terminal_share": None,
              "flows": [], "growth_path": []}

    if not _is_num(fcf0):
        result["reason"] = "Free cash flow indisponibil (lipsește CFO sau capex)."
        return result
    if fcf0 <= 0:
        result["reason"] = "Free cash flow negativ sau zero: DCF pe FCF nu se aplică."
        return result
    if not _is_num(shares) or shares <= 0:
        result["reason"] = "Numărul de acțiuni lipsește."
        return result
    if not _is_num(net_debt_value):
        result["reason"] = "Datoria netă lipsește (datorie totală sau numerar)."
        return result
    if not all(_is_num(x) for x in (growth, discount_rate, terminal_growth)):
        result["reason"] = "Ipoteze incomplete (creștere, rată de scont sau g terminal)."
        return result
    if terminal_growth > MAX_TERMINAL_GROWTH + 1e-12:
        result["reason"] = "Creșterea terminală depășește 3%: nicio companie nu crește la nesfârșit peste economie."
        return result
    if discount_rate <= terminal_growth:
        result["reason"] = "Rata de scont trebuie să fie mai mare decât creșterea terminală."
        return result

    path = growth_path(growth, terminal_growth, years)
    flows, pv_sum, fcf = [], 0.0, fcf0
    for t, g in enumerate(path, start=1):
        fcf = fcf * (1 + g)
        pv = fcf / (1 + discount_rate) ** t
        flows.append({"year": t, "growth": g, "fcf": fcf, "pv": pv})
        pv_sum += pv
    terminal_value = fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** years
    enterprise = pv_sum + pv_terminal
    equity = enterprise - net_debt_value

    result.update(enterprise_value=enterprise, equity_value=equity, pv_fcf=pv_sum,
                  pv_terminal=pv_terminal, terminal_share=pv_terminal / enterprise,
                  flows=flows, growth_path=path)
    if discount_rate - terminal_growth < MIN_SPREAD_STABLE - 1e-12:
        result["warnings"].append("Diferența dintre rata de scont și g terminal e sub 2 pp: rezultatul e instabil.")
    if result["terminal_share"] > 0.75:
        result["warnings"].append(
            f"Valoarea terminală reprezintă {result['terminal_share'] * 100:.0f}% din valoarea întreprinderii: "
            "rezultatul depinde aproape în întregime de ipotezele pe termen lung.")
    if equity <= 0:
        result["reason"] = "Datoria netă depășește valoarea întreprinderii: capital propriu negativ în model."
        return result
    result["per_share"] = equity / shares
    return result


def dcf_sensitivity(fcf0, growth, net_debt_value, shares, discount_rates, terminal_growths, years=5):
    """Tabel de sensibilitate: valoare pe acțiune pentru fiecare pereche (rată de scont, g terminal).

    Rânduri = rate de scont, coloane = g terminal (fracții). Combinațiile fără sens
    (r ≤ g, capital negativ) rămân NaN.
    """
    table = pd.DataFrame(index=list(discount_rates), columns=list(terminal_growths), dtype=float)
    for r in discount_rates:
        for g in terminal_growths:
            value = dcf_fcf(fcf0, growth, r, g, net_debt_value, shares, years)["per_share"]
            table.loc[r, g] = float("nan") if value is None else value
    table.index.name = "Rată de scont"
    table.columns.name = "g terminal"
    return table


def dcf_inputs(annual_income, annual_balance, annual_cashflow, quarterly_cashflow=None,
               quarterly_balance=None):
    """Strânge din situațiile financiare tot ce intră în DCF, cu sursa fiecărei valori.

    FCF: TTM din trimestriale când există 4 trimestre consecutive; altfel ultimul an fiscal.
    Datoria netă: din cel mai recent bilanț (trimestrial dacă are și datorie, și numerar;
    altfel anual). Întoarce un dict; valorile care nu pot fi citite sunt None.
    """
    balance = quarterly_balance if net_debt(quarterly_balance) is not None else annual_balance
    fcf, fcf_basis = fcf_ttm(quarterly_cashflow), "TTM (ultimele 4 trimestre)"
    cfo = ttm_sum(quarterly_cashflow, CFO_ROWS)
    capex = ttm_sum(quarterly_cashflow, CAPEX_ROWS)
    if fcf is None:
        cfo = stmt_value(annual_cashflow, CFO_ROWS)
        capex = stmt_value(annual_cashflow, CAPEX_ROWS)
        fcf = free_cash_flow(cfo, capex)
        date = stmt_date(annual_cashflow)
        fcf_basis = f"ultimul an fiscal ({date:%Y-%m-%d})" if (fcf is not None and date is not None) else None
    history = fcf_history(annual_cashflow)
    shares, is_diluted = diluted_shares(annual_income, annual_balance)
    return {
        "fcf": fcf, "fcf_basis": fcf_basis, "cfo": cfo, "capex": capex,
        "fcf_history": history, "fcf_cagr": fcf_cagr(history),
        "fcf_average": fcf_average(history), "fcf_average_years": min(len(history), 3),
        "total_debt": total_debt(balance),
        "cash": stmt_value(balance, CASH_ROWS),
        "net_debt": net_debt(balance),
        "balance_date": stmt_date(balance),
        "shares": shares, "shares_diluted": is_diluted,
        "interest_expense": stmt_value(annual_income, INTEREST_ROWS),
        "tax_rate": effective_tax_rate(annual_income),
    }


# --- Indicatori de bază din situațiile financiare ----------------------------
# Rezervă pentru când Yahoo nu trimite rezumatul companiei (`Ticker.info`). Cheile
# poartă aceleași nume și aceleași unități ca în `info`, ca restul aplicației să le
# citească la fel: rentabilitățile și marjele sunt fracții, `debtToEquity` e în procente.

NET_INCOME_ROWS = ("Net Income Common Stockholders", "Net Income")
REVENUE_ROWS = ("Total Revenue", "Operating Revenue")
OPERATING_INCOME_ROWS = ("Operating Income", "EBIT")
EQUITY_ROWS = ("Stockholders Equity", "Common Stock Equity")
TOTAL_ASSETS_ROWS = ("Total Assets",)
CURRENT_ASSETS_ROWS = ("Current Assets",)
CURRENT_LIABILITIES_ROWS = ("Current Liabilities",)
INVENTORY_ROWS = ("Inventory",)


def _ratio(numerator, denominator, positive_denominator=True):
    if not _is_num(numerator) or not _is_num(denominator) or denominator == 0:
        return None
    if positive_denominator and denominator < 0:
        return None
    return numerator / denominator


def _flow(quarterly, annual, names):
    """Flux pe ultimele 4 trimestre; dacă nu există 4 trimestre consecutive, ultimul an fiscal."""
    value = ttm_sum(quarterly, names)
    return value if value is not None else stmt_value(annual, names)


def ratios_from_statements(annual_income, annual_balance, annual_cashflow=None, quarterly_income=None,
                           quarterly_balance=None, quarterly_cashflow=None, price=None, per_share_ok=True):
    """Indicatorii fundamentali de bază, calculați din situațiile financiare.

    Fluxurile (venituri, profit, CFO) sunt TTM când există 4 trimestre, altfel ultimul an
    fiscal; soldurile vin din cel mai recent bilanț. Rentabilitățile folosesc soldul de la
    sfârșitul perioadei, deci pot diferi ușor de cele din Yahoo (care mediază perioada).

    `per_share_ok=False` când nu se știe dacă situațiile sunt în moneda de tranzacționare
    (ADR-uri): atunci indicatorii care compară prețul cu valori contabile (P/E, P/BV) și
    cei pe acțiune rămân None. Orice intrare lipsă dă None pentru indicatorul respectiv.
    """
    balance = quarterly_balance if stmt_value(quarterly_balance, TOTAL_ASSETS_ROWS) is not None else annual_balance
    net_income = _flow(quarterly_income, annual_income, NET_INCOME_ROWS)
    revenue = _flow(quarterly_income, annual_income, REVENUE_ROWS)
    operating_income = _flow(quarterly_income, annual_income, OPERATING_INCOME_ROWS)
    equity = stmt_value(balance, EQUITY_ROWS)
    assets = stmt_value(balance, TOTAL_ASSETS_ROWS)
    current_assets = stmt_value(balance, CURRENT_ASSETS_ROWS)
    current_liabilities = stmt_value(balance, CURRENT_LIABILITIES_ROWS)
    inventory = stmt_value(balance, INVENTORY_ROWS)
    debt = total_debt(balance)

    out = {
        "totalRevenue": revenue,
        "netIncomeToCommon": net_income,
        "operatingCashflow": _flow(quarterly_cashflow, annual_cashflow, CFO_ROWS),
        "totalDebt": debt,
        "totalCash": stmt_value(balance, CASH_ROWS),
        "returnOnEquity": _ratio(net_income, equity),          # capital propriu negativ -> N/A
        "returnOnAssets": _ratio(net_income, assets),
        "profitMargins": _ratio(net_income, revenue),
        "operatingMargins": _ratio(operating_income, revenue),
        "currentRatio": _ratio(current_assets, current_liabilities),
        "quickRatio": (_ratio(current_assets - inventory, current_liabilities)
                       if _is_num(current_assets) and _is_num(inventory) else None),
        "trailingEps": None, "bookValue": None, "trailingPE": None, "priceToBook": None,
    }
    leverage = _ratio(debt, equity)
    out["debtToEquity"] = None if leverage is None else leverage * 100

    if per_share_ok:
        diluted, _ = diluted_shares(annual_income, balance)
        period_end_shares = stmt_value(balance, BASIC_SHARES_ROWS) or diluted
        eps = _ratio(net_income, diluted)
        book = _ratio(equity, period_end_shares)
        out["trailingEps"], out["bookValue"] = eps, book
        if _is_num(price) and price > 0:
            out["trailingPE"] = price / eps if (eps is not None and eps > 0) else None      # pierdere -> N/A
            out["priceToBook"] = price / book if (book is not None and book > 0) else None
    return out


# --- Graham ------------------------------------------------------------------

GRAHAM_MAX_GROWTH = 15.0        # creșterea din formula revizuită se plafonează (procente)


def graham_number(eps, book_value_per_share):
    """Numărul Graham: √(22,5 × EPS × valoare contabilă pe acțiune).

    Prețul maxim pe care Graham îl accepta pentru o acțiune defensivă (P/E ≤ 15 și
    P/BV ≤ 1,5). Doar pentru EPS și valoare contabilă pozitive; altfel None.
    """
    if not _is_num(eps) or not _is_num(book_value_per_share) or eps <= 0 or book_value_per_share <= 0:
        return None
    return math.sqrt(22.5 * eps * book_value_per_share)


def graham_revised(eps, growth_pct, aaa_yield_pct):
    """Formula revizuită a lui Graham: V = EPS × (8,5 + 2g) × 4,4 / Y.

    g = creșterea anuală așteptată a profitului, în procente, plafonată la 15;
    Y = randamentul curent al obligațiunilor corporative AAA, în procente. Factorul
    4,4 / Y scade valoarea când dobânzile sunt peste nivelul din 1962 (4,4%); fără el
    formula supraevaluează sistematic. None pentru EPS ≤ 0, Y lipsă sau multiplu ≤ 0.
    """
    if not _is_num(eps) or eps <= 0 or not _is_num(growth_pct) or not _is_num(aaa_yield_pct) or aaa_yield_pct <= 0:
        return None
    multiple = 8.5 + 2 * min(growth_pct, GRAHAM_MAX_GROWTH)
    if multiple <= 0:
        return None
    return eps * multiple * 4.4 / aaa_yield_pct


# --- Altman Z ----------------------------------------------------------------

RETAINED_EARNINGS_ROWS = ("Retained Earnings",)
TOTAL_LIABILITIES_ROWS = ("Total Liabilities Net Minority Interest", "Total Liabilities")
EBIT_ROWS = ("EBIT", "Operating Income")
# Sectoare (denumirile Yahoo) în care domină producția: se aplică Z-ul original.
MANUFACTURING_SECTORS = frozenset({"industrials", "basic materials", "energy", "consumer defensive", "consumer cyclical"})
ALTMAN_ZONES = {"Z": (1.81, 2.99), "Z''": (1.1, 2.6)}       # (sub = dificultate, peste = sigur)


def is_financial_issuer(sector, symbol=None):
    """Bancă, asigurător, fond sau alt emitent financiar (inclusiv lista BVB)."""
    if (sector or "").strip().lower() in ("financial services", "financial", "financials"):
        return True
    sym = str(symbol or "").upper()
    return sym.endswith(".RO") and sym[:-3] in BVB_FINANCIALS


def altman_variant(sector, symbol=None):
    """Varianta de scor care se aplică: "Z", "Z''" sau None (sector financiar).

    Z (1968) e calibrat pe companii de producție listate în SUA. Z'' (fără rotația
    activelor, cu capital propriu contabil) e varianta pentru servicii și piețe emergente:
    se folosește pentru celelalte sectoare, pentru BVB și când sectorul nu e cunoscut.
    """
    if is_financial_issuer(sector, symbol):
        return None
    if str(symbol or "").upper().endswith(".RO"):
        return "Z''"
    return "Z" if (sector or "").strip().lower() in MANUFACTURING_SECTORS else "Z''"


def altman_zone(value, variant):
    """"safe", "grey" sau "distress" după pragurile variantei; None dacă lipsește scorul."""
    if not _is_num(value) or variant not in ALTMAN_ZONES:
        return None
    low, high = ALTMAN_ZONES[variant]
    if value < low:
        return "distress"
    return "safe" if value > high else "grey"


def altman_z(annual_income, annual_balance, quarterly_income=None, quarterly_balance=None, market_cap=None):
    """Altman Z și Z'' din situațiile financiare. Scorurile nu sunt plafonate sau „corectate".

    Z   = 1,2·X1 + 1,4·X2 + 3,3·X3 + 0,6·X4 + 1,0·X5
    Z'' = 6,56·X1 + 3,26·X2 + 6,72·X3 + 1,05·X4'
      X1 = capital de lucru / active      X2 = rezultat reportat / active
      X3 = EBIT / active                  X5 = venituri / active
      X4 = capitalizare bursieră / datorii totale;  X4' = capital propriu contabil / datorii totale
    EBIT și veniturile sunt TTM când există 4 trimestre; soldurile, din cel mai recent bilanț.
    Un scor e None dacă îi lipsește orice componentă (`missing` spune care). `market_cap`
    trebuie să fie în moneda situațiilor financiare; dacă nu e sigur, se transmite None.
    """
    balance = quarterly_balance if stmt_value(quarterly_balance, TOTAL_ASSETS_ROWS) is not None else annual_balance
    assets = stmt_value(balance, TOTAL_ASSETS_ROWS)
    liabilities = stmt_value(balance, TOTAL_LIABILITIES_ROWS)
    current_assets = stmt_value(balance, CURRENT_ASSETS_ROWS)
    current_liabilities = stmt_value(balance, CURRENT_LIABILITIES_ROWS)
    working_capital = (current_assets - current_liabilities
                       if _is_num(current_assets) and _is_num(current_liabilities) else None)
    x = {
        "X1": _ratio(working_capital, assets),
        "X2": _ratio(stmt_value(balance, RETAINED_EARNINGS_ROWS), assets),
        "X3": _ratio(_flow(quarterly_income, annual_income, EBIT_ROWS), assets),
        "X4": _ratio(market_cap, liabilities),
        "X4_book": _ratio(stmt_value(balance, EQUITY_ROWS), liabilities),
        "X5": _ratio(_flow(quarterly_income, annual_income, REVENUE_ROWS), assets),
    }
    out = {"z": None, "z2": None, "components": x, "balance_date": stmt_date(balance),
           "missing": [k for k, v in x.items() if v is None]}
    if all(x[k] is not None for k in ("X1", "X2", "X3", "X4", "X5")):
        out["z"] = 1.2 * x["X1"] + 1.4 * x["X2"] + 3.3 * x["X3"] + 0.6 * x["X4"] + 1.0 * x["X5"]
    if all(x[k] is not None for k in ("X1", "X2", "X3", "X4_book")):
        out["z2"] = 6.56 * x["X1"] + 3.26 * x["X2"] + 6.72 * x["X3"] + 1.05 * x["X4_book"]
    return out


# --- Piotroski F-Score -------------------------------------------------------

def _short(value):
    """Sumă scurtă pentru tabele: 111,48 mld, 626,6 mil. Destule zecimale ca două sume apropiate să se distingă."""
    a = abs(value)
    for limit, unit in ((1e9, "mld"), (1e6, "mil")):
        if a >= limit:
            return f"{value / limit:,.2f} {unit}"
    return f"{value:,.2f}"


GROSS_PROFIT_ROWS = ("Gross Profit",)
EBITDA_ROWS = ("EBITDA", "Normalized EBITDA")
SHARES_COUNT_ROWS = ("Ordinary Shares Number", "Share Issued")


def piotroski(annual_income, annual_balance, annual_cashflow):
    """Piotroski F-Score: 9 criterii, ultimul an fiscal față de cel precedent.

    Întoarce {"passed", "evaluable", "criteria", "year", "prior_year"}; `criteria` e o listă de
    dict-uri {"name", "passed" (True / False / None), "detail"}. Un criteriu fără date e None,
    nu picat: scorul se citește „passed din evaluable". Fără doi ani de situații, toate sunt None.
    Rentabilitatea și rotația folosesc activele de la sfârșitul anului.
    """
    def val(df, rows, col):
        return stmt_value(df, rows, col)

    def pct(v):
        return "N/A" if v is None else f"{v * 100:.1f}%"

    def num2(v):
        return "N/A" if v is None else f"{v:.2f}"

    def compare(now, before, better, fmt):
        if now is None or before is None:
            return None, f"{fmt(now)} față de {fmt(before)}"
        return better(now, before), f"{fmt(now)} față de {fmt(before)}"

    ni = [val(annual_income, NET_INCOME_ROWS, c) for c in (0, 1)]
    rev = [val(annual_income, REVENUE_ROWS, c) for c in (0, 1)]
    gp = [val(annual_income, GROSS_PROFIT_ROWS, c) for c in (0, 1)]
    ta = [val(annual_balance, TOTAL_ASSETS_ROWS, c) for c in (0, 1)]
    ltd = [val(annual_balance, LONG_DEBT_ROWS, c) for c in (0, 1)]
    ca = [val(annual_balance, CURRENT_ASSETS_ROWS, c) for c in (0, 1)]
    cl = [val(annual_balance, CURRENT_LIABILITIES_ROWS, c) for c in (0, 1)]
    shares = [val(annual_balance, SHARES_COUNT_ROWS, c) or val(annual_income, DILUTED_SHARES_ROWS, c) for c in (0, 1)]
    cfo = val(annual_cashflow, CFO_ROWS, 0)

    roa = [_ratio(ni[c], ta[c]) for c in (0, 1)]
    leverage = [_ratio(ltd[c], ta[c]) for c in (0, 1)]
    liquidity = [_ratio(ca[c], cl[c]) for c in (0, 1)]
    margin = [_ratio(gp[c], rev[c]) for c in (0, 1)]
    turnover = [_ratio(rev[c], ta[c]) for c in (0, 1)]

    criteria = []

    def add(name, result):
        criteria.append({"name": name, "passed": result[0], "detail": result[1]})

    add("1. Rentabilitatea activelor (ROA) pozitivă", (None if roa[0] is None else roa[0] > 0, pct(roa[0])))
    add("2. Flux de numerar din exploatare pozitiv", (None if cfo is None else cfo > 0, "pozitiv" if (cfo or 0) > 0 else ("N/A" if cfo is None else "negativ")))
    add("3. ROA în creștere", compare(roa[0], roa[1], lambda a, b: a > b, pct))
    add("4. Flux din exploatare peste profitul net",
        (None if cfo is None or ni[0] is None else cfo > ni[0],
         "N/A" if cfo is None or ni[0] is None else f"CFO {_short(cfo)} față de profit net {_short(ni[0])}"))
    add("5. Datorie pe termen lung / active în scădere",
        compare(leverage[0], leverage[1], lambda a, b: a < b or (a == 0 and b == 0), pct))
    add("6. Lichiditate curentă în creștere", compare(liquidity[0], liquidity[1], lambda a, b: a > b, num2))
    # Toleranță de 0,1% pentru rotunjiri; orice emisiune reală de acțiuni pică criteriul.
    add("7. Fără emisiune de acțiuni noi",
        (None if shares[0] is None or shares[1] is None else shares[0] <= shares[1] * 1.001,
         "N/A" if shares[0] is None or shares[1] is None else f"{(shares[0] / shares[1] - 1) * 100:+.1f}% acțiuni"))
    add("8. Marjă brută în creștere", compare(margin[0], margin[1], lambda a, b: a > b, pct))
    add("9. Rotația activelor în creștere", compare(turnover[0], turnover[1], lambda a, b: a > b, num2))

    return {"passed": sum(1 for c in criteria if c["passed"] is True),
            "evaluable": sum(1 for c in criteria if c["passed"] is not None),
            "criteria": criteria,
            "year": stmt_date(annual_balance, 0), "prior_year": stmt_date(annual_balance, 1)}


# --- Îndatorare --------------------------------------------------------------

def leverage_ratios(annual_income, annual_balance, quarterly_income=None, quarterly_balance=None):
    """Datorie netă / EBITDA și acoperirea dobânzii (EBIT / cheltuieli cu dobânzile).

    Fluxurile sunt TTM când există 4 trimestre, altfel ultimul an fiscal; datoria netă, din
    cel mai recent bilanț. Datorie netă / EBITDA e None când EBITDA ≤ 0 (raportul nu are sens)
    și poate fi negativ (numerar net). Acoperirea e None când dobânda lipsește sau e zero.
    """
    balance = quarterly_balance if net_debt(quarterly_balance) is not None else annual_balance
    net = net_debt(balance)
    ebitda = _flow(quarterly_income, annual_income, EBITDA_ROWS)
    ebit = _flow(quarterly_income, annual_income, EBIT_ROWS)
    interest = _flow(quarterly_income, annual_income, INTEREST_ROWS)
    coverage = None
    if _is_num(ebit) and _is_num(interest) and interest != 0:
        coverage = ebit / abs(interest)
    return {"net_debt": net, "ebitda": ebitda, "ebit": ebit, "interest_expense": interest,
            "net_debt_to_ebitda": _ratio(net, ebitda), "interest_coverage": coverage}
