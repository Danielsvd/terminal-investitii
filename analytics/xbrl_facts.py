"""Situațiile financiare XBRL de la SEC (`companyfacts`) → serii curate: anual, trimestrial, TTM, point-in-time.

Faza 3 SEC (propunerea `claude/propunere_faza3_xbrl_2026-11-27.md`, deciziile D1–D10 acceptate de Daniel pe 27.11.2026).
Funcții pure: fără rețea, fără Streamlit. Aceeași bibliotecă servește aplicația și testul istoric.

Reguli (D2–D5), corectate după proba din 27.11.2026 (52 de simboluri):
  * Fiecare indicator are o listă de etichete US-GAAP care înseamnă ACELAȘI lucru, în ordinea priorității. Seria se
    ancorează pe eticheta celei mai recente perioade și coboară în timp cu ea; trece la altă etichetă doar unde cea
    curentă lipsește (ex. `SalesRevenueNet` → `RevenueFromContractWithCustomer…` în 2018). La o astfel de „cusătură”,
    dacă ambele etichete au valoare într-o perioadă mai nouă de aceeași lungime, diferența > `SEAM_TOL` (1%) marchează
    perioada „conflict” (exclusă din CAGR). Mărimi diferite stau în indicatori diferiți (profit total vs. profit pentru
    acționarii comuni; numerar cu / fără numerar restricționat).
  * Doar unitățile `USD`, `USD/shares`, `shares`. Anii fiscali vin DOAR din 10-K (unele companii pun în 10-Q și cifre
    pe ultimele 12 luni, ex. Amazon); trimestrele din 10-Q și 10-K. 20-F → nimic (emitenții străini rămân pe Yahoo, D7).
  * Perioada se deduce din `start`–`end`, NU din `fy`/`fp`. Trimestrele fluxului de numerar (raportat doar cumulat în
    10-Q) și T4 se obțin din diferențe de valori cumulate cu același început.
  * Point-in-time (D4): o valoare e cunoscută din ziua lucrătoare de după `filed`. `asof=None` = ultima versiune.
  * Split-uri (D5): o valoare din raportul depus la data F e deja ajustată pentru split-urile de până la F; se mai
    înmulțește doar cu split-urile de DUPĂ F. Acțiunile raportate greșit în mii / milioane (rapoarte XBRL vechi) se
    corectează cu un multiplu de 1.000 când profit / EPS raportat confirmă scara (`scale_fixed`).
  * Lipsă = None / NaN, niciodată 0.
"""
import math
from datetime import date, timedelta

import pandas as pd

ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT", "10-KT/A")
FORMS = ANNUAL_FORMS + ("10-Q", "10-Q/A")
FOREIGN_FORMS = ("20-F", "20-F/A", "40-F", "40-F/A")
SEAM_TOL = 0.01                 # la schimbarea etichetei: diferență > 1% între etichete → conflict
TTM_CHECK_TOL = 0.005           # TTM la încheierea anului vs. valoarea anuală
EPS_TOL = 0.02                  # EPS calculat vs. raportat
SCALE_TOL = 0.05                # scara acțiunilor: raportul trebuie să fie 1.000^k cu cel mult 5% abatere
Q_DAYS = (80, 100)
H_DAYS = (170, 190)
M9_DAYS = (260, 285)
FY_DAYS = (350, 380)
FY_END_TOL_DAYS = 7             # bilanțul „de la încheierea anului”: cel mult o săptămână distanță
SAME_LENGTH_DAYS = 10           # perioade „de aceeași lungime” la controlul cusăturii

# indicator → (unitate, tip, [(taxonomie, etichetă), ...]) — etichete cu ACELAȘI înțeles, în ordinea priorității
CONCEPTS = {
    "revenue": ("USD", "duration", [
        ("us-gaap", "Revenues"),                                            # total (inclusiv alte venituri din exploatare)
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
        ("us-gaap", "RevenuesNetOfInterestExpense"),                        # bănci: venituri nete
        ("us-gaap", "SalesRevenueGoodsNet"),
    ]),
    "net_income": ("USD", "duration", [                                     # profitul atribuibil acționarilor companiei
        ("us-gaap", "NetIncomeLoss"),
    ]),
    "net_income_common": ("USD", "duration", [                              # după dividendele preferențiale (numărătorul EPS)
        ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersDiluted"),
        ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic"),
    ]),
    "eps_diluted": ("USD/shares", "duration", [
        ("us-gaap", "EarningsPerShareDiluted"),
    ]),
    "shares_diluted": ("shares", "duration", [
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"),   # ex. XOM: o singură cifră
    ]),
    # rezerva profitului atribuibil (unele companii nu etichetează NetIncomeLoss: AVGO, OXY, VG):
    # profitul total minus partea minoritarilor
    "profit_loss": ("USD", "duration", [
        ("us-gaap", "ProfitLoss"),
    ]),
    "nci": ("USD", "duration", [
        ("us-gaap", "NetIncomeLossAttributableToNoncontrollingInterest"),
    ]),
    "nci_redeemable": ("USD", "duration", [
        ("us-gaap", "NetIncomeLossAttributableToRedeemableNoncontrollingInterest"),
    ]),
    "nci_nonredeemable": ("USD", "duration", [
        ("us-gaap", "NetIncomeLossAttributableToNonredeemableNoncontrollingInterest"),
    ]),
    "cfo": ("USD", "duration", [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    ]),
    "capex": ("USD", "duration", [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "PaymentsToAcquireProductiveAssets"),
        ("us-gaap", "PaymentsToAcquireOilAndGasPropertyAndEquipment"),
        ("us-gaap", "PaymentsToAcquireOilAndGasProperty"),
        ("us-gaap", "PaymentsToExploreAndDevelopOilAndGasProperties"),
        ("us-gaap", "PaymentsForProceedsFromProductiveAssets"),
    ]),
    "capex_software": ("USD", "duration", [                                 # software capitalizat (adăugat la capex PP&E)
        ("us-gaap", "PaymentsToDevelopSoftware"),
        ("us-gaap", "PaymentsForSoftware"),
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
    ]),
    "cash": ("USD", "instant", [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
    ]),
    "debt_lt": ("USD", "instant", [                                         # datoria pe termen lung totală (incl. partea curentă)
        ("us-gaap", "LongTermDebt"),
    ]),
    "shares_out": ("shares", "instant", [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ]),
}
PPE_CAPEX = "PaymentsToAcquirePropertyPlantAndEquipment"
# indicatori la care ancora e eticheta cu valoarea (absolută) cea mai mare în perioada recentă, nu prima din listă:
# la petrol și gaze, capex-ul pe proprietăți petroliere e mai mare decât cel pe PP&E (DVN: 7,45 vs. 3,65 mld.)
ANCHOR_MAX = ("capex",)
NEIGHBOR_SCALE_RANGE = (0.5, 2.0)   # scara acțiunilor confirmată de anii vecini: raport la mediană / 1.000^k
DURATION_INDICATORS = tuple(k for k, v in CONCEPTS.items() if v[1] == "duration")
INSTANT_INDICATORS = tuple(k for k, v in CONCEPTS.items() if v[1] == "instant")
# nu se adună pe trimestre: EPS-ul anual ≠ suma EPS-urilor trimestriale; acțiunile sunt medii.
# Dividendul pe acțiune SE adună (plățile din an).
NON_ADDITIVE = ("eps_diluted", "shares_diluted")

_COLS = ["prio", "concept", "start", "end", "days", "val", "form", "filed", "accn"]
_OUT = ["val", "start", "filed", "accn", "concept", "conflict"]


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
    us = facts.get("us-gaap") or {}
    if "ifrs-full" in facts and not any(us.values()):
        return "20-F"
    forms = set()
    for concepts in us.values():
        for rows in (concepts.get("units") or {}).values():
            forms.update(r.get("form") for r in rows[:50])
        if len(forms) > 3:
            break
    if forms & set(FORMS):
        return "10-K"
    if forms & set(FOREIGN_FORMS) or "ifrs-full" in facts:
        return "20-F"
    return None


def raw_facts(cf, indicator, asof=None, forms=FORMS):
    """Toate valorile brute ale indicatorului (toate etichetele lui), ca DataFrame cu coloanele
    `prio, concept, start, end, days, val, form, filed, accn`. Cu `asof`: doar valorile cunoscute la acea dată."""
    unit, kind, concepts = CONCEPTS[indicator]
    facts = (cf or {}).get("facts") or {}
    asof_d = _day(asof) if asof is not None else None
    rows = []
    for prio, (tax, name) in enumerate(concepts):
        node = (facts.get(tax) or {}).get(name) or {}
        for r in (node.get("units") or {}).get(unit) or []:
            if r.get("form") not in forms:
                continue
            val, end, filed = _num(r.get("val")), _day(r.get("end")), _day(r.get("filed"))
            if val is None or end is None or filed is None:
                continue
            if asof_d is not None and available_on(filed) > asof_d:
                continue
            start = _day(r.get("start"))
            if kind == "duration" and start is None:
                continue
            days = (end - start).days + 1 if start else 0
            rows.append((prio, name, start, end, days, val, r.get("form"), filed, r.get("accn")))
    return pd.DataFrame(rows, columns=_COLS)


def _latest(df):
    """Pentru fiecare (etichetă, perioadă): rândul din raportul cel mai recent (ultima versiune, inclusiv retratări)."""
    key = df["start"].map(str) + "|" + df["end"].map(str)        # map(str): None → "None" și în pandas 3
    return (df.assign(_k=key).sort_values(["filed", "accn"]).drop_duplicates(["prio", "_k"], keep="last")
            .drop(columns="_k"))


def _plen(p):
    return (p[1] - p[0]).days if p[0] else 0


def _seam_conflict(by_period, periods, at, old, new):
    """La trecerea de la eticheta `old` (perioadele mai noi) la `new` (perioada `at`): cele două etichete se compară
    în cea mai apropiată perioadă mai nouă, de aceeași lungime, unde ambele au valoare. Fără suprapunere → False."""
    length = _plen(at)
    for p in reversed(periods):                          # periods e descrescător: reversed = de la cea mai veche
        if p[1] <= at[1] or abs(_plen(p) - length) > SAME_LENGTH_DAYS:
            continue
        avail = by_period[p]
        if old in avail and new in avail:
            a, b = avail[old]["val"], avail[new]["val"]
            den = max(abs(a), abs(b))
            return bool(den > 0 and abs(a - b) / den > SEAM_TOL)
    return False


def _resolve(df, anchor_max=False):
    """O valoare pe perioadă, ancorată pe eticheta celei mai recente perioade (vezi docstring-ul modulului).
    `anchor_max`: ancora inițială e eticheta cu valoarea absolută cea mai mare (nu prima din listă).
    Întoarce rândurile alese, cu coloana `conflict` (cusătură cu diferență > SEAM_TOL)."""
    if df.empty:
        return df.assign(conflict=pd.Series(dtype=bool))
    latest = _latest(df)
    by_period = {}
    for _, r in latest.iterrows():
        by_period.setdefault((r["start"], r["end"]), {})[r["prio"]] = r
    periods = sorted(by_period, key=lambda p: (p[1], p[0] or date.min), reverse=True)
    chosen, current = [], None
    for p in periods:
        avail = by_period[p]
        conflict = False
        if current is None or current not in avail:
            new = (max(avail, key=lambda k: abs(avail[k]["val"])) if anchor_max and current is None
                   else min(avail))
            if current is not None:
                conflict = _seam_conflict(by_period, periods, p, current, new)
            current = new
        row = avail[current].copy()
        row["conflict"] = conflict
        chosen.append(row)
    return pd.DataFrame(chosen).reset_index(drop=True)


def _in(days, rng):
    return days is not None and rng[0] <= days <= rng[1]


def _empty():
    return pd.DataFrame(columns=_OUT).rename_axis("end")


def _frame(df):
    if df.empty:
        return _empty()
    out = df.set_index("end")[_OUT].sort_index()
    return out[~out.index.duplicated(keep="last")]


def annual(cf, indicator, asof=None):
    """Valorile anuale (exercițiu de 350–380 de zile, doar din 10-K) ale unui indicator de flux, pe data încheierii.
    Coloane: `val, start, filed, accn, concept, conflict`."""
    if CONCEPTS[indicator][1] != "duration":
        raise ValueError(f"{indicator}: indicator de bilanț; folosește balance_at_year_end()")
    raw = raw_facts(cf, indicator, asof, forms=ANNUAL_FORMS)
    raw = raw[raw["days"].apply(lambda d: _in(d, FY_DAYS))]
    return _frame(_resolve(raw, anchor_max=indicator in ANCHOR_MAX))


def fiscal_year_ends(cf, asof=None):
    """Datele de încheiere a exercițiilor financiare (din profitul net sau cifra de afaceri anuală, din 10-K)."""
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
    return _frame(_resolve(df.assign(start=None)))


def balance_at_year_end(cf, indicator, asof=None, ends=None):
    """Valoarea de bilanț la fiecare încheiere de exercițiu (cel mult `FY_END_TOL_DAYS` distanță)."""
    inst = instants(cf, indicator, asof)
    ends = fiscal_year_ends(cf, asof) if ends is None else ends
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
    cumulate cu același început (T2 = 6 luni − 3 luni, T3 = 9 luni − 6 luni, T4 = an − 9 luni). Valorile pe 12 luni
    din 10-Q (TTM raportat de unele companii) nu intră. Indicatorii pe acțiune nu se derivează."""
    if CONCEPTS[indicator][1] != "duration":
        raise ValueError(f"{indicator}: indicator de bilanț")
    raw = raw_facts(cf, indicator, asof)
    if not raw.empty:
        raw = raw[~(raw["days"].apply(lambda d: _in(d, FY_DAYS)) & ~raw["form"].isin(ANNUAL_FORMS))]
    res = _resolve(raw, anchor_max=indicator in ANCHOR_MAX)
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
    if splits is None or len(splits) == 0 or when is None:
        return 1.0
    s = pd.Series(splits).dropna()
    idx = pd.DatetimeIndex(s.index)
    s.index = idx.tz_localize(None) if idx.tz is not None else idx
    after = s[(s.index > pd.Timestamp(when)) & (s > 0)]
    return float(after.prod()) if len(after) else 1.0


def scale_correction(shares, implied):
    """Multiplul de 1.000 (1.000^k, k ≠ 0) cu care trebuie înmulțite acțiunile raportate ca să se potrivească cu
    acțiunile implicite (profit / EPS raportat), sau 1.0. Rapoartele XBRL vechi au uneori acțiunile în mii/milioane."""
    shares, implied = _num(shares), _num(implied)
    if not shares or not implied or shares <= 0 or implied <= 0:
        return 1.0
    r = implied / shares
    k = round(math.log10(r) / 3)
    if k == 0:
        return 1.0
    return float(1000.0 ** k) if abs(r / 1000.0 ** k - 1) <= SCALE_TOL else 1.0


def _col(df, ends):
    return [df.loc[e, "val"] if e in df.index else float("nan") for e in ends]


def annual_table(cf, asof=None, splits=None):
    """Tabelul anual (un rând pe exercițiu, indexat pe data încheierii), în USD și pe acțiune.

    Coloane: revenue, net_income, net_income_common, cfo, capex (pozitiv; + software capitalizat lângă eticheta PP&E),
    fcf, sbc, equity, cash, debt_lt, shares_diluted (scară corectată, ajustate cu split-urile de după depunere),
    eps (= profit pentru acționarii comuni / acțiuni), eps_reported (ajustat), eps_check, revenue_ps, fcf_ps, bvps,
    dps (ajustat; din trimestre când lipsește anualul), filed (cel mai recent raport folosit), conflicts (indicatorii
    cu cusătură în conflict), scale_fixed (scara acțiunilor corectată), ni_proxy (profitul = profit total − partea
    minoritarilor, fiindcă NetIncomeLoss lipsește). Lipsă = NaN."""
    ends = fiscal_year_ends(cf, asof)
    if not ends:
        return pd.DataFrame()
    t = pd.DataFrame(index=pd.Index(ends, name="end"))
    filed = {e: None for e in ends}
    conflicts = {e: [] for e in ends}
    frames = {}

    def put(name, df):
        frames[name] = df
        t[name] = _col(df, ends)
        for e in ends:
            if e in df.index:
                f = df.loc[e, "filed"]
                filed[e] = f if filed[e] is None else max(filed[e], f)
                if bool(df.loc[e, "conflict"]):
                    conflicts[e].append(name)

    for ind in ("revenue", "net_income", "net_income_common", "cfo", "capex", "capex_software", "sbc",
                "eps_diluted", "shares_diluted", "dps"):
        put(ind, annual(cf, ind, asof))

    # profitul atribuibil lipsă: profitul total − partea minoritarilor (o singură etichetă sau suma celor două)
    pl = annual(cf, "profit_loss", asof)
    nci = annual(cf, "nci", asof)
    nci_r, nci_n = annual(cf, "nci_redeemable", asof), annual(cf, "nci_nonredeemable", asof)
    t["ni_proxy"] = False
    for e in ends:
        if not pd.isna(t.loc[e, "net_income"]) or e not in pl.index:
            continue
        if e in nci.index:
            minority = nci.loc[e, "val"]
        elif e in nci_r.index and e in nci_n.index:
            minority = nci_r.loc[e, "val"] + nci_n.loc[e, "val"]
        else:
            continue                                     # partea minoritarilor necunoscută: profitul rămâne lipsă
        t.loc[e, "net_income"] = pl.loc[e, "val"] - minority
        t.loc[e, "ni_proxy"] = True
        filed[e] = max(filed[e] or pl.loc[e, "filed"], pl.loc[e, "filed"])
    for ind in ("equity", "cash", "debt_lt"):
        put(ind, balance_at_year_end(cf, ind, asof, ends))

    # dividendul anual lipsă (unele companii îl raportează doar pe trimestru): suma celor 4 trimestre din exercițiu
    dps_filed = {e: (frames["dps"].loc[e, "filed"] if e in frames["dps"].index else None) for e in ends}
    dps_q = None
    for e in ends:
        if not pd.isna(t.loc[e, "dps"]):
            continue
        dps_q = quarterly(cf, "dps", asof) if dps_q is None else dps_q
        inside = dps_q[(dps_q.index > e - timedelta(days=FY_DAYS[1])) & (dps_q.index <= e)]
        if len(inside) == 4 and not inside["conflict"].any():
            t.loc[e, "dps"] = float(inside["val"].sum())
            dps_filed[e] = max(inside["filed"])

    # capex: ieșire de numerar, pozitivă; software-ul capitalizat se adaugă doar lângă eticheta PP&E (altfel e inclus)
    capex = t["capex"].abs()
    sw = t.pop("capex_software").abs()
    is_ppe = [e in frames["capex"].index and frames["capex"].loc[e, "concept"] == PPE_CAPEX for e in ends]
    t["capex"] = [c + s if ppe and not pd.isna(s) else c for c, s, ppe in zip(capex, sw, is_ppe)]
    t["fcf"] = t["cfo"] - t["capex"]

    # split-uri: doar cele de după depunerea valorii (raportul e deja ajustat pentru cele dinainte)
    def factor(name, e):
        df = frames[name]
        return split_factor_after(splits, df.loc[e, "filed"]) if e in df.index else 1.0

    sh_f = pd.Series([factor("shares_diluted", e) for e in ends], index=t.index)
    eps_f = pd.Series([factor("eps_diluted", e) for e in ends], index=t.index)
    dps_f = pd.Series([split_factor_after(splits, dps_filed[e]) for e in ends], index=t.index)
    eps_rep_raw = t.pop("eps_diluted")
    numer = t["net_income_common"].where(t["net_income_common"].notna(), t["net_income"])
    # scara acțiunilor, confirmată de profit / EPS raportat (din același raport, deci fără ajustare de split)
    scale = pd.Series([scale_correction(s, n / r if r else None)
                       for s, n, r in zip(t["shares_diluted"], numer, eps_rep_raw)], index=t.index)
    adj = t["shares_diluted"] * scale * sh_f
    # a doua verificare: anii vecini (EPS-ul raportat e prea rotunjit când profitul e aproape zero, ex. NVDA FY2009)
    for e in ends:
        if scale[e] != 1.0 or pd.isna(adj[e]) or adj[e] <= 0:
            continue
        others = adj.drop(index=e).dropna()
        others = others[others > 0]
        if len(others) < 2:
            continue
        r = float(others.median()) / float(adj[e])
        k = round(math.log10(r) / 3)
        if k != 0 and NEIGHBOR_SCALE_RANGE[0] <= r / 1000.0 ** k <= NEIGHBOR_SCALE_RANGE[1]:
            scale[e] = 1000.0 ** k
            adj[e] = adj[e] * scale[e]
    t["scale_fixed"] = scale != 1.0
    t["shares_diluted"] = adj
    t["eps_reported"] = eps_rep_raw / eps_f
    t["dps"] = t["dps"] / dps_f
    sh = t["shares_diluted"].where(t["shares_diluted"] > 0)
    t["eps"] = numer / sh
    t["eps_check"] = ((t["eps"] - t["eps_reported"]).abs() / t["eps_reported"].abs()).where(t["eps_reported"] != 0)
    t["revenue_ps"] = t["revenue"] / sh
    t["fcf_ps"] = t["fcf"] / sh
    t["bvps"] = t["equity"] / sh
    t["filed"] = [filed[e] for e in ends]
    t["conflicts"] = [conflicts[e] for e in ends]
    return t


def consistency(series):
    """(ani cu creștere față de anul anterior, ani comparați) pe o serie anuală ordonată; perechile cu lipsă se sar."""
    vals = [(_num(v)) for v in series]
    pairs = [(a, b) for a, b in zip(vals, vals[1:]) if a is not None and b is not None]
    return sum(1 for a, b in pairs if b > a), len(pairs)


def diagnose(cf):
    """Pentru proba de pe server: ([(indicator, etichetă, unitate, {formular: nr. valori})], [etichete de venit /
    profit din JSON care NU sunt în CONCEPTS, primele 15]). Pe JSON-ul complet (nu `slim`)."""
    facts = (cf or {}).get("facts") or {}
    out = []
    for ind, (_, _, concepts) in CONCEPTS.items():
        for tax, name in concepts:
            node = (facts.get(tax) or {}).get(name)
            if node is None:
                continue
            for u, rows in (node.get("units") or {}).items():
                forms = {}
                for r in rows:
                    forms[r.get("form")] = forms.get(r.get("form"), 0) + 1
                out.append((ind, name, u, forms))
    known = {n for _, _, lst in CONCEPTS.values() for _, n in lst}
    extra = []
    for n, node in sorted((facts.get("us-gaap") or {}).items()):
        if n in known or not any(k in n for k in ("Revenue", "NetIncome", "ProfitLoss", "PaymentsToAcquire",
                                                  "PaymentsFor", "CapitalExpenditure")):
            continue
        n10k = sum(1 for rows in (node.get("units") or {}).values() for r in rows if r.get("form") in ANNUAL_FORMS)
        if n10k:
            extra.append(f"{n} (10-K {n10k})")
    return out, extra[:30]
