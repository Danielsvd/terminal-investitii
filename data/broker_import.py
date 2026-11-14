"""Importul exporturilor de la brokeri (Tradeville, XTB, IBKR) în formatul foii `Tranzactii`.

Funcții pure peste rânduri deja citite din fișier (liste de liste); singurele funcții care
ating un fișier sunt `read_workbook` și `read_csv_rows`. Fiecare rând necunoscut e raportat,
nu ignorat tăcut.

Formatul de ieșire: dict-uri cu cheile din `analytics.ledger.COLUMNS`.
- `Amount` e efectul net în numerar, cu semn. `Fees` e informativ.
- `ID` e stabil între exporturi succesive, ca un rând deja importat să fie recunoscut:
  XTB-<id operațiune>, TV-<momentul operațiunii, la milisecundă>, IB-<TransactionID IBKR>.
- `Notes` nu conține numere de cont sau coduri de client.
"""
import csv
import datetime as dt
import io
import re
from zoneinfo import ZoneInfo

from analytics.ledger import COLUMNS

BUCHAREST, UTC = ZoneInfo("Europe/Bucharest"), ZoneInfo("UTC")
CASH_SYMBOLS = {"RON", "EUR", "USD"}
TV_BOND_MARKETS = {"REGT", "EREGT", "POFB", "EPOFB", "XRB"}
TV_BOND_SYMBOL = re.compile(r"R\d{4}AE?")
TV_RIGHTS_SYMBOL = re.compile(r".+R\d\d")

XTB_TYPES = {"Stock purchase": "BUY", "Stock sell": "SELL", "Dividend": "DIV", "Withholding tax": "TAX", "RO tax": "TAX",
             "SEC fee": "FEE", "Correction": "FEE", "Free funds interest": "INTEREST", "Deposit": "DEPOSIT",
             "Withdrawal": "WITHDRAW", "Transfer": "FX", "Subaccount transfer": "TRANSFER"}
XTB_TRADE = re.compile(r"^(OPEN|CLOSE) BUY (?:[A-Z0-9.]+ )?([\d.]+)(?:/[\d.]+)? @ ([\d.]+)$")
XTB_RO_TAX = re.compile(r"(?:RO tax|Romania Tax \d+%) ([A-Z0-9]+\.[A-Z]+)")
XTB_RATE = re.compile(r"(?:Exchange rate|Transaction rate):\s*([\d.]+)")
XTB_PAIR = re.compile(r"\b([A-Z]{3}) to ([A-Z]{3})\b")


def _num(value, digits=6):
    """Număr rotunjit pentru foaie; valoarea lipsă rămâne "" (celulă goală), nu 0."""
    if value is None or value == "":
        return ""
    number = round(float(value), digits)
    return int(number) if number == int(number) else number


def _stamp(moment):
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _record(**fields):
    row = {col: "" for col in COLUMNS}
    row.update(fields)
    return row


def read_workbook(file):
    """Citește un .xlsx: dict nume foaie -> listă de rânduri (tupluri de valori)."""
    import openpyxl
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # exporturile XTB nu au stil implicit; avertismentul e inofensiv
        workbook = openpyxl.load_workbook(file, data_only=True)
    return {ws.title: [tuple(r) for r in ws.iter_rows(values_only=True)] for ws in workbook.worksheets}


def read_csv_rows(file):
    """Citește un .csv (cale, bytes sau fișier încărcat) ca listă de rânduri (liste de text)."""
    if isinstance(file, (bytes, str)) and not (isinstance(file, str) and file.endswith(".csv")):
        raw = file
    elif hasattr(file, "read"):
        raw = file.read()
    else:
        with open(file, "rb") as handle:
            raw = handle.read()
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    return [row for row in csv.reader(io.StringIO(text)) if row]


def detect_broker(sheets):
    """"xtb", "tradeville", "ibkr" sau None, după structura fișierului."""
    for rows in sheets.values():
        if rows and len(rows[0]) > 1 and rows[0][0] in ("HEADER", "BOF") and str(rows[0][1]).isupper():
            return "ibkr"
    if "Cash Operations" in sheets:
        return "xtb"
    for rows in sheets.values():
        if rows and {"simbol", "cant", "pret", "suma", "valuta", "tcont"} <= {str(c) for c in rows[0] if c is not None}:
            return "tradeville"
    return None


# --- Tradeville -------------------------------------------------------------

def tradeville_class(symbol, market, entry_type):
    if entry_type == "RIGHTS":
        return "Drepturi"
    if market in TV_BOND_MARKETS or TV_BOND_SYMBOL.fullmatch(symbol or "") or symbol == "SBET24":
        return "Obligatiuni"
    if symbol == "EBGLD1":
        return "Altele"
    if symbol and "ETF" in symbol:
        return "ETF"
    return "Actiuni RO"


def parse_tradeville(rows):
    """Raportul de activitate Tradeville -> (înregistrări, probleme).

    Antetul are două coloane numite „data”: prima e momentul operațiunii, a doua tipul ei
    (cump, vanz, div, in, out, comis). Fișierul e în ordine descrescătoare.
    Dividendele sunt raportate nete: `cant` e suma după impozit, `comis` comisionul brokerului.
    """
    if not rows:
        return [], ["Fișier Tradeville gol."]
    header = [str(c) if c is not None else "" for c in rows[0]]
    header[0], header[1] = "ts", "tip"
    needed = {"simbol", "cant", "pret", "comis", "suma", "nrtranz", "valuta", "obs", "idord", "cont", "market", "simbolb"}
    if not needed <= set(header):
        return [], [f"Coloane lipsă în raportul Tradeville: {', '.join(sorted(needed - set(header)))}"]

    data = [dict(zip(header, r)) for r in rows[1:] if r and r[0] is not None]
    client = ""
    for d in data:
        if d.get("cont"):
            client = str(d["cont"]).split("-")[0].split(".")[0]
            break
    market_of = {d["simbol"]: d["market"] for d in data if d["tip"] in ("cump", "vanz")}

    records, problems, seen = [], [], {}
    for d in reversed(data):                                  # cronologic
        moment, tip, sym = d["ts"], d["tip"], d["simbol"]
        if not isinstance(moment, dt.datetime):
            problems.append(f"Tradeville: dată necitibilă ({moment!r}), rând ignorat.")
            continue
        obs = str(d["obs"] or "").strip()
        low = obs.lower()
        key = moment.strftime("%Y%m%d%H%M%S%f")[:17]
        seen[key] = seen.get(key, 0) + 1
        rec = dict(Date=_stamp(moment), Currency=d["valuta"], Fees=_num(d["comis"] or 0), Amount=_num(d["suma"] or 0),
                   Broker="Tradeville", ID=f"TV-{key}" + (f"-{seen[key]}" if seen[key] > 1 else ""),
                   Notes=re.sub(re.escape(client) + r"[.\-]?", "cont ", obs).strip() if client else obs)
        entry_type, symbol, quantity, price = None, "", "", ""
        if tip in ("cump", "vanz"):
            entry_type, symbol, quantity, price = ("BUY" if tip == "cump" else "SELL"), sym, _num(d["cant"]), _num(d["pret"])
        elif tip == "div":
            entry_type, symbol = ("COUPON" if "cupon" in low else "DIV"), (d["simbolb"] or "")
        elif tip == "comis":
            entry_type, rec["Fees"] = "FEE", 0
        elif tip in ("in", "out") and sym in CASH_SYMBOLS:
            if low.startswith("depunere"):
                entry_type = "DEPOSIT"
            elif low.startswith("retragere"):
                entry_type = "WITHDRAW"
            elif low.startswith("tf ") and "curs" in low:
                entry_type = "FX"
            elif low.startswith("transfer din"):
                entry_type = "TRANSFER"
            elif "fractie" in low:
                entry_type, symbol = "DIV", (d["simbolb"] or "")       # răscumpărare de fracție: numerar atribuit simbolului
            elif "taxa" in low:
                entry_type = "FEE"
        elif tip in ("in", "out"):
            symbol, quantity = sym, _num((1 if tip == "in" else -1) * d["cant"])
            if TV_RIGHTS_SYMBOL.fullmatch(sym or "") and "drepturi" in low:
                entry_type = "RIGHTS"
            elif "consolidare" in low:
                entry_type = "SPLIT"
            elif "gratuite" in low or "mcs" in low:
                entry_type = "BONUS"
        if entry_type is None:
            problems.append(f"Tradeville {moment:%Y-%m-%d}: operațiune necunoscută ({tip} {sym} „{obs}”), rând ignorat.")
            continue
        if symbol:
            cls = tradeville_class(symbol, market_of.get(symbol), entry_type)
            symbol = symbol if cls in ("Obligatiuni", "Drepturi", "Altele") else symbol + ".RO"
        else:
            cls = "Numerar"
        records.append(_record(Symbol=symbol, Class=cls, Type=entry_type, Quantity=quantity, Price=price, **rec))
    return records, problems


# --- XTB ---------------------------------------------------------------------

def xtb_symbol(ticker):
    """Simbolul XTB în format Yahoo: MSFT.US -> MSFT, EGLN.UK -> EGLN.L, SAP.DE rămâne."""
    if not ticker:
        return ""
    if ticker.endswith(".US"):
        return ticker[:-3]
    if ticker.endswith(".UK"):
        return ticker[:-3] + ".L"
    return ticker


def _table(rows, first_header):
    """Rândurile de sub antetul care începe cu `first_header`, ca dict-uri."""
    for i, row in enumerate(rows):
        if row and row[0] == first_header:
            header = [str(c) if c is not None else "" for c in row]
            return [dict(zip(header, r)) for r in rows[i + 1:] if r and r[0] is not None]
    return []


def xtb_currency(sheets):
    """Moneda contului, din rezumatul foii „Open Positions”; None dacă nu apare."""
    for row in sheets.get("Open Positions", []):
        if row and len(row) > 3 and row[1] == "Open position value" and row[3]:
            return str(row[3]).upper()
    return None


def parse_xtb(sheets, currency=None):
    """Exportul XTB (foaia „Cash Operations”) -> (înregistrări, probleme, control).

    Orele din export sunt UTC și se convertesc în ora României. Cantitatea și prețul
    tranzacțiilor se citesc din comentariu („OPEN BUY 1 @ 46.44”, „CLOSE BUY 1/4 @ 37.46”).
    `control`: {"reported": totalul din raport, "parsed": suma `Amount` citită}; dacă diferă,
    lipsește ceva din import.
    """
    currency = (currency or xtb_currency(sheets) or "").upper()
    if not currency:
        return [], ["XTB: moneda contului nu a putut fi dedusă din fișier; alege-o manual."], {}
    cash_rows = sheets.get("Cash Operations", [])
    operations = _table(cash_rows, "Type")
    category = {}
    for d in operations:
        if d.get("Ticker") and d.get("Category") in ("STOCK", "ETF", "ETC"):
            category[d["Ticker"]] = d["Category"]
    for d in _table(sheets.get("Closed Positions", []), "Instrument"):
        if d.get("Ticker") and d.get("Category") in ("STOCK", "ETF", "ETC"):
            category.setdefault(d["Ticker"], d["Category"])

    records, problems, reported = [], [], None
    for d in reversed(operations):
        kind, ticker, moment, amount = d["Type"], d.get("Ticker") or "", d.get("Time"), d.get("Amount")
        comment, product = str(d.get("Comment") or ""), str(d.get("Product") or "")
        if kind == "Total":
            reported = amount
            continue
        entry_type = XTB_TYPES.get(kind)
        if entry_type is None or not isinstance(moment, dt.datetime):
            problems.append(f"XTB {currency}: operațiune necunoscută („{kind}”, {comment[:60]}), rând ignorat.")
            continue
        quantity = price = ""
        if entry_type in ("BUY", "SELL"):
            match = XTB_TRADE.match(comment.strip())
            if not match or (match.group(1) == "OPEN") != (entry_type == "BUY"):
                problems.append(f"XTB {currency}: tranzacție cu comentariu necitibil („{comment[:60]}”), rând ignorat.")
                continue
            quantity, price = _num(float(match.group(2))), _num(float(match.group(3)))
        elif kind == "RO tax":
            match = XTB_RO_TAX.search(comment)
            ticker = match.group(1) if match else ""

        if entry_type == "DEPOSIT":
            note = "Depunere"
        elif entry_type == "WITHDRAW":
            note = "Retragere"
        elif entry_type == "TRANSFER":
            note = f"Transfer intre subconturi ({product})"
        elif entry_type == "FX":
            rate, pair = XTB_RATE.search(comment), XTB_PAIR.search(comment)
            note = "Schimb valutar" + (f" {pair.group(1)}>{pair.group(2)}" if pair else "") + (f" curs {rate.group(1)}" if rate else "")
        else:
            note = re.sub(r"\s*///.*$", "", comment).strip()

        if not ticker:
            cls = "Numerar"
        elif category.get(ticker) in ("ETF", "ETC"):
            cls = "ETF"
        elif category.get(ticker) == "STOCK":
            cls = "Actiuni SUA" if ticker.endswith(".US") else "Actiuni UE"
        else:
            problems.append(f"XTB {currency}: categorie necunoscută pentru {ticker}; clasa a rămas „Altele”.")
            cls = "Altele"
        local = moment.replace(tzinfo=UTC).astimezone(BUCHAREST).replace(tzinfo=None)
        records.append(_record(Date=_stamp(local), Symbol=xtb_symbol(ticker), Class=cls, Type=entry_type, Quantity=quantity,
                               Price=price, Currency=currency, Fees=0, Amount=_num(amount), Broker="XTB",
                               ID=f"XTB-{d.get('ID')}", Notes=note))
    control = {"reported": reported, "parsed": round(sum(r["Amount"] or 0 for r in records), 2)}
    return records, problems, control


# --- IBKR (Interactive Brokers) ----------------------------------------------
#
# Formatul așteptat: Activity Flex Query în CSV, cu „Include section code and line descriptor”
# (fiecare rând începe cu HEADER/DATA și codul secțiunii), fără header/trailer, date yyyy-MM-dd.
# Secțiuni folosite: TRNT (Trades, nivel EXECUTION), CTRN (Cash Transactions, nivel DETAIL),
# CRTT (Cash Report, pentru control), POST (Open Positions, pentru control).
# Orele din raport sunt ora New York-ului (fusul implicit al rapoartelor IBKR) și se
# convertesc în ora României.

NEW_YORK = ZoneInfo("America/New_York")
IBKR_US_EXCHANGES = {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "NYSEARCA", "NYSENAT", "IEX", "PINK", "BEX"}
IBKR_SUFFIX = {"IBIS": ".DE", "IBIS2": ".DE", "FWB": ".DE", "FWB2": ".DE", "LSE": ".L", "LSEETF": ".L",
               "AEB": ".AS", "SBF": ".PA", "BVME": ".MI", "BVME.ETF": ".MI", "EBS": ".SW", "BM": ".MC",
               "VSE": ".VI", "SFB": ".ST", "ENEXT.BE": ".BR"}
# Instrumente al căror simbol Yahoo nu se poate deduce din simbolul IBKR + bursă: ISIN -> (Yahoo, notă).
IBKR_YAHOO = {"IE00B4ND3602": ("PPFB.DE", "ETC aur")}      # iShares Physical Gold: EGLN la IBKR, PPFB pe Xetra la Yahoo
IBKR_CASH_TYPES = {"Deposits/Withdrawals": None, "Dividends": "DIV", "Payment In Lieu Of Dividends": "DIV",
                   "Withholding Tax": "TAX", "871(m) Withholding": "TAX", "Broker Interest Received": "INTEREST",
                   "Broker Interest Paid": "INTEREST", "Bond Interest Received": "COUPON", "Bond Interest Paid": "COUPON",
                   "Other Fees": "FEE", "Commission Adjustments": "FEE", "Broker Fees": "FEE", "Advisor Fees": "FEE"}


def ibkr_sections(rows):
    """Rândurile DATA grupate pe codul secțiunii: {cod: [dict coloană -> text]}."""
    headers, out = {}, {}
    for row in rows:
        if len(row) < 2:
            continue
        kind, code = row[0], row[1]
        if kind == "HEADER":
            headers[code] = list(row[2:])
            out.setdefault(code, [])
        elif kind == "DATA" and code in headers:
            out[code].append(dict(zip(headers[code], row[2:])))
    return out


def _f(text):
    """Text IBKR -> float; celula goală rămâne None (nu 0)."""
    text = str(text if text is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def ibkr_moment(text):
    """„2026-11-12;03:38:44” (ora New York) -> datetime în ora României; „2026-11-06” -> ziua la 00:00."""
    text = str(text or "").strip()
    for fmt in ("%Y-%m-%d;%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d;%H%M%S", "%Y-%m-%d, %H:%M:%S"):
        try:
            moment = dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
        return moment.replace(tzinfo=NEW_YORK).astimezone(BUCHAREST).replace(tzinfo=None)
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def ibkr_symbol(d):
    """Simbolul Yahoo pentru un rând IBKR -> (simbol, notă, problemă sau None)."""
    isin = (d.get("ISIN") or "").strip()
    if isin in IBKR_YAHOO:
        yahoo, note = IBKR_YAHOO[isin]
        return yahoo, note, None
    symbol = (d.get("Symbol") or "").strip()
    if not symbol:
        return "", "", None
    if d.get("AssetClass") == "OPT":
        return symbol.replace(" ", ""), "", None                  # simbolul OCC, scris ca la Yahoo (fără spații)
    exchange = (d.get("ListingExchange") or "").strip().upper()
    if exchange in IBKR_US_EXCHANGES:
        return symbol.replace(" ", "-"), "", None                 # BRK B -> BRK-B
    if exchange in IBKR_SUFFIX:
        return symbol.replace(" ", "-") + IBKR_SUFFIX[exchange], "", None
    return symbol, "", (f"IBKR: bursa „{exchange or '?'}” a lui {symbol} nu are corespondent Yahoo cunoscut; "
                        "simbolul a rămas cel IBKR (corectează-l în foaie dacă Yahoo nu-l găsește).")


def ibkr_class(d):
    """Clasa din foaie pentru un instrument IBKR."""
    asset, sub = d.get("AssetClass"), (d.get("SubCategory") or "").upper()
    if asset == "OPT":
        return "Optiuni"
    if asset == "BOND":
        return "Obligatiuni"
    if asset == "STK" and sub in ("ETF", "ETC", "ETN"):
        return "ETF"
    if asset == "STK":
        exchange = (d.get("ListingExchange") or "").strip().upper()
        return "Actiuni SUA" if exchange in IBKR_US_EXCHANGES else "Actiuni UE"
    return "Altele"


def _ibkr_id(d):
    return (d.get("TransactionID") or d.get("TradeID") or "").strip()


def _ibkr_trade(d, problems):
    """Un rând din Trades (nivel EXECUTION) -> înregistrări."""
    moment = ibkr_moment(d.get("DateTime")) or ibkr_moment(d.get("TradeDate"))
    tid = _ibkr_id(d)
    if moment is None or not tid:
        problems.append(f"IBKR: tranzacție fără dată sau ID ({d.get('Symbol')}), rând ignorat.")
        return []
    stamp, asset, symbol_ib = _stamp(moment), d.get("AssetClass"), (d.get("Symbol") or "").strip()
    currency = (d.get("CurrencyPrimary") or "").upper()
    quantity, proceeds = _f(d.get("Quantity")), _f(d.get("Proceeds"))
    taxes, commission = _f(d.get("Taxes")) or 0.0, _f(d.get("IBCommission")) or 0.0
    commission_cur = (d.get("IBCommissionCurrency") or currency).upper()
    if quantity is None or proceeds is None:
        problems.append(f"IBKR {stamp[:10]}: tranzacție {symbol_ib} fără cantitate sau sumă, rând ignorat.")
        return []

    fee_row = []
    if commission and (asset == "CASH" or commission_cur != currency):
        fee_row = [_record(Date=stamp, Symbol="", Class="Numerar", Type="FEE", Currency=commission_cur,
                           Fees=_num(-commission), Amount=_num(commission), Broker="IBKR", ID=f"IB-{tid}-FEE",
                           Notes=f"Comision {symbol_ib}")]

    if asset == "CASH":                                       # schimb valutar: EUR.USD, cantitatea e în prima monedă
        base, _, quote = symbol_ib.partition(".")
        if not base or not quote or quote.upper() != currency:
            problems.append(f"IBKR {stamp[:10]}: schimb valutar necitibil ({symbol_ib}), rând ignorat.")
            return []
        price = _f(d.get("TradePrice"))
        pair = f"{base}>{quote}" if quantity < 0 else f"{quote}>{base}"
        note = f"Schimb valutar {pair}" + (f" curs {price:g}" if price else "")
        legs = [_record(Date=stamp, Symbol="", Class="Numerar", Type="FX", Currency=base.upper(), Fees=0,
                        Amount=_num(quantity), Broker="IBKR", ID=f"IB-{tid}-{base.upper()}", Notes=note),
                _record(Date=stamp, Symbol="", Class="Numerar", Type="FX", Currency=currency, Fees=0,
                        Amount=_num(proceeds + taxes), Broker="IBKR", ID=f"IB-{tid}-{currency}", Notes=note)]
        return legs + fee_row

    side = (d.get("Buy/Sell") or "").upper()
    if side not in ("BUY", "SELL"):
        problems.append(f"IBKR {stamp[:10]}: tranzacție {symbol_ib} fără sens BUY/SELL („{side}”), rând ignorat.")
        return []
    symbol, note, problem = ibkr_symbol(d)
    if problem:
        problems.append(problem)
    cls = ibkr_class(d)
    if cls == "Altele":
        problems.append(f"IBKR: clasa {d.get('AssetClass')} a lui {symbol_ib} nu e tratată special; a intrat la „Altele”.")
    multiplier = _f(d.get("Multiplier")) or 1.0
    units = abs(quantity) * (multiplier if asset == "OPT" else 1.0)   # opțiuni: în acțiuni-suport, ca prețul pe acțiune
    amount = proceeds + taxes + (commission if commission_cur == currency else 0.0)
    if asset == "OPT" and side == "SELL" and (d.get("Open/CloseIndicator") or "").upper().startswith("O"):
        problems.append(f"IBKR {stamp[:10]}: opțiune vândută în deschidere (short) {symbol}: registrul nu urmărește încă "
                        "pozițiile short; numerarul e corect, poziția și profitul pe acest simbol nu.")
    parts = [p for p in (d.get("Description") or "", note,
                         f"IBKR: {symbol_ib} @ {d.get('Exchange') or d.get('ListingExchange') or '?'}") if p]
    if asset == "OPT":
        parts.append(f"{abs(quantity):g} contracte × {multiplier:g}")
    trade = _record(Date=stamp, Symbol=symbol, Class=cls, Type=side, Quantity=_num(units), Price=_num(_f(d.get("TradePrice"))),
                    Currency=currency, Fees=_num(-commission if commission_cur == currency else 0),
                    Amount=_num(amount), Broker="IBKR", ID=f"IB-{tid}", Notes=" · ".join(parts))
    return [trade] + fee_row


def _ibkr_cash(d, problems):
    """Un rând din Cash Transactions (nivel DETAIL) -> înregistrare sau None."""
    kind, amount = (d.get("Type") or "").strip(), _f(d.get("Amount"))
    moment = ibkr_moment(d.get("Date/Time")) or ibkr_moment(d.get("SettleDate"))
    tid = _ibkr_id(d)
    if kind not in IBKR_CASH_TYPES:
        problems.append(f"IBKR: operațiune de numerar necunoscută („{kind}”, {d.get('Description', '')[:50]}), rând ignorat.")
        return None
    if amount is None or moment is None or not tid:
        problems.append(f"IBKR: operațiune „{kind}” fără sumă, dată sau ID, rând ignorat.")
        return None
    entry_type = IBKR_CASH_TYPES[kind] or ("DEPOSIT" if amount >= 0 else "WITHDRAW")
    symbol, extra, problem = ibkr_symbol(d)
    if problem:
        problems.append(problem)
    cls = ibkr_class(d) if symbol else "Numerar"
    if entry_type in ("DEPOSIT", "WITHDRAW"):
        note = "Depunere" if entry_type == "DEPOSIT" else "Retragere"
    else:
        note = " · ".join(p for p in ((d.get("Description") or "").strip(), extra) if p)
    return _record(Date=_stamp(moment), Symbol=symbol, Class=cls, Type=entry_type, Currency=(d.get("CurrencyPrimary") or "").upper(),
                   Fees=0, Amount=_num(amount), Broker="IBKR", ID=f"IB-{tid}", Notes=note)


def parse_ibkr(rows):
    """Activity Flex Query IBKR (CSV cu coduri de secțiune) -> (înregistrări, probleme, control).

    - Trades: doar rândurile EXECUTION. Cumpărare/vânzare: `Amount` = încasări + taxe + comision
      (când comisionul e în moneda titlului). Schimb valutar (AssetClass CASH): două rânduri FX,
      câte unul pe monedă, plus comisionul ca rând FEE separat, în moneda în care a fost plătit.
    - Cash Transactions: doar rândurile DETAIL (IBKR repetă fiecare sumă și pe un rând SUMMARY,
      cu contul „-”; luate amândouă, depunerile s-ar dubla).
    - Opțiuni: cantitatea e în acțiuni-suport (contracte × multiplicator), ca valoarea = cantitate × preț.
    `control`: {"cash": {monedă: (raportat, citit)}, "positions": {simbol: cantitate}} — numerarul
    raportat e variația din Cash Report (sfârșit − început); pozițiile sunt cele de la finalul raportului.
    """
    sections = ibkr_sections(rows)
    if "TRNT" not in sections and "CTRN" not in sections:
        return [], ["IBKR: fișierul nu are secțiunile Trades și Cash Transactions (sau lipsesc codurile de secțiune: "
                    "în Flex Query trebuie bifat „Include section code and line descriptor”)."], {}
    records, problems = [], []
    for d in sections.get("TRNT", []):
        if d.get("LevelOfDetail", "EXECUTION").upper() == "EXECUTION":
            records.extend(_ibkr_trade(d, problems))
    for d in sections.get("CTRN", []):
        if d.get("LevelOfDetail", "DETAIL").upper() == "DETAIL" and d.get("ClientAccountID") != "-":
            rec = _ibkr_cash(d, problems)
            if rec:
                records.append(rec)
    if sections.get("OPTT"):
        problems.append(f"IBKR: {len(sections['OPTT'])} exercitări/asignări/expirări de opțiuni în raport; "
                        "verifică în registru cantitățile de opțiuni și de acțiuni-suport.")
    if sections.get("TRFR"):
        problems.append(f"IBKR: {len(sections['TRFR'])} transferuri de titluri în raport; nu sunt importate.")
    if sections.get("CORP"):
        problems.append(f"IBKR: {len(sections['CORP'])} acțiuni corporative (split, fuziune) în raport; nu sunt importate.")

    cash = {}
    for d in sections.get("CRTT", []):
        if (d.get("LevelOfDetail") or "").lower() == "currency":
            cur = (d.get("CurrencyPrimary") or "").upper()
            start, end = _f(d.get("StartingCash")), _f(d.get("EndingCash"))
            if cur and start is not None and end is not None:
                parsed = sum(r["Amount"] or 0 for r in records if r["Currency"] == cur)
                cash[cur] = (round(end - start, 2), round(parsed, 2))
    positions = {}
    for d in sections.get("POST", []):
        if (d.get("LevelOfDetail") or "SUMMARY").upper() == "SUMMARY":
            symbol, _, _ = ibkr_symbol(d)
            qty = _f(d.get("Quantity"))
            if symbol and qty is not None:
                mult = (_f(d.get("Multiplier")) or 1.0) if d.get("AssetClass") == "OPT" else 1.0
                positions[symbol] = positions.get(symbol, 0.0) + qty * mult
    return records, problems, {"cash": cash, "positions": positions}


def position_gaps(records, reported, broker="IBKR"):
    """Cantitățile din registru (doar `broker`) față de cele raportate de broker.

    `records`: dict-uri cu Symbol, Type, Quantity, Broker (registrul din foaie + rândurile noi).
    Întoarce [(simbol, registru, broker)] pentru simbolurile care diferă (inclusiv cele vândute
    complet în registru dar încă raportate, și invers).
    """
    held = {}
    for rec in records:
        if rec.get("Broker") != broker or not rec.get("Symbol"):
            continue
        typ, qty = str(rec.get("Type") or "").upper(), rec.get("Quantity")
        if typ not in ("OPEN", "BUY", "SELL", "BONUS", "SPLIT") or qty in (None, "") or qty != qty:
            continue
        held[rec["Symbol"]] = held.get(rec["Symbol"], 0.0) + (-float(qty) if typ == "SELL" else float(qty))
    gaps = []
    for symbol in sorted(set(held) | set(reported)):
        mine, theirs = held.get(symbol, 0.0), reported.get(symbol, 0.0)
        if abs(mine - theirs) > 1e-6:
            gaps.append((symbol, round(mine, 6), round(theirs, 6)))
    return gaps


# --- Comun -------------------------------------------------------------------

def parse_export(sheets, currency=None):
    """Recunoaște brokerul și întoarce (broker, înregistrări, probleme, control)."""
    broker = detect_broker(sheets)
    if broker == "xtb":
        records, problems, control = parse_xtb(sheets, currency)
        return "XTB " + (currency or xtb_currency(sheets) or "?"), records, problems, control
    if broker == "ibkr":
        rows = next(r for r in sheets.values() if r and r[0] and r[0][0] in ("HEADER", "BOF"))
        records, problems, control = parse_ibkr(rows)
        return "IBKR", records, problems, control
    if broker == "tradeville":
        rows = next(r for r in sheets.values() if r and "simbol" in {str(c) for c in r[0] if c is not None})
        records, problems = parse_tradeville(rows)
        return "Tradeville", records, problems, {}
    return None, [], ["Fișier nerecunoscut: nu e nici raport de activitate Tradeville, nici export XTB, "
                      "nici Flex Query IBKR (CSV cu coduri de secțiune)."], {}


def new_records(records, existing_ids):
    """Înregistrările al căror ID nu e deja în foaie, în ordine cronologică, fără dubluri între ele."""
    known, out = set(existing_ids), []
    for rec in sorted(records, key=lambda r: (r["Date"], r["ID"])):
        if rec["ID"] and rec["ID"] not in known:
            known.add(rec["ID"])
            out.append(rec)
    return out


def to_sheet_rows(records):
    """Rânduri pentru `append_rows`, în ordinea coloanelor foii."""
    return [[rec[col] for col in COLUMNS] for rec in records]
