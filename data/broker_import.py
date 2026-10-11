"""Importul exporturilor de la brokeri (Tradeville, XTB) în formatul foii `Tranzactii`.

Funcții pure peste rânduri deja citite din fișier (liste de liste); singura funcție care
atinge un fișier e `read_workbook`. Fiecare rând necunoscut e raportat, nu ignorat tăcut.

Formatul de ieșire: dict-uri cu cheile din `analytics.ledger.COLUMNS`.
- `Amount` e efectul net în numerar, cu semn. `Fees` e informativ.
- `ID` e stabil între exporturi succesive, ca un rând deja importat să fie recunoscut:
  XTB-<id operațiune>, respectiv TV-<momentul operațiunii, la milisecundă>.
- `Notes` nu conține numere de cont sau coduri de client.
"""
import datetime as dt
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


def detect_broker(sheets):
    """"xtb", "tradeville" sau None, după structura fișierului."""
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


# --- Comun -------------------------------------------------------------------

def parse_export(sheets, currency=None):
    """Recunoaște brokerul și întoarce (broker, înregistrări, probleme, control)."""
    broker = detect_broker(sheets)
    if broker == "xtb":
        records, problems, control = parse_xtb(sheets, currency)
        return "XTB " + (currency or xtb_currency(sheets) or "?"), records, problems, control
    if broker == "tradeville":
        rows = next(r for r in sheets.values() if r and "simbol" in {str(c) for c in r[0] if c is not None})
        records, problems = parse_tradeville(rows)
        return "Tradeville", records, problems, {}
    return None, [], ["Fișier nerecunoscut: nu e nici raport de activitate Tradeville, nici export XTB."], {}


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
