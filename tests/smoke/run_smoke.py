"""Test de fum: rulează toată aplicația cu date simulate și raportează excepțiile.

Nu atinge rețeaua și nu are nevoie de chei: yfinance, FRED, RSS și Google Sheets
sunt înlocuite cu date sintetice, iar modelele grele (Prophet, FinBERT) cu
substitute. Verifică un singur lucru: că fiecare secțiune se execută până la
capăt, inclusiv pe simboluri cu date lipsă (None), ca la BVB și la ETF-uri.
NU verifică dacă cifrele reale de la Yahoo sunt corecte.

Rulare, din rădăcina repo-ului:
    pip install -r requirements.txt pytest      # torch, transformers și prophet nu sunt necesare aici
    python tests/smoke/run_smoke.py

Cod de ieșire 0 = nicio excepție și niciun mesaj de eroare neașteptat.
"""
import os
import re
import sys
import time
import types

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else ROOT
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)
from data import fred_calendar as fred_cal   # noqa: E402  (deciziile FOMC din calendarul Fed)

TODAY = pd.Timestamp.today().normalize()
NO_DATA = {"NODATA.RO"}            # simbol fără nicio cotație
NEW_LISTING = {"NEWCO"}            # simbol listat de 60 de ședințe


# --------------------------------------------------------------------------
# Date sintetice
# --------------------------------------------------------------------------
def _seed(sym):
    return sum(ord(c) * (i + 1) for i, c in enumerate(sym)) % (2**31)


def _ohlcv(sym, n, tz=None):
    """Mers aleator determinist, cu o corecție de 25% la mijloc (ca stopul ATR să fie atins)."""
    idx = pd.bdate_range(end=TODAY, periods=n, tz=tz)
    if sym in NO_DATA:
        return pd.DataFrame(np.nan, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])
    rng = np.random.default_rng(_seed(sym))
    rets = rng.normal(0.0004, 0.015, n)
    if n > 200:
        rets[n // 2: n // 2 + 10] = -0.03
    close = (20 + _seed(sym) % 300) * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.003, n)),
        "High": close * 1.012,
        "Low": close * 0.988,
        "Close": close,
        "Volume": rng.integers(50_000, 5_000_000, n).astype(float),
    }, index=idx)
    if sym in NEW_LISTING:
        df.iloc[:-60] = np.nan
    return df


def _n_rows(period=None, start=None):
    if start is not None:
        return max(len(pd.bdate_range(pd.Timestamp(start).tz_localize(None), TODAY)), 2)
    return {"1d": 1, "5d": 5, "1mo": 22, "3mo": 64, "6mo": 127, "1y": 252, "2y": 504, "3y": 756, "5y": 1260}.get(period, 252)


RICH_INFO = dict(
    longName="Compania Test SA", sector="Technology", industry="Software", currency="USD",
    marketCap=2.5e12, trailingPE=28.0, forwardPE=25.0, priceToBook=9.0, trailingEps=6.1, bookValue=4.2,
    returnOnEquity=0.35, returnOnAssets=0.18, profitMargins=0.24, operatingMargins=0.30,
    debtToEquity=140.0, currentRatio=1.1, quickRatio=0.9, beta=1.2, payoutRatio=0.15, dividendRate=0.96,
    totalRevenue=3.8e11, netIncomeToCommon=9.7e10, totalCash=6e10, totalDebt=1.1e11,
    operatingCashflow=1.1e11, currentPrice=180.0, previousClose=178.5,
    recommendationKey="buy", recommendationMean=1.9, targetMeanPrice=210.0,
    heldPercentInstitutions=0.61, fullTimeEmployees=150000, city="Cupertino", country="US",
    website="https://example.com", longBusinessSummary="Descriere de test. " * 60,
    companyOfficers=[{"name": "Ion Popescu", "title": "CEO & Director"}],
)
# Ca la BVB / ETF-uri: cheile există, dar multe au valoarea None
SPARSE_INFO = dict(
    longName=None, sector=None, industry=None, currency="RON",
    marketCap=None, trailingPE=None, forwardPE=None, priceToBook=None, trailingEps=None, bookValue=None,
    returnOnEquity=None, returnOnAssets=None, profitMargins=None, operatingMargins=None,
    debtToEquity=None, currentRatio=None, quickRatio=None, beta=None, payoutRatio=None, dividendRate=None,
    totalRevenue=None, netIncomeToCommon=None, totalCash=None, totalDebt=None, operatingCashflow=None,
    currentPrice=None, previousClose=None, recommendationKey=None, recommendationMean=None,
    targetMeanPrice=None, heldPercentInstitutions=None, fullTimeEmployees=None, city=None, country=None,
    website=None, longBusinessSummary=None, companyOfficers=None,
)
# Companie cu pierderi: profit net negativ, EPS negativ
LOSS_INFO = dict(RICH_INFO, longName="Pierderi Corp", sector="Financial Services", trailingEps=-1.2,
                 trailingPE=None, netIncomeToCommon=-5e8, profitMargins=-0.12, returnOnEquity=-0.2,
                 debtToEquity=900.0, payoutRatio=None, dividendRate=None)


def _info_for(sym):
    if sym == "EMPTY":   # ca pe Streamlit Cloud când Yahoo refuză endpoint-ul de fundamentale
        return {"trailingPegRatio": None}
    if sym.endswith(".RO") or sym.endswith(".DE"):
        return dict(SPARSE_INFO)
    if sym == "LOSS":
        return dict(LOSS_INFO)
    if sym.endswith(".PA"):   # companie din zona euro cu date complete: AAA în EUR = proxy
        return dict(RICH_INFO, currency="EUR", financialCurrency="EUR")
    return dict(RICH_INFO)


class _FastInfo:
    def __init__(self, sym):
        df = _ohlcv(sym, 5)
        self.last_price = float(df["Close"].iloc[-1]) if df["Close"].notna().any() else None
        self.previous_close = float(df["Close"].iloc[-2]) if df["Close"].notna().any() else None
        self.currency = "RON" if sym.endswith(".RO") else ("EUR" if sym.endswith((".DE", ".PA")) else "USD")
        self.market_cap = 1e9


class FakeTicker:
    def __init__(self, sym, *a, **k):
        self.ticker = sym

    def history(self, period="1mo", **k):
        if self.ticker in NO_DATA or self.ticker == "INVALID":
            return pd.DataFrame()
        df = _ohlcv(self.ticker, _n_rows(period), tz="America/New_York").dropna(how="all")
        if self.ticker == "^TNX":      # randament în procente, ca la Yahoo (4,2 = 4,2%)
            df = df.assign(Close=4.2)
        if self.ticker == "^IRX":      # titluri SUA 3 luni, în procente
            df = df.assign(Close=3.9)
        df.index.name = "Date"
        return df

    @property
    def info(self):
        return _info_for(self.ticker)

    @property
    def fast_info(self):
        return _FastInfo(self.ticker)

    @property
    def earnings_history(self):
        if self.ticker.endswith(".RO"):
            return None
        return pd.DataFrame({"epsEstimate": [1.0, 1.1], "epsActual": [1.05, 1.0],
                             "epsDifference": [0.05, -0.1], "surprisePercent": [0.05, -0.09]},
                            index=pd.to_datetime(["2025-12-31", "2026-03-31"]))

    def _fin(self, freq, n):
        cols = pd.date_range(end=TODAY, periods=n, freq=freq)[::-1]
        if self.ticker.endswith(".RO"):
            return pd.DataFrame()
        return pd.DataFrame([[1e9 * (i + 5) for i in range(n)], [1e8 * (i - 1) for i in range(n)],
                             [1.55e10] * n, [3.5e9] * n, [1.8e10] * n, [1.15e11] * n, [3e8 * (i + 5) for i in range(n)],
                             [4e8 * (i + 5) for i in range(n)], [3.6e8 * (i + 5) for i in range(n)]],
                            index=["Total Revenue", "Net Income", "Diluted Average Shares",
                                   "Interest Expense", "Tax Provision", "Pretax Income", "EBIT", "Gross Profit", "EBITDA"], columns=cols)

    def _stmt(self, freq, n, rows):
        """Bilanț / flux de numerar sintetic. .RO: gol (ca la Yahoo). NEWCO: capex peste CFO (FCF negativ)."""
        if self.ticker.endswith(".RO"):
            return pd.DataFrame()
        cols = pd.date_range(end=TODAY, periods=n, freq=freq)[::-1]
        scale = 0.25 if n == 5 else 1.0     # trimestrele sunt un sfert din an
        data = {name: [value * (scale if flow else 1.0) * (1 - 0.05 * i) for i in range(n)]
                for name, (value, flow) in rows.items()}
        return pd.DataFrame(data, index=cols).T

    _BALANCE = {"Total Debt": (1.1e11, False), "Cash Cash Equivalents And Short Term Investments": (6e10, False),
                "Ordinary Shares Number": (1.5e10, False), "Total Assets": (3.6e11, False),
                "Stockholders Equity": (7e10, False), "Current Assets": (1.4e11, False),
                "Current Liabilities": (1.3e11, False), "Inventory": (7e9, False),
                "Retained Earnings": (2e10, False), "Total Liabilities Net Minority Interest": (2.9e11, False),
                "Long Term Debt": (9e10, False)}

    def _cash_rows(self):
        capex = -3e11 if self.ticker == "NEWCO" else -1.0e10
        return {"Operating Cash Flow": (1.1e11, True), "Capital Expenditure": (capex, True)}

    @property
    def balance_sheet(self):
        return self._stmt(pd.offsets.YearEnd(), 4, self._BALANCE)

    @property
    def quarterly_balance_sheet(self):
        return self._stmt(pd.offsets.QuarterEnd(), 5, self._BALANCE)

    @property
    def cashflow(self):
        return self._stmt(pd.offsets.YearEnd(), 4, self._cash_rows())

    @property
    def quarterly_cashflow(self):
        return self._stmt(pd.offsets.QuarterEnd(), 5, self._cash_rows())

    @property
    def financials(self):
        return self._fin(pd.offsets.YearEnd(), 4)

    @property
    def quarterly_financials(self):
        return self._fin(pd.offsets.QuarterEnd(), 5)

    @property
    def options(self):
        if "." in self.ticker or self.ticker == "EMPTY":
            return ()
        return tuple((TODAY + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in (3, 31, 59))

    def option_chain(self, exp):
        spot = _FastInfo(self.ticker).last_price
        strikes = np.round(spot * np.linspace(0.8, 1.2, 17), 0)
        rng = np.random.default_rng(_seed(self.ticker))

        def side():
            return pd.DataFrame({"strike": strikes,
                                 "openInterest": rng.integers(0, 5000, len(strikes)).astype(float),
                                 "volume": rng.integers(0, 2000, len(strikes)).astype(float),
                                 "impliedVolatility": rng.uniform(0.15, 0.6, len(strikes)),
                                 "bid": np.full(len(strikes), 1.0)})
        calls, puts = side(), side()
        calls.loc[0, "openInterest"] = np.nan   # Yahoo trimite uneori NaN
        return types.SimpleNamespace(calls=calls, puts=puts)

    @property
    def major_holders(self):
        if self.ticker.endswith(".RO") or self.ticker == "EMPTY":
            return pd.DataFrame()
        # Formatul yfinance 1.x, inclusiv rândurile care NU sunt deținerea totală (număr, free float)
        return pd.DataFrame({"Value": [0.021, 0.61, 0.63, 3500.0]},
                            index=["insidersPercentHeld", "institutionsPercentHeld",
                                   "institutionsFloatPercentHeld", "institutionsCount"])

    @property
    def institutional_holders(self):
        if self.ticker.endswith(".RO"):
            return None
        return pd.DataFrame({"Holder": ["Fond A", "Fond B"], "pctHeld": [0.08, 0.07], "Value": [2e11, 1.7e11]})


def fake_download(tickers, period=None, start=None, group_by="column", **k):
    """Imită forma din yfinance 1.x: MultiIndex mereu, (câmp, simbol) sau (simbol, câmp)."""
    if isinstance(tickers, str):
        tickers = tickers.split()
    n = _n_rows(period, start)
    frames = {t: _ohlcv(t, n) for t in tickers}
    if group_by == "ticker":
        out = pd.concat(frames, axis=1)
    else:
        out = pd.concat(frames, axis=1).swaplevel(axis=1).sort_index(axis=1)
    out.index.name = "Date"
    return out


class FakeTickers:
    def __init__(self, symbols, *a, **k):
        if isinstance(symbols, str):
            symbols = symbols.split()
        self.tickers = {s: FakeTicker(s) for s in symbols}


FRED_DAILY = {"DFF": 4.33, "DGS3MO": 4.20, "DGS2": 3.90, "DGS10": 4.40, "BAMLH0A0HYM2": 3.10, "BAMLC0A0CM": 0.85}


def fake_fred(code, source, start, end):
    if code in FRED_DAILY:                      # randamente zilnice (dobânzi comparate), ușor crescătoare
        didx = pd.bdate_range(end=TODAY, periods=560)
        return pd.DataFrame({code: [FRED_DAILY[code] + 0.001 * i for i in range(len(didx))]}, index=didx)
    idx = pd.date_range(end=TODAY - pd.offsets.MonthBegin(2), periods=16, freq="MS")
    base = {"CPIAUCSL": 320.0, "CPILFESL": 325.0, "PCEPILFE": 125.0, "UNRATE": 4.2, "FEDFUNDS": 4.3,
            "PAYEMS": 159000.0, "ADPCHGS": 134000.0, "JTSJOL": 7400.0, "RSAFS": 720000.0,
            "INDPRO": 103.0, "HOUST": 1350.0, "UMCSENT": 62.0, "IRLTLT01DEM156N": 3.1, "AAA": 5.4, "GS10": 4.5}[code]
    step = 0.0025 if code not in ("UNRATE", "FEDFUNDS", "IRLTLT01DEM156N", "AAA", "GS10") else 0.0
    return pd.DataFrame({code: [base * (1 + step) ** i for i in range(len(idx))]}, index=idx)


def fake_feed(url):
    # Pe fluxul unui simbol, știrile pare îl menționează (rămân după filtrul pe emitent), cele impare nu.
    sym = url.split("s=", 1)[1].split("&", 1)[0].split(".")[0] if "s=" in url else ""

    def entry(i):
        ts = time.gmtime(time.time() - i * 3600)
        about = f"{sym} " if sym and i % 2 == 0 else ""
        return types.SimpleNamespace(
            title=f"Știre <b>test</b> {i}: {about}banca centrală și tehnologia AI", link="https://example.com/a",
            summary="<p>Rezumat cu <a href='x'>HTML</a> netăiat corect " + "text " * 80,
            published_parsed=ts, updated_parsed=ts, published="x")
    feed = types.SimpleNamespace(get=lambda k, d=None: "Ziarul Financiar Test" if k == "title" else d)
    return types.SimpleNamespace(entries=[entry(i) for i in range(6)], feed=feed)


# --------------------------------------------------------------------------
# Google Sheets simulat (date inventate; nu conține portofoliul real)
# --------------------------------------------------------------------------
SHEETS = {
    None: [  # prima foaie: portofoliu
        {"Symbol": "AAPL", "Date": "2025-01-10", "Quantity": 0.31, "AvgPrice": 150.0, "Currency": "USD"},
        {"Symbol": "SAP.DE", "Date": "10-02-2025", "Quantity": "5", "AvgPrice": "100,5", "Currency": "EUR"},
        {"Symbol": "NEWCO", "Date": "2026-06-01", "Quantity": 3, "AvgPrice": 20.0, "Currency": "EUR"},
        {"Symbol": "TLV.RO", "Date": "2024-02-10", "Quantity": 367, "AvgPrice": 14.891, "Currency": "RON"},
        {"Symbol": "NODATA.RO", "Date": "2024-02-10", "Quantity": 100, "AvgPrice": 2.0, "Currency": "RON"},
    ],
    "watchlist": [
        {"Symbol": "MSFT", "TargetPrice": 300, "Notes": "test"},
        {"Symbol": "TLV.RO", "TargetPrice": "25,5", "Notes": ""},
    ],
    "BVB": [["Multipli", "Indicator", "TLV", "SNP"], ["", "Multipli de preț", "", ""],
            ["", "P/E TTM ", "8,5", "6,1"], ["", "Rentabilitate capital (ROE)", "22,10%", "#DIV/0!"],
            ["", "EPS TTM", "1,8417 lei", "0,05 lei"]],
    "Tranzactii": [
        ["Date", "Symbol", "Class", "Type", "Quantity", "Price", "Currency", "Fees", "Amount", "Broker", "ID", "Notes"],
        ["2024-01-02 10:00:00", "", "Numerar", "DEPOSIT", "", "", "RON", "0", "5000", "BrokerA", "d1", ""],
        ["2024-01-03 10:00:00", "TLV.RO", "Actiuni RO", "BUY", "100", "20", "RON", "5", "-2005", "BrokerA", "t1", ""],
        ["2024-06-03 10:00:00", "TLV.RO", "Actiuni RO", "BONUS", "10", "", "RON", "0", "0", "BrokerA", "t2", ""],
        ["2024-07-01 10:00:00", "TLV.RO", "Actiuni RO", "DIV", "", "", "RON", "1", "99,5", "BrokerA", "t3", ""],
        ["2024-09-02 10:00:00", "TLV.RO", "Actiuni RO", "SELL", "30", "25", "RON", "3", "747", "BrokerA", "t4", ""],
        ["2024-02-01 10:00:00", "", "Numerar", "DEPOSIT", "", "", "EUR", "0", "1000", "BrokerB", "d2", ""],
        ["2024-02-02 10:00:00", "SAP.DE", "Actiuni UE", "BUY", "2", "150,5", "EUR", "0", "-301", "BrokerB", "e1", ""],
        ["2024-02-05 10:00:00", "R9901AE", "Obligatiuni", "BUY", "3", "99", "EUR", "0", "-297", "BrokerA", "o1", ""],
        ["2024-08-05 10:00:00", "R9901AE", "Obligatiuni", "COUPON", "", "", "EUR", "0", "15", "BrokerA", "o2", ""],
        ["2024-03-01 10:00:00", "", "Numerar", "FX", "", "", "USD", "0", "200", "BrokerB", "f1", ""],
        ["2024-03-04 10:00:00", "AAPL", "Actiuni SUA", "BUY", "0,5", "180", "USD", "0", "-90", "BrokerB", "u1", ""],
        ["2024-05-16 10:00:00", "AAPL", "Actiuni SUA", "DIV", "", "", "USD", "0", "0,12", "BrokerB", "u2", ""],
        ["2024-05-16 10:00:00", "AAPL", "Actiuni SUA", "TAX", "", "", "USD", "0", "-0,01", "BrokerB", "u3", ""],
        ["data greșită", "AAPL", "Actiuni SUA", "BUY", "1", "1", "USD", "0", "-1", "BrokerB", "u4", ""],
    ],
    # preț manual pentru obligațiunea din registrul fals; data de azi, ca să nu apară avertismentul de preț vechi
    "Preturi_manuale": [["Symbol", "Price", "Date"], ["R9901AE", "101,25", __import__("datetime").date.today().isoformat()],
                        ["R9902AE", "", ""]],
    # istoric IV la bani sintetic pentru AAPL: 25 de zile, IV 0,20 … 0,44
    "IV_istoric": [["Date", "Symbol", "IV_ATM"]] + [
        [(__import__("datetime").date.today() - __import__("datetime").timedelta(days=40 - i)).isoformat(), "AAPL", (20 + i) / 100]
        for i in range(25)],
    "GLOBAL": [["Industrie", "Companii", "Capitalizare", "Preț actiune", "P/E", "ROE", "Recomandare"],
               ["Tech", "AAPL", "2.500.000.000.000 $", "180,5 $", "28,1", "35,2%", "Buy"],
               ["", "SAP", "200.000.000.000€", "150,5€", "#DIV/0!", "-%", "Ține"]],
}


class FakeWorksheet:
    def __init__(self, name):
        self.name = name

    def get_all_records(self, *a, **k):
        return [dict(r) for r in SHEETS[self.name]]

    def get_all_values(self, *a, **k):
        data = SHEETS[self.name]
        if data and isinstance(data[0], dict):
            return [list(data[0].keys())] + [[str(v) for v in r.values()] for r in data]
        return [list(r) for r in data]

    def append_row(self, row, *a, **k):
        pass

    def append_rows(self, rows, *a, **k):
        pass

    def find(self, *a, **k):
        return None

    def delete_rows(self, *a, **k):
        pass


class FakeSpreadsheet:
    sheet1 = FakeWorksheet(None)

    def worksheet(self, name):
        if name not in SHEETS:
            import gspread
            raise gspread.exceptions.WorksheetNotFound(name)
        return FakeWorksheet(name)


# Fișierele „Analiză companii” (Drive): un emitent cu șablon de bancă (sărit) și 8 emitenți cu date cumulate.
COMPANY_FILES = ["0-Dashboard", "TLV", "SNP", "TGN", "SNG", "EL", "H2O", "TEL", "WINE", "DN Agrar"]
_RAW_LABELS = ["Datorii totale", "Cifră de afaceri", "Profit Net", "Profit distribuit", "Ebitda",
               "Rezultat operațional", "Capital propriu (Activ Net)", "Conturi la bănci", "Datorii termen scurt",
               "Datorii termen lung", "Active totale", "Active imobilizate", "Active circulante (curente)",
               "Stocuri", "Creanțe", "Preț curent acțiune", "Nr. de acțiuni emise"]


def _company_sheets(name):
    """(Date Q, Date Anuale) sintetice, ca valorile nevizualizate din Google Sheets."""
    if name == "TLV":
        return [["Criteriu", "Q1 2025"], ["Datorii la clienți (depozite)", 1e9]], [["Criteriu", "2024"]]
    k = 1 + _seed(name) % 7
    sym = f"{name.split()[0].upper()}.RO"
    # Prețul din fișier = prețul pe care îl va folosi aplicația (live reușit sau ultima închidere), ca să nu
    # declanșeze avertismentul de split pe datele sintetice.
    price = _FastInfo(sym).last_price if _seed(sym) % 2 == 0 else float(_ohlcv(sym, 1260)["Close"].iloc[-1])
    q_per = [f"Q{i} {y}" for y in (2024, 2025, 2026) for i in (1, 2, 3)][:-1] + ["Q3 2026"]
    a_per = ["2023", "2024", "2025", "2026", "2025/2024", "Media"]

    def block(periods, annual):
        rows = [["Criteriu"] + periods, ["Marjă netă"] + ["#DIV/0!"] * len(periods)]
        for lab in _RAW_LABELS:
            vals = []
            for j, per in enumerate(periods):
                future = per.startswith("Q3 2026") or per in ("2026", "2025/2024", "Media")
                if future:
                    vals.append(0 if lab != "Preț curent acțiune" else 10)
                    continue
                months = 12 if annual else 3 * int(per[1])
                base = (1 + 0.1 * j) * k * 1e7
                v = {"Cifră de afaceri": base * months, "Profit Net": base * months * 0.1,
                     "Ebitda": base * months * 0.2, "Rezultat operațional": base * months * 0.15,
                     "Profit distribuit": base * 0.5 if annual else 0,
                     "Capital propriu (Activ Net)": 5e8 * k, "Datorii totale": 3e8 * k,
                     "Active totale": 8e8 * k, "Active imobilizate": 5e8 * k, "Active circulante (curente)": 3e8 * k,
                     "Datorii termen scurt": 1e8 * k, "Datorii termen lung": 2e8 * k - j * 1e6,
                     "Stocuri": 5e7, "Creanțe": 8e7, "Conturi la bănci": 6e7,
                     "Preț curent acțiune": price, "Nr. de acțiuni emise": 1e8}[lab]
                vals.append(v)
            rows.append([lab] + vals)
        return rows

    return block(q_per, False), block(a_per, True)


class FakeHTTPClient:
    def request(self, method, url, params=None, **k):
        return types.SimpleNamespace(json=lambda: {"files": [{"id": "folder-1", "name": "Analiză companii"}]})

    def values_batch_get(self, file_id, ranges, params=None):
        q, a = _company_sheets(file_id)
        return {"valueRanges": [{"values": q}, {"values": a}]}


class FakeClient:
    http_client = FakeHTTPClient()

    def open(self, name):
        return FakeSpreadsheet()

    def list_spreadsheet_files(self, title=None, folder_id=None):
        return [{"id": n, "name": n, "modifiedTime": "2026-10-01T00:00:00Z"} for n in COMPANY_FILES]


# --------------------------------------------------------------------------
# Instalarea substitutelor
# --------------------------------------------------------------------------
def install_fakes():
    import feedparser
    import gspread
    import httpx
    import pandas_datareader.data as web
    import yfinance as yf
    from google.oauth2.service_account import Credentials

    yf.Ticker, yf.Tickers, yf.download = FakeTicker, FakeTickers, fake_download
    web.DataReader = fake_fred
    feedparser.parse = fake_feed
    gspread.authorize = lambda creds: FakeClient()
    Credentials.from_service_account_info = classmethod(lambda cls, info, scopes=None: object())

    # Prețurile live: jumătate reușesc, jumătate eșuează (ca la un 429), fără rețea
    class _Resp:
        def __init__(self, sym):
            self.status_code = 200 if _seed(sym) % 2 == 0 and sym not in NO_DATA else 429
            self._sym = sym

        def json(self):
            return {"chart": {"result": [{"meta": {"regularMarketPrice": _FastInfo(self._sym).last_price}}]}}

    class _AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **k):
            return _Resp(url.split("/chart/")[1].split("?")[0])

    httpx.AsyncClient = _AsyncClient

    # BCE: randamentul titlurilor de stat RO pe 10 ani (rata fără risc pentru RON)
    import requests

    EUROSTAT_ROBOR = {"id": ["freq", "int_rt", "geo", "time"], "size": [1, 1, 1, 2],
                      "dimension": {"time": {"category": {"index": {"2026-07": 0, "2026-08": 1}}}},
                      "value": {"0": 6.02, "1": 5.87}}

    def _bnr_xml(days):
        cubes = "".join(f'<Cube date="{d:%Y-%m-%d}"><Rate currency="EUR">{5.0 + i * 0.01:.4f}</Rate>'
                        f'<Rate currency="USD">{4.4 + i * 0.01:.4f}</Rate><Rate currency="HUF" multiplier="100">1.3{i}</Rate></Cube>'
                        for i, d in enumerate(days))
        return ('<?xml version="1.0" encoding="utf-8"?><DataSet xmlns="http://www.bnr.ro/xsd"><Body>'
                f'<OrigCurrency>RON</OrigCurrency>{cubes}</Body></DataSet>').encode()

    def fake_requests_get(url, *a, **k):
        if "bnr.ro" in url:                     # cursurile BNR: anul trecut indisponibil (ramura de eroare)
            if f"nbrfxrates{TODAY.year - 1}" in url:
                raise requests.ConnectionError("BNR anul trecut indisponibil în testul de fum")
            days = ([pd.Timestamp(TODAY) - pd.Timedelta(days=d) for d in (3, 2, 1)] if "10days" in url
                    else [pd.Timestamp(TODAY) - pd.DateOffset(months=2), pd.Timestamp(TODAY) - pd.Timedelta(days=4)])
            return types.SimpleNamespace(status_code=200, content=_bnr_xml(sorted(days)), raise_for_status=lambda: None)
        if "api.stlouisfed.org" in url:          # calendarul FRED (releases/dates): 2 publicări urmărite + una neurmărită
            assert k.get("params", {}).get("api_key") == FAKE_FRED_KEY
            d0 = pd.Timestamp(TODAY)
            rel = [{"release_id": 50, "release_name": "Employment Situation", "date": f"{d0 + pd.Timedelta(days=1):%Y-%m-%d}"},
                   {"release_id": 10, "release_name": "Consumer Price Index", "date": f"{d0 + pd.Timedelta(days=7):%Y-%m-%d}"},
                   {"release_id": 18, "release_name": "H.15 Selected Interest Rates", "date": f"{d0:%Y-%m-%d}"}]
            return types.SimpleNamespace(status_code=200, json=lambda: {"release_dates": rel}, raise_for_status=lambda: None)
        if "ec.europa.eu/eurostat" in url:      # ROBOR 3M (Eurostat irt_st_m)
            return types.SimpleNamespace(status_code=200, text="", json=lambda: EUROSTAT_ROBOR, raise_for_status=lambda: None)
        if "data-api.ecb.europa.eu" not in url:
            raise requests.ConnectionError("fără rețea în testul de fum")
        ok = lambda: None
        months = pd.period_range(end=pd.Period(TODAY, "M") - 1, periods=15, freq="M")
        if "/FM/" in url:                       # dobânda de depozit BCE, zilnic
            days = pd.bdate_range(end=TODAY, periods=300)
            return types.SimpleNamespace(status_code=200, raise_for_status=ok, text="KEY,TIME_PERIOD,OBS_VALUE\n" + "".join(
                f"X,{d:%Y-%m-%d},{2.0 if i > 150 else 2.25}\n" for i, d in enumerate(days)))
        if "/IRS/M.DE" in url or "/ICP/" in url:  # Bund 10 ani / inflația HICP, lunar
            base = 2.6 if "/IRS/" in url else (5.0 if "M.RO" in url else 2.1)
            return types.SimpleNamespace(status_code=200, raise_for_status=ok, text="KEY,TIME_PERIOD,OBS_VALUE\n" + "".join(
                f"X,{m_},{base + 0.01 * i:.2f}\n" for i, m_ in enumerate(months)))
        if "/IRS/M.RO" in url and "lastNObservations" in str(k.get("params", {})) and k["params"]["lastNObservations"] >= 30:
            return types.SimpleNamespace(status_code=200, raise_for_status=ok, text="KEY,TIME_PERIOD,OBS_VALUE\n" + "".join(
                f"X,{m_},{7.0 + 0.02 * i:.2f}\n" for i, m_ in enumerate(months)))
        if "/YC/" in url:                       # curba AAA a zonei euro, 10 ani − 3 luni, zilnic
            return types.SimpleNamespace(status_code=200, text="KEY,TIME_PERIOD,OBS_VALUE\nX,2026-10-20,0.61\nX,2026-10-21,0.64\n")
        return types.SimpleNamespace(status_code=200, text="KEY,TIME_PERIOD,OBS_VALUE\nX,2026-07,7.25\nX,2026-08,7.12\n")

    requests.get = fake_requests_get

    # Modele grele înlocuite cu substitute (nu testăm aici calitatea predicțiilor)
    prophet = types.ModuleType("prophet")

    class Prophet:
        def __init__(self, **k):
            pass

        def add_regressor(self, name):
            pass

        def fit(self, df):
            self._df = df

        def make_future_dataframe(self, periods):
            last = self._df["ds"].iloc[-1]
            fut = pd.date_range(last + pd.Timedelta(days=1), periods=periods)
            return pd.DataFrame({"ds": list(self._df["ds"]) + list(fut)})

        def predict(self, future):
            y = float(self._df["y"].iloc[-1])
            return future.assign(yhat=y, yhat_upper=y * 1.1, yhat_lower=y * 0.9)

    prophet.Prophet = Prophet
    sys.modules["prophet"] = prophet

    transformers = types.ModuleType("transformers")
    transformers.pipeline = lambda *a, **k: (lambda text: [{"label": "positive", "score": 0.8}])
    sys.modules["transformers"] = transformers


# --------------------------------------------------------------------------
# Rularea
# --------------------------------------------------------------------------
FAKE_FRED_KEY = "0" * 32

EXPECTED_MESSAGES = (
    "(prag absolut 45%, netestat)", "(prag absolut 20%, netestat)",   # nivelul IV la bani pe lanțul sintetic
    "(netestat):** call-urile", "DIVERGENȚĂ (netestat)", "PESIMISM ÎN OPȚIUNI (netestat)", "NEUTRU (netestat)",  # verdict put/call
    "Fără opțiuni listate în Yahoo", "Yahoo nu a trimis lista de expirări", "Nicio expirare cu 20–45", "Greeks N/A",   # simboluri fără opțiuni sau fără cotații
    "Simbol invalid sau date indisponibile",      # simbolul INVALID
    "Foaia BVB nu mai are rândurile așteptate",    # foaia BVB simulată are doar 3 rânduri de indicatori
    "Fără preț disponibil pentru",                 # NODATA.RO în portofoliu
    "Marja de siguranță indisponibilă",            # DCF neaplicabil (sector financiar, FCF negativ, fără situații)
    "DCF: Valoarea terminală reprezintă",          # avertisment informativ: pondere mare a valorii terminale
    "DCF: Capex-ul consumă", "DCF: FCF-ul curent este",   # avertisment informativ: FCF deformat
    "DCF: Diferența dintre rata de scont",         # avertisment informativ: beta mic pe datele sintetice
    "SUPRAEVALUARE CRITICĂ",                       # verdict informativ: DCF sub preț pe datele sintetice
    "Datele despre acționari sunt momentan",       # fără date de acționariat
    "ACTIVITATE INSTITUȚIONALĂ EXTREMĂ",           # avertisment informativ de volum
    "STRATEGIE SHORT VOL", "ALERTA IV",            # avertismente informative de opțiuni
    "Concentrare mare",                            # avertisment informativ de portofoliu
    "Risc Ridicat",                                # avertisment informativ Sortino
    "Sortino sub 1", "Sortino între 1 și 2",       # avertisment informativ Sortino (TWR pe monedă)
    "Istoricul real al pozițiilor în",            # tab fără registru reconstruibil în datele de test
    "Prea puține zile cu randament în fereastra", # fereastră scurtă în datele de test
    "Tranziție", "Frică Extremă", "MOD DEFENSIV",   # texte informative macro
    "AI-ul nu a găsit o soluție convergentă",      # optimizator pe date sintetice
    # titluri de secțiune afișate cu st.error / st.warning / st.success (nu sunt erori)
    "Vulnerabilități (Potential Risks)", "PUNCTE SLABE", "OPORTUNITĂȚI", "Cea mai slabă lună",
    "Companii Small-Cap", "AMENINȚĂRI",
    "Yahoo nu a trimis rezumatul companiei",       # banner: indicatori calculați din situațiile financiare
    "Yahoo nu a trimis datele fundamentale",       # banner pentru simbolurile fără fundamentale
    "zile comune de tranzacționare; sunt necesare",  # tabul EUR din datele de test: o poziție listată de 60 de zile
    "Sunt necesare cel puțin 2 active cu istoric",  # tabul RON din datele de test are un singur simbol cu preț
)


def _ro_num(text):
    """„+1,39”, „7,28%”, „1.250,5000” → float; „N/A” → None (tabelele afișate ca text în format românesc)."""
    if text == "N/A":
        return None
    return float(str(text).replace("%", "").replace("+", "").replace(".", "").replace(",", "."))


def main():
    install_fakes()
    from streamlit.testing.v1 import AppTest

    problems, notes = [], []

    def check(at, label):
        for exc in at.exception:
            problems.append(f"[{label}] EXCEPȚIE: {exc.message}\n{''.join(exc.stack_trace or [])[-1200:]}")
        for kind in ("error", "warning"):
            for el in getattr(at, kind):
                text = str(el.value)
                bucket = notes if any(m in text for m in EXPECTED_MESSAGES) else problems
                bucket.append(f"[{label}] st.{kind}: {text[:200]}")
        # HTML în st.markdown: o linie goală urmată de o linie indentată cu 4+ spații transformă
        # restul blocului în cod (Daniel a văzut `<li style=...>` afișat ca text, 13.10.2026).
        for el in at.markdown:
            text = str(el.value)
            if "<li" in text and re.search(r"\n[ \t]*\n {4,}<", text):
                problems.append(f"[{label}] HTML afișat ca bloc de cod: {text[:120]!r}")

    at = AppTest.from_file(os.path.join(APP_DIR, "app.py"), default_timeout=300)
    at.secrets["gcp_service_account"] = {"type": "service_account"}
    at.secrets["FRED_API_KEY"] = FAKE_FRED_KEY   # cheie falsă: calendarul macro rulează pe răspunsul fals de mai sus
    at.run()
    check(at, "1. Știri")

    radio = lambda: at.sidebar.radio[0]

    radio().set_value("2. Analiză Companie").run()
    check(at, "2. Companie AAPL (date complete)")
    has_global = lambda: any("Scor fundamental (descriptiv" in str(h.value) for h in at.subheader)
    assert has_global(), "lipsește secțiunea Scor fundamental la AAPL"
    comp_df = next((d.value for d in at.dataframe if "Componentă" in getattr(d.value, "columns", [])), None)
    assert comp_df is not None, "lipsește tabelul cu componentele scorului fundamental la AAPL"
    by_comp = dict(zip(comp_df["Componentă"], comp_df["Notă"]))
    assert "E/P față de AAA" in by_comp, f"lipsește componenta E/P față de AAA: {list(by_comp)}"
    assert by_comp.get("GN (Graham)") == "informativ", f"GN trebuie să fie informativ: {by_comp.get('GN (Graham)')}"
    print(f"AAPL componente: {by_comp}")
    page_text = lambda: " ".join(str(e.value) for e in list(at.markdown) + list(at.caption) + list(at.info) + list(at.success) + list(at.error) + list(at.warning))
    curve_expect = {"AAPL": "SUA, titluri de stat 10 ani 4,20% − 3 luni 3,90%", "SAP.DE": "zona euro, curba titlurilor de stat AAA",
                    "SNP.RO": "România, titluri de stat 10 ani 7,12% (BCE) − ROBOR 3M 5,87% (Eurostat), 2026-08"}
    assert curve_expect["AAPL"] in page_text(), "lipsește curba SUA în ratingul AAPL"
    metric_labels = [str(m.label) for m in at.metric]
    assert "Delta call / put" in metric_labels, f"lipsesc greeks la AAPL: {metric_labels}"
    rank_metric = next((m for m in at.metric if str(m.label) == "IV Rank"), None)
    assert rank_metric is not None and str(rank_metric.value) != "N/A", f"IV Rank lipsește la AAPL: {rank_metric and rank_metric.value}"
    assert "25 zile de istoric" in str(rank_metric.delta), f"numărul de zile din IV_istoric: {rank_metric.delta}"
    raw_bold = [str(m.value)[-120:] for m in at.markdown if "<div" in str(m.value) and "(netestat):" in str(m.value) and "**" in str(m.value)]
    assert not raw_bold, f"casete HTML cu ** neinterpretat: {raw_bold}"
    assert "expirarea **" in page_text() and "zile; aleasă din 20–45" in page_text(), "lipsește expirarea 20–45 de zile la AAPL"

    def master_matches_top(sym):
        """Sinteza „Decizie Master AI”: pasul 1 are exact eticheta și scorul din secțiunea de scor de sus;
        metoda veche e doar în arhivă (expander)."""
        texts = [str(m.value) for m in at.markdown]
        master = [t for t in texts if "Pasul 1 · Fundamental" in t and "Pasul 3 · Context" in t]
        assert master, f"lipsește sinteza Master AI în trei pași la {sym}"
        m = re.search(r"Pasul 1 · Fundamental</span><br>\s*<span[^>]*>([^<]+)</span>\s*<span[^>]*>&nbsp;([^<]+)</span>", master[0])
        assert m, f"pasul 1 necitibil la {sym}: {master[0][:200]!r}"
        label, detail = m.group(1), m.group(2)
        top = [t for t in texts if "Pasul 3 · Context" not in t and label in t and detail in t]
        assert top, f"pasul 1 din Master AI ({label} {detail}) diferă de secțiunea de scor la {sym}"
        assert "Pasul 3 · Context" in master[0] and "fără puncte" in master[0], f"lipsește pasul 3 la {sym}"
        assert any("Metoda veche" in str(e.label) for e in at.expander), f"lipsește arhiva metodei vechi la {sym}"
        print(f"{sym} Master AI pasul 1: {label} {detail}")

    alpha_aapl = next((m for m in at.metric if m.label == "Alpha (1Y)"), None)
    alpha_help = alpha_aapl.proto.help if alpha_aapl is not None else ""
    assert "SPY" in alpha_help and "rată fără risc" in alpha_help, f"alpha AAPL: {alpha_help!r}"
    print(f"AAPL alpha: {alpha_help}")
    master_matches_top("AAPL")
    for sym in ("TLV.RO", "SNP.RO", "SAP.DE", "EURO.PA", "LOSS", "NEWCO", "EMPTY", "INVALID"):
        at.sidebar.text_input[0].set_value(sym).run()
        check(at, f"2. Companie {sym}")
        if sym in ("SAP.DE", "EURO.PA", "LOSS", "NEWCO"):      # LOSS = emitent financiar: P/BV față de ROE
            assert has_global(), f"lipsește secțiunea Scor fundamental la {sym}"
        if sym in ("TLV.RO", "SNP.RO"):   # dividendul BVB din fișierul companiei, nu din Yahoo
            div_m = next((m for m in at.metric if m.label == "Dividend (Randament)"), None)
            print(f"{sym} dividend: {div_m and div_m.value} | {div_m and div_m.proto.help[-160:]}")
            if sym == "SNP.RO":
                assert div_m is not None and "Analiză companii" in div_m.proto.help and div_m.value != "N/A", \
                    f"dividend SNP.RO: {div_m and div_m.value}"
        if sym == "EURO.PA":     # alpha UE: față de Euro Stoxx 50, nu față de SPY
            alpha_m = next((m for m in at.metric if m.label == "Alpha (1Y)"), None)
            alpha_help = alpha_m.proto.help if alpha_m is not None else ""
            assert "Euro Stoxx 50" in alpha_help and "SPY" not in alpha_help, f"alpha EURO.PA: {alpha_help!r}"
            print(f"EURO.PA alpha: {alpha_help}")
        if sym == "EURO.PA":     # EUR: AAA = Bund + marja AAA − GS10 din SUA, pe aceeași lună
            df_c = next((d.value for d in at.dataframe if "Componentă" in getattr(d.value, "columns", [])), None)
            ep_txt = dict(zip(df_c["Componentă"], df_c["Explicație"])).get("E/P față de AAA", "") if df_c is not None else ""
            assert "FRED AAA − GS10" in ep_txt, f"proxy AAA EUR fără marja pe aceeași lună: {ep_txt!r}"
            print(f"EURO.PA E/P: {ep_txt}")
        if sym in ("TLV.RO", "SNP.RO"):
            assert not has_global(), f"secțiunea Scor fundamental nu trebuie să apară la {sym}"
        if sym in curve_expect:
            assert curve_expect[sym] in page_text(), f"lipsește curba pieței în ratingul {sym}"
        if sym in ("SAP.DE", "SNP.RO"):
            master_matches_top(sym)
        if sym == "SNP.RO":
            assert any("Decizie BVB" in str(h.value) for h in at.subheader), "lipsește secțiunea Decizie BVB"
            at.button(key="bvb_files_reload").click().run()
            check(at, "2. Companie SNP.RO după recitirea fișierelor")
    at.sidebar.text_input[0].set_value("AAPL").run()
    for opt in ("1 Lună", "5 Ani"):
        at.selectbox[0].set_value(opt).run()
        check(at, f"2. Companie AAPL interval {opt}")

    for section in ("3. Portofoliu", "4. Piață Globală", "5. Import Date", "6. Rezumatul Zilei",
                    "7. Scanner Volum", "8. Watchlist"):
        radio().set_value(section).run()
        check(at, section)
        if section == "6. Rezumatul Zilei":
            # data ședinței, BET proxy, VIX/VIX3M, lățimea pe o zi; scorurile vechi (Fear & Greed, Pulsul BVB) au ieșit
            mets = {m.label: str(m.value) for m in at.metric}
            print("Rezumat:", {k: mets[k] for k in mets if k.startswith(("BET", "VIX", "Lățime", "RSP", "S&P"))})
            for lbl in ("BET (proxy: ETF TVBETETF)", "Mișcarea BET în σ", "VIX / VIX3M", "RSP − SPY (o zi)", "S&P 500"):
                assert lbl in mets and mets[lbl] != "N/A", (lbl, mets.get(lbl))
            assert any("Ultima ședință:" in str(e.value) for e in at.markdown), "lipsește data ultimei ședințe"
            sig_tabs = [getattr(d.value, "data", d.value) for d in at.dataframe if "σ" in getattr(getattr(d.value, "data", d.value), "columns", [])]
            assert len(sig_tabs) >= 3, "lipsesc topurile cu σ"
            all_text = " ".join(str(e.value) for e in list(at.markdown) + list(at.info) + list(at.caption))
            assert "Sentimentul Wall Street" not in all_text and "Pulsul Pieței" not in all_text, "scorurile vechi încă afișate"
            # etapa B, „Portofoliul meu azi”, totul în lei: acțiuni (TLV.RO, SAP.DE, AAPL) + titlul de stat R9901AE + numerar
            for lbl in ("Valoare totală azi (lei)", "Acțiuni și ETF", "Titluri de stat", "Numerar"):
                assert lbl in mets and mets[lbl] != "N/A", (lbl, mets.get(lbl))
            pday = next((getattr(d.value, "data", d.value) for d in at.dataframe if "Contribuție" in
                         getattr(getattr(d.value, "data", d.value), "columns", [])), None)
            cash_t = next((getattr(d.value, "data", d.value) for d in at.dataframe if "Numerar" in
                           getattr(getattr(d.value, "data", d.value), "columns", [])), None)
            assert pday is not None and cash_t is not None, "lipsește tabelul „Portofoliul meu azi” sau numerarul"
            print("Portofoliul azi:", {k: mets[k] for k in ("Valoare totală azi (lei)", "Acțiuni și ETF", "Titluri de stat", "Numerar")},
                  pday[["Simbol", "Piață", "P/L (lei)", "Contribuție", "Valoare (lei)", "Stare"]].values.tolist(),
                  cash_t.values.tolist())
            assert set(pday["Simbol"]) == {"TLV", "SAP.DE", "AAPL", "R9901AE"}, list(pday["Simbol"])
            assert "P/L (monedă)" not in pday.columns, "sumele din Rezumat trebuie să fie doar în lei"
            assert not pday.isna().any().any() and not cash_t.isna().any().any(), "celule nule (se afișează „None”)"
            lei = lambda t: _ro_num(t.replace(" lei", ""))
            eq_rows, bond_rows = pday[pday["Piață"] != "Titluri de stat"], pday[pday["Piață"] == "Titluri de stat"]
            assert abs(sum(lei(v) for v in eq_rows["Valoare (lei)"]) - lei(mets["Acțiuni și ETF"])) <= 0.5 * (len(eq_rows) + 1)
            assert abs(sum(lei(v) for v in bond_rows["Valoare (lei)"]) - lei(mets["Titluri de stat"])) <= 0.5 * (len(bond_rows) + 1)
            assert abs(sum(lei(v) for v in cash_t["Valoare (lei)"]) - lei(mets["Numerar"])) <= 0.5 * (len(cash_t) + 1)
            parts = sum(lei(mets[k]) for k in ("Acțiuni și ETF", "Titluri de stat", "Numerar"))
            assert abs(parts - lei(mets["Valoare totală azi (lei)"])) <= 1.5, (parts, mets["Valoare totală azi (lei)"])
            ok_rows = pday[pday["Stare"] == ""]
            if len(ok_rows):
                tot_pp = sum(_ro_num(v.replace(" pp", "")) for v in ok_rows["Contribuție"])
                eq_delta = next(m.delta for m in at.metric if m.label == "Acțiuni și ETF")      # „-276 lei · -0,83%”
                assert abs(tot_pp - _ro_num(eq_delta.split(" · ")[1])) <= 0.011 * len(ok_rows), (tot_pp, eq_delta)
            # Wall Street: valoarea tranzacționată în lei (cursul BNR), nu în USD
            assert mets.get("Valoare tranzacționată (universul urmărit)", "").endswith("lei"), mets.get("Valoare tranzacționată (universul urmărit)")
        if section == "5. Import Date":
            # foile BVB și GLOBAL ca atare: lipsă = N/A (nu 0), media și mediana BVB, moneda din celulă
            frames = [getattr(d.value, "data", d.value) for d in at.dataframe]
            bvb_t = next((f for f in frames if "P/E TTM" in getattr(f, "columns", [])), None)
            assert bvb_t is not None and list(bvb_t.index)[:2] == ["TLV", "SNP"], "lipsește tabelul BVB transpus"
            assert pd.isna(bvb_t.loc["SNP", "Rentabilitate capital (ROE)"]), "#DIV/0! din BVB a devenit număr"
            glob_t = next((f for f in frames if "Capitalizare (mld.)" in getattr(f, "columns", [])), None)
            assert glob_t is not None and list(glob_t["Companii"]) == ["AAPL", "SAP"], "lipsește tabelul GLOBAL"
            assert list(glob_t["Industrie"]) == ["Tech", "Tech"] and pd.isna(glob_t.at[1, "P/E"]), glob_t
            print("Import Date:", bvb_t.shape, glob_t.shape)
        if section == "3. Portofoliu":
            pos_df = next((d.value for d in at.dataframe if {"Simbol", "Cantitate", "Sursă preț"} <= set(getattr(d.value, "columns", []))), None)
            assert pos_df is not None, "lipsește tabelul pozițiilor deschise"
            qty = dict(zip(pos_df["Simbol"], pos_df["Cantitate"]))
            # registrul fals are 100 + 10 − 30 = 80 TLV.RO; foaia Portofoliu are 367 (doar comparată)
            assert qty.get("TLV.RO") == "80", f"pozițiile nu vin din registrul Tranzactii: {qty}"
            # titlul de stat nu e în foaie: rămâne din registru
            assert qty.get("R9901AE") == "3", f"titlul de stat lipsește din poziții: {qty}"
            # taburile pe monedă folosesc tot registrul: tabul RON are poziții (nu mesajul de portofoliu gol)
            assert not any("Nu există acțiuni sau ETF-uri deschise" in str(e.value) for e in at.info), "taburile nu primesc pozițiile din registru"
            # Diagnosticul pe monedă vine din TWR-ul registrului: Sharpe are rata fără risc a monedei în „?”
            sharpe_tabs = [m for m in at.metric if m.label == "Sharpe Ratio"]
            print("Sharpe pe taburi:", [str(m.value) for m in sharpe_tabs])
            assert sharpe_tabs and all("Rata fără risc" in m.proto.help for m in sharpe_tabs), "Sharpe fără rata monedei"
            assert any(str(m.value) != "N/A" for m in sharpe_tabs), "Sharpe N/A pe toate taburile"
            # „Detaliu Poziții” din taburi: coloanele în română, cu pondere; ponderile unui tab dau 100%
            detail_dfs = [d.value for d in at.dataframe
                          if {"Cost mediu", "Pondere %"} <= set(getattr(d.value, "columns", []))]
            assert detail_dfs, "lipsește tabelul Detaliu Poziții cu coloanele în română"
            for dd in detail_dfs:
                w = dd.data["Pondere %"] if hasattr(dd, "data") else dd["Pondere %"]
                assert abs(float(w.sum()) - 100.0) < 1e-6 or w.isna().all(), f"ponderile nu dau 100%: {list(w)}"
            # sectoarele: la BVB din harta proprie (Yahoo nu trimite sectorul), cu verdictul marcat „netestat”
            sec_caps = " ".join(str(c.value) for c in at.caption)
            assert "sector din harta proprie BVB: TLV.RO" in sec_caps, "TLV.RO nu primește sectorul din harta BVB"
            assert any("netestat" in str(e.value) for e in list(at.success) + list(at.warning) if "sector" in str(e.value)), \
                "verdictul pe sectoare nu e marcat netestat"
            print("Detaliu Poziții:",[list(dd.data["Simbol"]) if hasattr(dd, "data") else list(dd["Simbol"]) for dd in detail_dfs])
            for rng in ("1Z", "1L", "5A"):
                at.select_slider[0].set_value(rng).run()
                check(at, f"3. Portofoliu interval {rng}")
            at.button(key="manual_prices_reload").click().run()
            check(at, "3. Portofoliu după reîncărcarea prețurilor manuale")
            at.toggle(key="ledger_curve").set_value(True).run()
            check(at, "3. Portofoliu cu curba reală din registru")
        if section == "4. Piață Globală":
            # verdictul macro pe piloni: 6 piloni, fiecare cu pondere, scor (sau N/A) și bază; ponderile dau 100
            pil = next((d.value for d in at.dataframe if {"Pilon", "Pondere %", "Scor", "Bază"} <= set(
                getattr(getattr(d.value, "data", d.value), "columns", []))), None)
            assert pil is not None, "lipsește tabelul pilonilor verdictului macro"
            pil_df = getattr(pil, "data", pil)
            assert len(pil_df) == 6 and float(pil_df["Pondere %"].sum()) == 100.0, pil_df
            print("Piloni macro:", list(zip(pil_df["Pilon"], pil_df["Scor"])))
            assert not any("VIX 20,0" in str(e.value) for e in at.markdown), "VIX de rezervă 20 încă afișat"
            # calendarul macro FRED: pe răspunsul fals, doar cele 2 publicări urmărite (H.15 nu e urmărită), ordonate
            cal = next((getattr(d.value, "data", d.value) for d in at.dataframe if {"Publicare", "Peste"} <= set(
                getattr(getattr(d.value, "data", d.value), "columns", []))), None)
            assert cal is not None, "lipsește calendarul macro"
            # deciziile FOMC vin din calendarul Fed (data reală de azi): se scot înainte de comparația cu răspunsul fals
            cal_fomc = cal[cal["Publicare"] == fred_cal.FOMC_LABEL]
            assert len(cal_fomc) == len(fred_cal.fomc_in_window(TODAY.date())), cal
            cal = cal[cal["Publicare"] != fred_cal.FOMC_LABEL]
            assert list(cal["Publicare"]) == ["Piața muncii (NFP, șomaj)", "Inflația CPI"] and list(cal["Peste"]) == ["mâine", "7 zile"], cal
            print("Calendar macro:", cal[["Data", "Publicare"]].values.tolist())
            # piețele în care investește: regim tehnic + randament în lei (curs BNR)
            reg = next((getattr(d.value, "data", d.value) for d in at.dataframe if {"Trend", "Scor moment"} <= set(
                getattr(getattr(d.value, "data", d.value), "columns", []))), None)
            assert reg is not None and len(reg) == 6, "lipsește tabelul regimului piețelor"
            rr = next((getattr(d.value, "data", d.value) for d in at.dataframe if {"Randament local", "Randament în lei"} <= set(
                getattr(getattr(d.value, "data", d.value), "columns", []))), None)
            assert rr is not None, "lipsește tabelul randamentului în lei"
            print("Regim:", reg[["Piață", "Trend", "Scor moment"]].values.tolist())
            print("Randament în lei:", rr[["Piață", "Randament local", "Efect curs față de leu", "Randament în lei"]].values.tolist())
            bet = rr.set_index("Piață").loc["BET (prin ETF TVBETETF)"]
            assert bet["Efect curs față de leu"] == 0.0 or pd.isna(bet["Randament local"]), "BET are efect valutar"
            for hz in ("1 lună", "De la 1 ian."):
                at.radio(key="mk_horizon").set_value(hz).run()
                check(at, f"4. Piață Globală orizont {hz}")
                if hz == "1 lună":
                    rr1 = next(getattr(d.value, "data", d.value) for d in at.dataframe if "Randament în lei" in
                               getattr(getattr(d.value, "data", d.value), "columns", [])).set_index("Piață")
                    print("Randament 1 lună:", rr1[["Randament local", "Efect curs față de leu", "Randament în lei"]].to_dict("index"))
                    sp = rr1.loc["S&P 500"]
                    assert abs((1 + sp["Randament local"]) * (1 + sp["Efect curs față de leu"]) - 1 - sp["Randament în lei"]) < 1e-12
            # dobânzi și risc de țară: 17 rânduri; lipsa unei serii = rând N/A marcat, nu valoare presupusă
            rt = next((getattr(d.value, "data", d.value) for d in at.dataframe if {"Indicator", "Δ 3 luni (pp)", "Sursă"} <= set(
                getattr(getattr(d.value, "data", d.value), "columns", []))), None)
            assert rt is not None and len(rt) == 17, "lipsește tabelul dobânzilor"
            print("Dobânzi:", rt[["Indicator", "Valoare (%)", "Perioadă", "Sursă"]].values.tolist())
            for _, r_ in rt.iterrows():
                assert pd.isna(r_["Valoare (%)"]) == r_["Sursă"].endswith("indisponibil"), r_.to_dict()
            # lățimea pieței SUA: 2 perechi, diferența = randament A − randament B
            br = next((getattr(d.value, "data", d.value) for d in at.dataframe if "Diferență 12 luni" in
                       getattr(getattr(d.value, "data", d.value), "columns", [])), None)
            assert br is not None and len(br) == 2, "lipsește tabelul lățimii pieței"
            print("Lățime:", br[["Pereche", "Diferență 3 luni", "Detaliu 3 luni"]].values.tolist())
            # prima de risc a acțiunilor: BVB (mediana E/P din foaie − RO 10 ani) și S&P 500 (1/P/E SPY − DGS10)
            erp = next((getattr(d.value, "data", d.value) for d in at.dataframe if "Prima de risc (pp)" in
                        getattr(getattr(d.value, "data", d.value), "columns", [])), None)
            assert erp is not None and list(erp["Piață"]) == ["BVB", "S&P 500"], "lipsește tabelul primei de risc"
            print("Prima de risc:", erp[["Piață", "Randamentul profiturilor (E/P)", "Titluri de stat 10 ani", "Prima de risc (pp)"]].values.tolist())
            # tabelul e text gata formatat (2 zecimale, lipsă = „N/A”, niciodată „None”): toleranță de rotunjire
            assert not erp.isna().any().any(), "celule nule în prima de risc (se afișează „None”)"
            for _, e_ in erp.iterrows():
                ep_, bd_, gp_ = (_ro_num(e_[c]) for c in ("Randamentul profiturilor (E/P)", "Titluri de stat 10 ani", "Prima de risc (pp)"))
                if gp_ is not None:
                    assert abs(ep_ - bd_ - gp_) <= 0.011, e_.to_dict()
            # indicatori macro: interpretarea dobânzilor pe o lună în puncte de bază; ghidul spune că e pe o ședință, netestat
            infos = " ".join(str(e.value) for e in at.info)
            assert ("pb într-o lună" in infos) or ("nu are istoric suficient" in infos), "lipsește interpretarea dobânzilor"
            assert "Semnale pe variația unei singure ședințe" in infos, "ghidul macro nu e marcat pe o ședință / netestat"
            # rotația Nasdaq/Dow pe o lună calendaristică; barometrul refolosește sentimentul de sus (nu rulează FinBERT a doua oară)
            assert any("Trend Rotație (1 lună)" in str(e.value) for e in at.markdown), "rotația nu e pe o lună"
            assert not any("Barometrul de sentiment se actualizează" in str(e.value) for e in at.info), "barometrul a căzut pe excepție"
            # harta sectoarelor: implicit pe o lună; și pe „Azi”
            assert any("Liderii lunii" in str(e.value) for e in at.markdown), "harta sectoarelor nu e pe o lună implicit"
            at.radio(key="sec_heat_hz").set_value("Azi").run()
            check(at, "4. Piață Globală harta sectoarelor azi")
            assert any("Liderii zilei" in str(e.value) for e in at.markdown), "harta sectoarelor pe azi"
            at.radio(key="sec_heat_hz").set_value("1 lună").run()
            # sinteza AI: fără zerouri inventate, nota „netestate” la final
            md_all = " ".join(str(e.value) for e in at.markdown)
            assert "Reguli orientative, netestate; mărfurile și dolarul pe o lună" in md_all, "sinteza macro fără nota de bază"
            # stresul pieței: VIX/VIX3M și spread-urile de credit, cu valori (nu N/A) pe datele false
            stress_m = {m.label: str(m.value) for m in at.metric if m.label in ("VIX / VIX3M", "Spread high yield SUA", "Spread investment grade SUA")}
            print("Stres:", stress_m)
            assert len(stress_m) == 3 and all(v != "N/A" for v in stress_m.values()), stress_m
            # cursul valutar față de leu: BNR + live; anul trecut lipsește → 1 an / de la 1 ian. N/A, cu nota
            fx = next((d.value for d in at.dataframe if {"Curs BNR", "Live (Yahoo)", "1 an %"} <= set(
                getattr(getattr(d.value, "data", d.value), "columns", []))), None)
            assert fx is not None, "lipsește tabelul cursului valutar"
            fx_df = getattr(fx, "data", fx).set_index("Valută")
            print("Curs BNR:", fx_df[["Curs BNR", "Zi %", "1 lună %", "1 an %"]].head(3).to_dict("index"))
            assert not fx_df.isna().any().any(), "celule nule în cursul valutar (se afișează „None”)"
            assert abs(_ro_num(fx_df.loc["Euro", "Curs BNR"]) - 5.02) < 1e-9, fx_df.loc["Euro"]
            assert fx_df.loc["Euro", "1 an %"] == "N/A", "1 an fără istoric trebuie N/A"
            assert fx_df.loc["Liră sterlină", "Curs BNR"] == "N/A", "valută absentă la BNR trebuie N/A"
            assert any(f"nbrfxrates{TODAY.year - 1}.xml" in str(c.value) for c in at.caption), "lipsește nota fișierului BNR necitit"
        if section == "7. Scanner Volum":
            at.button[0].click().run()
            check(at, "7. Scanner Volum (după scanare)")

    print(f"Director testat: {APP_DIR}")
    print(f"Mesaje așteptate (informative): {len(notes)}")
    if problems:
        print(f"\nPROBLEME: {len(problems)}\n")
        for p in problems:
            print(p, "\n")
        sys.exit(1)
    print("OK: toate secțiunile au rulat fără excepții și fără erori neașteptate.")


if __name__ == "__main__":
    main()
