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
import sys
import time
import types

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else ROOT
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

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
    return {"1d": 1, "5d": 5, "1mo": 22, "3mo": 64, "6mo": 127, "1y": 252, "3y": 756, "5y": 1260}.get(period, 252)


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
    return dict(RICH_INFO)


class _FastInfo:
    def __init__(self, sym):
        df = _ohlcv(sym, 5)
        self.last_price = float(df["Close"].iloc[-1]) if df["Close"].notna().any() else None
        self.previous_close = float(df["Close"].iloc[-2]) if df["Close"].notna().any() else None
        self.currency = "RON" if sym.endswith(".RO") else ("EUR" if sym.endswith(".DE") else "USD")
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
                             [1.55e10] * n, [3.5e9] * n, [1.8e10] * n, [1.15e11] * n],
                            index=["Total Revenue", "Net Income", "Diluted Average Shares",
                                   "Interest Expense", "Tax Provision", "Pretax Income"], columns=cols)

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
                "Ordinary Shares Number": (1.5e10, False)}

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
                                 "impliedVolatility": rng.uniform(0.15, 0.6, len(strikes))})
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


def fake_fred(code, source, start, end):
    idx = pd.date_range(end=TODAY - pd.offsets.MonthBegin(2), periods=16, freq="MS")
    base = {"CPIAUCSL": 320.0, "CPILFESL": 325.0, "PCEPILFE": 125.0, "UNRATE": 4.2, "FEDFUNDS": 4.3,
            "PAYEMS": 159000.0, "ADPCHGS": 134000.0, "JTSJOL": 7400.0, "RSAFS": 720000.0,
            "INDPRO": 103.0, "HOUST": 1350.0, "UMCSENT": 62.0, "IRLTLT01DEM156N": 3.1}[code]
    step = 0.0025 if code not in ("UNRATE", "FEDFUNDS", "IRLTLT01DEM156N") else 0.0
    return pd.DataFrame({code: [base * (1 + step) ** i for i in range(len(idx))]}, index=idx)


def fake_feed(url):
    def entry(i):
        ts = time.gmtime(time.time() - i * 3600)
        return types.SimpleNamespace(
            title=f"Știre <b>test</b> {i}: banca centrală și tehnologia AI", link="https://example.com/a",
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
    "GLOBAL": [["Industrie", "Companii", "Capitalizare", "Preț actiune", "P/E", "ROE", "Recomandare"],
               ["Tech", "AAPL", "2.500.000.000.000", "180,5", "28,1", "35,2%", "Buy"]],
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


class FakeClient:
    def open(self, name):
        return FakeSpreadsheet()


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

    def fake_requests_get(url, *a, **k):
        if "data-api.ecb.europa.eu" not in url:
            raise requests.ConnectionError("fără rețea în testul de fum")
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
EXPECTED_MESSAGES = (
    "Simbol invalid sau date indisponibile",      # simbolul INVALID
    "Fără preț disponibil pentru",                 # NODATA.RO în portofoliu
    "Marja de siguranță indisponibilă",            # DCF neaplicabil (sector financiar, FCF negativ, fără situații)
    "DCF: Valoarea terminală reprezintă",          # avertisment informativ: pondere mare a valorii terminale
    "DCF: Capex-ul consumă", "DCF: FCF-ul curent este",   # avertisment informativ: FCF deformat
    "SUPRAEVALUARE CRITICĂ",                       # verdict informativ: DCF sub preț pe datele sintetice
    "Datele despre acționari sunt momentan",       # fără date de acționariat
    "ACTIVITATE INSTITUȚIONALĂ EXTREMĂ",           # avertisment informativ de volum
    "STRATEGIE SHORT VOL", "ALERTA IV",            # avertismente informative de opțiuni
    "Concentrare mare",                            # avertisment informativ de portofoliu
    "Risc Ridicat",                                # avertisment informativ Sortino
    "Tranziție", "Frică Extremă", "MOD DEFENSIV",   # texte informative macro
    "AI-ul nu a găsit o soluție convergentă",      # optimizator pe date sintetice
    # titluri de secțiune afișate cu st.error / st.warning / st.success (nu sunt erori)
    "Vulnerabilități (Potential Risks)", "PUNCTE SLABE", "OPORTUNITĂȚI", "Cea mai slabă lună",
    "Companii Small-Cap", "AMENINȚĂRI",
    "Yahoo nu a trimis datele fundamentale",       # banner pentru simbolurile fără fundamentale
    "Sunt necesare cel puțin 2 active cu istoric",  # tabul RON din datele de test are un singur simbol cu preț
)


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

    at = AppTest.from_file(os.path.join(APP_DIR, "app.py"), default_timeout=300)
    at.secrets["gcp_service_account"] = {"type": "service_account"}
    at.run()
    check(at, "1. Știri")

    radio = lambda: at.sidebar.radio[0]

    radio().set_value("2. Analiză Companie").run()
    check(at, "2. Companie AAPL (date complete)")
    for sym in ("TLV.RO", "SAP.DE", "LOSS", "NEWCO", "EMPTY", "INVALID"):
        at.sidebar.text_input[0].set_value(sym).run()
        check(at, f"2. Companie {sym}")
    at.sidebar.text_input[0].set_value("AAPL").run()
    for opt in ("1 Lună", "5 Ani"):
        at.selectbox[0].set_value(opt).run()
        check(at, f"2. Companie AAPL interval {opt}")

    for section in ("3. Portofoliu", "4. Piață Globală", "5. Import Date", "6. Rezumatul Zilei",
                    "7. Scanner Volum", "8. Watchlist"):
        radio().set_value(section).run()
        check(at, section)
        if section == "3. Portofoliu":
            for rng in ("1Z", "1L", "5A"):
                at.select_slider[0].set_value(rng).run()
                check(at, f"3. Portofoliu interval {rng}")
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
