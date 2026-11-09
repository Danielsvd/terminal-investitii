"""Pagina Watchlist: ultimul preț cu data lui și citirea foii (valori sintetice)."""
import numpy as np
import pandas as pd

from analytics import watchlist as wl


def test_last_quotes_calendare_diferite_si_lipsa():
    idx = pd.to_datetime(["2026-11-04", "2026-11-05", "2026-11-06"])
    close = pd.DataFrame({
        "AAPL": [200.0, 201.5, 203.25],
        "TLV.RO": [30.1, 30.4, np.nan],      # BVB fără bara de vineri: ultimul preț e cel de joi
        "GOL": [np.nan, np.nan, np.nan],     # simbol fără date: lipsește, nu 0
        "ZERO": [10.0, 0.0, np.nan],         # 0 nu e preț: rămâne 10 de miercuri
    }, index=idx)
    q = wl.last_quotes(close)
    assert q["AAPL"] == {"price": 203.25, "date": pd.Timestamp("2026-11-06")}
    assert q["TLV.RO"] == {"price": 30.4, "date": pd.Timestamp("2026-11-05")}
    assert q["ZERO"] == {"price": 10.0, "date": pd.Timestamp("2026-11-04")}
    assert "GOL" not in q


def test_last_quotes_index_cu_fus_orar_si_gol():
    idx = pd.DatetimeIndex(["2026-11-06 00:00"], tz="America/New_York")
    q = wl.last_quotes(pd.DataFrame({"MSFT": [500.0]}, index=idx))
    assert q["MSFT"]["date"] == pd.Timestamp("2026-11-06")
    assert wl.last_quotes(pd.DataFrame()) == {}
    assert wl.last_quotes(None) == {}


def test_parse_watchlist_coloane_dupa_nume_si_valori():
    values = [
        ["Notes", "Symbol", "TargetPrice"],   # ordinea din foaie nu contează
        ["suport", " tsla ", 250],
        ["", "TLV.RO", "25,5"],               # text românesc
        ["fără țintă", "MSFT", ""],           # ținta goală: None, nu 0
        ["", "", 10],                         # rând fără simbol: ignorat
        ["scurt", "SAP.DE"],                  # rând mai scurt decât antetul
    ]
    df, problems = wl.parse_watchlist(values)
    assert problems == []
    assert df["Symbol"].tolist() == ["TSLA", "TLV.RO", "MSFT", "SAP.DE"]
    assert df["TargetPrice"].tolist()[:2] == [250.0, 25.5]
    assert pd.isna(df["TargetPrice"].iloc[2]) and pd.isna(df["TargetPrice"].iloc[3])   # lipsă, nu 0
    assert df["Notes"].tolist() == ["suport", "", "fără țintă", "scurt"]


def test_parse_watchlist_foaie_necitita_antet_gresit_goala():
    df, problems = wl.parse_watchlist(None)
    assert df.empty and problems == ["Foaia watchlist nu a putut fi citită."]
    df, problems = wl.parse_watchlist([["Simbol", "Tinta"], ["AAPL", 100]])
    assert df.empty and "Symbol, TargetPrice" in problems[0]
    df, problems = wl.parse_watchlist([["Symbol", "TargetPrice", "Notes"]])
    assert df.empty and problems == []
    df, problems = wl.parse_watchlist([])
    assert df.empty and problems == []
    # fără Notes: se citește, notele rămân goale
    df, problems = wl.parse_watchlist([["Symbol", "TargetPrice"], ["AAPL", 100]])
    assert problems == [] and df["Notes"].tolist() == [""]


def test_symbol_rows_doar_coloana_symbol_de_jos_in_sus():
    values = [
        ["Symbol", "TargetPrice", "Notes"],
        ["AAPL", 150, "TSLA"],        # nota conține exact „TSLA”: nu se șterge rândul ăsta
        ["TSLA", 250, ""],
        ["MSFT", 300, ""],
        [" tsla ", 240, "duplicat"],  # spații și litere mici: același simbol
    ]
    assert wl.symbol_rows(values, "TSLA") == [5, 3]
    assert wl.symbol_rows(values, "tsla ") == [5, 3]
    assert wl.symbol_rows(values, "MSFT") == [4]
    assert wl.symbol_rows(values, "NVDA") == []
    assert wl.symbol_rows(values, "") == []
    assert wl.symbol_rows([["Notes", "Symbol"], ["x", "AAPL"]], "AAPL") == [2]   # coloana după antet
    assert wl.symbol_rows([["Simbol"], ["AAPL"]], "AAPL") == []                   # fără coloana Symbol
    assert wl.symbol_rows([], "AAPL") == [] and wl.symbol_rows(None, "AAPL") == []


def test_normalize_symbol():
    assert wl.normalize_symbol(" tlv.ro ") == "TLV.RO"
    assert wl.normalize_symbol(None) == ""


def test_symbol_candidates_ca_in_analiza():
    assert wl.symbol_candidates(" tlv ") == ["TLV", "TLV.RO"]
    assert wl.symbol_candidates("TLV.RO") == ["TLV.RO"]
    assert wl.symbol_candidates("SAP.DE") == ["SAP.DE", "SAP.DE.RO"]   # ca get_stock_data; al doilea nu are preț
    assert wl.symbol_candidates("") == []


def test_target_zone_si_status_aceeasi_regula_ca_in_card():
    from data.helpers import entry_target_view, target_zone
    assert target_zone(100.0, 100.0) == "sub" and target_zone(95.0, 100.0) == "sub"
    assert target_zone(104.99, 100.0) == "aproape"       # +4,99% < 5%
    assert target_zone(105.0, 100.0) == "peste"           # exact +5%: nu mai e „aproape”, ca în card
    assert target_zone(None, 100.0) is None and target_zone(50.0, 0) is None
    # aceeași zonă → aceeași culoare în cardul din Analiză
    assert entry_target_view(104.99, 100.0)[2] == "#D29922" and entry_target_view(105.0, 100.0)[2] == "#8B949E"
    assert wl.status_label(95.0, 100.0) == "✅ La/sub țintă"
    assert wl.status_label(103.0, 100.0) == "🟡 Aproape (<5%)"
    assert wl.status_label(110.0, 100.0) == "⏳ Peste țintă"
    assert wl.status_label(None, 100.0) == "Fără preț"
    assert wl.status_label(50.0, None) == "Fără țintă"
    assert wl.status_label(None, None) == "Fără preț"


def test_price_date_text():
    today = pd.Timestamp("2026-11-09 14:00")                 # luni
    assert wl.price_date_text(pd.Timestamp("2026-11-06"), today) == "06.11.2026"              # vineri: 1 zi lucrătoare
    assert wl.price_date_text(pd.Timestamp("2026-11-09"), today, partial=True) == "09.11.2026 ⏳ ședință în curs"
    assert wl.price_date_text(pd.Timestamp("2026-11-04"), today) == "04.11.2026"              # 3 zile: încă fără ⚠️
    assert wl.price_date_text(pd.Timestamp("2026-11-03"), today) == "03.11.2026 ⚠️ veche (4 zile lucrătoare)"
    assert wl.price_date_text(None, today) == "N/A"


def test_coloana_date_citire_si_scriere_dupa_antet():
    values = [
        ["Symbol", "TargetPrice", "Notes", "Date"],
        ["AAPL", 150, "", "2026-11-09"],     # scrisă de aplicație
        ["MSFT", 300, "", "05.10.2026"],     # scrisă de mână, format românesc: 5 octombrie, nu 10 mai
        ["NVDA", 120, ""],                   # rând vechi, fără dată
        ["TSLA", 250, "", "nu e dată"],
    ]
    df, problems = wl.parse_watchlist(values)
    assert problems == []
    assert df["Date"].tolist()[:2] == [pd.Timestamp("2026-11-09"), pd.Timestamp("2026-10-05")]
    assert pd.isna(df["Date"].iloc[2]) and pd.isna(df["Date"].iloc[3])
    # foaie fără coloana Date: coloana există în DataFrame, goală
    df, _ = wl.parse_watchlist([["Symbol", "TargetPrice"], ["AAPL", 1]])
    assert "Date" in df.columns and pd.isna(df["Date"].iloc[0])

    # scrierea urmează antetul, nu poziția
    assert wl.sheet_row(["Symbol", "TargetPrice", "Notes", "Date"], "AAPL", 150, "n", "2026-11-09") == ["AAPL", 150.0, "n", "2026-11-09"]
    assert wl.sheet_row(["Symbol", "TargetPrice", "Notes"], "AAPL", 150, "n", "2026-11-09") == ["AAPL", 150.0, "n"]
    assert wl.sheet_row(["Date", "Notes", "Symbol", "TargetPrice", "Alta"], "AAPL", 0.495, None, "2026-11-09") == \
        ["2026-11-09", "", "AAPL", 0.495, ""]
    assert wl.sheet_row(["Simbol", "Tinta"], "AAPL", 1, "", "x") is None
    assert wl.sheet_row([], "AAPL", 1, "", "x") is None


def test_moneda_din_yahoo_cu_rezerva_sufixul():
    assert not wl.needs_yahoo_currency("AAPL") and not wl.needs_yahoo_currency("TLV.RO")
    assert wl.needs_yahoo_currency("EGLN.L") and wl.needs_yahoo_currency("SAP.DE") and wl.needs_yahoo_currency("brk-b.sw")
    assert wl.currency_label("EUR", "EGLN.L") == "EUR"          # Yahoo are prioritate față de sufix (.L → GBp)
    assert wl.currency_label("GBp", "SHEL.L") == "GBp"
    assert wl.currency_label(None, "EGLN.L") == "GBp (dedusă)"  # fără Yahoo: ghicit, marcat
    assert wl.currency_label(None, "AAPL") == "USD"
    assert wl.currency_label(None, "TLV.RO") == "RON"
    assert wl.currency_label(None, "XYZ.QQ") == "N/A"


# --- Coloanele Moment și Scor fundamental ---
def _closes(**series):
    idx = pd.bdate_range(end="2026-11-06", periods=300)    # ~14 luni: ajunge pentru momentumul 12-1
    return pd.DataFrame({k: v(np.arange(300)) for k, v in series.items()}, index=idx)


def test_moment_cells_pret_constant():
    # Preț constant: trend 0 din 3 (comparațiile sunt stricte) → 0; momentum 0% → (0 + 0,3) / 0,6 = 0,5.
    # Scor = (0 + 0,5) / 2 × 100 = 25 → BVB sub 40 = NEFAVORABIL.
    c = _closes(**{"TLV.RO": lambda i: np.full(len(i), 30.0), "AAPL": lambda i: np.full(len(i), 200.0)})
    cells = wl.moment_cells(c, ["TLV.RO", "AAPL", "LIPSA"])
    assert cells["TLV.RO"] == "NEFAVORABIL 25,0"
    assert cells["AAPL"] == "25,0 · fără verdict"
    assert cells["LIPSA"] == "N/A"


def test_moment_cells_crestere_si_istoric_scurt():
    # +0,2%/zi: preț peste SMA 50 și SMA 200, SMA 50 peste SMA 200 → trend 1; momentum 12-1 ≈ +58% (> +30%) → 1.
    c = _closes(**{"SNP.RO": lambda i: 0.5 * 1.002 ** i, "NOU": lambda i: np.where(i >= 260, 10.0, np.nan)})
    cells = wl.moment_cells(c, ["SNP.RO", "NOU"])
    assert cells["SNP.RO"] == "FAVORABIL 100,0"
    assert cells["NOU"] == "N/A"      # 40 de ședințe: nici trend (cere 50), nici momentum


def test_fundamental_cells():
    assert wl.bvb_fundamental_cell(72.4, 2.0) == "FAVORABIL 72"
    assert wl.bvb_fundamental_cell(55.0, np.nan) == "FĂRĂ TREAPTĂ 55"
    assert wl.bvb_fundamental_cell(np.nan, 1.0) == "N/A"
    assert wl.global_fundamental_cell(63.0, "NEUTRU") == "NEUTRU 63,0 · netestat"
    # 69,6 e sub pragul 70 (NEUTRU): fără zecimală ar fi apărut „NEUTRU 70”, în contradicție cu banda
    assert wl.global_fundamental_cell(69.6, "NEUTRU") == "NEUTRU 69,6 · netestat"
    assert wl.global_fundamental_cell(None, "DATE INSUFICIENTE") == "N/A"
