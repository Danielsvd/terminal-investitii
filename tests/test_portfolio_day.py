"""Rezumatul Zilei, etapa B — „Portofoliul meu azi” (analytics/portfolio_day.py). Doar valori sintetice."""
import numpy as np
import pandas as pd
import pytest

from analytics import portfolio_day as pdy

DAYS = pd.bdate_range("2026-03-02", periods=30)
SESSION, PREV = DAYS[-1], DAYS[-2]
FX = {"RON": 1.0, "USD": 4.5, "EUR": 5.0, "GBP": 5.8}


def _flat_then(prev, last, n=len(DAYS)):
    """Serie constantă la `prev`, cu ultima zi la `last`."""
    return [prev] * (n - 1) + [last]


def _setup():
    close = pd.DataFrame({
        "AAA.RO": _flat_then(10.0, 11.0),          # +100 lei pe 100 de acțiuni
        "BBB": _flat_then(100.0, 95.0),            # −50 USD pe 10 acțiuni = −225 lei
        "CCC.DE": _flat_then(200.0, 210.0),        # +50 EUR pe 5 acțiuni = +250 lei
        "DDD.RO": _flat_then(5.0, 5.0)[:-1] + [np.nan],   # fără cotație în ședință
        "EEE": _flat_then(50.0, 51.0),             # CHF: fără curs în dicționar
    }, index=DAYS)
    holdings = pd.DataFrame({
        "Symbol": ["AAA.RO", "BBB", "CCC.DE", "DDD.RO", "EEE"],
        "Currency": ["RON", "USD", "EUR", "RON", "CHF"],
        "Quantity": [100, 10, 5, 40, 3],
    })
    return holdings, close


def test_pl_pe_pozitii_si_total():
    holdings, close = _setup()
    rows, tot = pdy.portfolio_day(holdings, close, FX, trade_dates={"CCC.DE": SESSION})
    r = rows.set_index("Symbol")
    assert r.loc["AAA.RO", "PnL_RON"] == pytest.approx(100.0)
    assert r.loc["BBB", "PnL"] == pytest.approx(-50.0)
    assert r.loc["BBB", "PnL_RON"] == pytest.approx(-225.0)
    assert r.loc["CCC.DE", "PnL_RON"] == pytest.approx(250.0)
    # valoarea din ziua precedentă: 1.000 + 4.500 + 5.000 = 10.500 lei; P/L 125 lei → +1,1905%
    assert tot["prev_ron"] == pytest.approx(10_500.0)
    assert tot["pnl_ron"] == pytest.approx(125.0)
    assert tot["pct"] == pytest.approx(125.0 / 10_500.0)
    assert r.loc["CCC.DE", "Contribution"] == pytest.approx(250.0 / 10_500.0)
    assert rows["Contribution"].dropna().sum() == pytest.approx(tot["pct"])   # suma contribuțiilor = variația
    assert tot["top"] == ["CCC.DE", "AAA.RO"] and tot["bottom"] == ["BBB"]
    assert list(rows["Symbol"][:3]) == ["CCC.DE", "AAA.RO", "BBB"]           # sortat după contribuție


def test_excluse_si_marcate():
    holdings, close = _setup()
    rows, tot = pdy.portfolio_day(holdings, close, FX, trade_dates={"CCC.DE": SESSION, "AAA.RO": PREV})
    r = rows.set_index("Symbol")
    assert r.loc["DDD.RO", "Status"] == pdy.STATUS_STALE and pd.isna(r.loc["DDD.RO", "PnL_RON"])
    assert r.loc["EEE", "Status"] == pdy.STATUS_NO_FX and r.loc["EEE", "PnL"] == pytest.approx(3.0)
    assert dict(tot["excluded"]) == {"DDD.RO": pdy.STATUS_STALE, "EEE": pdy.STATUS_NO_FX}
    assert tot["n"] == 3 and tot["held"] == 5
    assert tot["approx"] == ["CCC.DE"]         # tranzacție în ziua ședinței; AAA.RO în ziua precedentă nu contează


def test_pe_piete_cu_data_sedintei():
    holdings, close = _setup()
    _, tot = pdy.portfolio_day(holdings, close, FX)
    bm = tot["by_market"]
    assert bm["bvb"]["pnl_ron"] == pytest.approx(100.0) and bm["bvb"]["n"] == 1 and bm["bvb"]["held"] == 2
    assert bm["us"]["pct"] == pytest.approx(-225.0 / 4_500.0)
    assert bm["eu"]["pct"] == pytest.approx(250.0 / 5_000.0)
    assert all(bm[m]["session"] == SESSION for m in ("bvb", "us", "eu"))


def test_sedinta_data_din_afara_taie_randurile_mai_noi():
    """Ședința SUA dată din universul întreg (miercuri): un rând nou (pre-market joi) nu intră."""
    holdings = pd.DataFrame({"Symbol": ["BBB"], "Currency": ["USD"], "Quantity": [10]})
    close = pd.DataFrame({"BBB": _flat_then(100.0, 95.0)}, index=DAYS)
    rows, tot = pdy.portfolio_day(holdings, close, FX, sessions={"us": PREV})
    assert rows.loc[0, "Status"] == pdy.STATUS_OK and rows.loc[0, "PnL"] == pytest.approx(0.0)
    assert tot["by_market"]["us"]["session"] == PREV


def test_sigma_peste_doua_alerta():
    idx = pd.bdate_range("2025-11-03", periods=90)
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, len(idx) - 1)
    rets[-1] = 0.08                            # ~8σ
    prices = 100 * np.concatenate([[1.0], np.cumprod(1 + rets)])
    close = pd.DataFrame({"ZZZ": prices}, index=idx)
    holdings = pd.DataFrame({"Symbol": ["ZZZ"], "Currency": ["USD"], "Quantity": [1]})
    rows, tot = pdy.portfolio_day(holdings, close, FX)
    assert rows.loc[0, "Sigma"] > pdy.SIGMA_ALERT and tot["alerts"] == ["ZZZ"]


def test_fara_pret_si_registru_gol():
    holdings = pd.DataFrame({"Symbol": ["XXX"], "Currency": ["USD"], "Quantity": [1]})
    rows, tot = pdy.portfolio_day(holdings, pd.DataFrame(), FX)
    assert rows.loc[0, "Status"] == pdy.STATUS_NO_PRICE
    assert tot["pnl_ron"] is None and tot["pct"] is None and tot["n"] == 0
    rows0, tot0 = pdy.portfolio_day(pdy.held_positions(None), pd.DataFrame(), FX)
    assert rows0.empty and tot0["pnl_ron"] is None


def test_held_positions_fara_obligatiuni_si_inchise():
    pos = pd.DataFrame({"Symbol": ["A.RO", "R2501A", "B"], "Class": ["Actiuni RO", "Obligatiuni", "Actiuni SUA"],
                        "Currency": ["RON", "RON", "USD"], "Quantity": [10.0, 5.0, 0.0]})
    assert list(pdy.held_positions(pos)["Symbol"]) == ["A.RO"]


def test_curs_bnr_si_pence():
    hist = pd.DataFrame({"EUR": [5.0, 5.01, np.nan], "USD": [4.4, 4.41, 4.42]},
                        index=pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04"]))
    rates, dates = pdy.fx_from_bnr(hist)
    assert rates["EUR"] == 5.01 and dates["EUR"] == pd.Timestamp("2026-03-03")     # NaN de la final ignorat
    assert rates["USD"] == 4.42 and rates["RON"] == 1.0
    assert pdy.rate_for("GBX", FX) == pytest.approx(0.058)
    assert pdy.rate_for("CHF", FX) is None


def test_ultima_tranzactie_pe_simbol():
    led = pd.DataFrame({"Date": pd.to_datetime(["2026-01-05 10:00", "2026-02-03 11:00", "2026-02-10 09:00"]),
                        "Symbol": ["A", "A", ""], "Type": ["BUY", "DIV", "DEPOSIT"]})
    assert pdy.last_trade_dates(led) == {"A": pd.Timestamp("2026-01-05")}      # DIV nu schimbă cantitatea
