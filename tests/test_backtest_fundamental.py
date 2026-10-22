"""Teste pentru analytics/backtest_fundamental.py și tools/backtest_fundamental.py. Date sintetice, fără rețea."""
import numpy as np
import pandas as pd
import pytest

from analytics import backtest_fundamental as bf
from analytics.global_fundamental import fundamental_score

YEARS = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]


def _stmts(shares=(100.0, 100.0, 100.0, 100.0), ni=(10.0, 11.0, 12.0, 13.0)):
    cols = pd.to_datetime(YEARS)
    income = pd.DataFrame({c: {"Net Income": ni[i], "Total Revenue": 100.0 + 10 * i, "Gross Profit": 40.0 + 5 * i,
                               "EBIT": 15.0 + i, "EBITDA": 20.0 + i, "Interest Expense": 1.0,
                               "Tax Provision": 3.0, "Pretax Income": 15.0, "Diluted Average Shares": shares[i]}
                           for i, c in enumerate(cols)})
    balance = pd.DataFrame({c: {"Total Assets": 200.0, "Total Liabilities Net Minority Interest": 100.0,
                                "Current Assets": 80.0, "Current Liabilities": 40.0, "Retained Earnings": 50.0,
                                "Stockholders Equity": 100.0, "Total Debt": 30.0,
                                "Cash Cash Equivalents And Short Term Investments": 10.0, "Long Term Debt": 25.0,
                                "Ordinary Shares Number": shares[i]}
                            for i, c in enumerate(cols)})
    cashflow = pd.DataFrame({c: {"Operating Cash Flow": 18.0 + i, "Capital Expenditure": -4.0}
                             for i, c in enumerate(cols)})
    # Yahoo dă coloanele cea mai recentă prima
    return income.iloc[:, ::-1], balance.iloc[:, ::-1], cashflow.iloc[:, ::-1]


def test_an_fiscal_cunoscut_dupa_90_de_zile():
    inc, _, _ = _stmts()
    # 31.12.2022 + 90 de zile = 31.03.2023
    assert pd.Timestamp(bf.statements_as_of(inc, "2023-03-30").columns[0]) == pd.Timestamp("2021-12-31")
    assert pd.Timestamp(bf.statements_as_of(inc, "2023-03-31").columns[0]) == pd.Timestamp("2022-12-31")
    assert bf.statements_as_of(inc, "2022-03-30") is None


def test_valoare_lunara_cu_decalaj_de_publicare():
    aaa = pd.Series([5.0, 5.2, 5.4], index=pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"]))
    # La 31 mai, media lui mai nu e publicată încă: se ia aprilie
    assert bf.value_as_of(aaa, "2024-05-31", publication_lag=pd.DateOffset(months=1)) == 5.0
    assert bf.value_as_of(aaa, "2024-06-03", publication_lag=pd.DateOffset(months=1)) == 5.2
    assert bf.value_as_of(aaa, "2024-03-15", publication_lag=pd.DateOffset(months=1)) is None


def test_splituri_raportate_si_retratate():
    splits = pd.Series([10.0], index=pd.to_datetime(["2023-06-10"]))
    assert bf.split_factor_after(splits, "2022-12-31") == 10.0
    assert bf.split_factor_after(splits, "2023-12-31") == 1.0
    # Raportat: 100 acțiuni înainte de split, 1.000 după → anii vechi trebuie înmulțiți cu 10
    inc, bal, _ = _stmts(shares=(100.0, 100.0, 1000.0, 1000.0))
    assert bf.share_basis(inc, bal, splits) == "reported"
    adj = bf.split_adjust_shares(inc, splits)
    assert adj.loc["Diluted Average Shares", pd.Timestamp("2022-12-31")] == 1000.0
    assert adj.loc["Net Income", pd.Timestamp("2022-12-31")] == 11.0          # profitul nu se atinge
    # Retratat: Yahoo dă deja 1.000 și pentru anii vechi
    inc, bal, _ = _stmts(shares=(1000.0, 1000.0, 1000.0, 1000.0))
    assert bf.share_basis(inc, bal, splits) == "restated"
    assert bf.share_basis(inc, bal, pd.Series(dtype=float)) == "none"
    # Nici una, nici alta (300 față de 1.000, split 10): nedecis
    inc, bal, _ = _stmts(shares=(300.0, 300.0, 1000.0, 1000.0))
    assert bf.share_basis(inc, bal, splits) is None


def test_intrarea_la_data_calculata_de_mana():
    inc, bal, cf = _stmts()
    # La 30.06.2024 ultimul an publicat e 2023: profit 12, 100 de acțiuni → EPS 0,12; preț 2,4 → P/E 20
    inp, fy = bf.inputs_at("2024-06-30", 2.4, inc, bal, cf, "Technology", "TEST", 0.04, 1.0, 5.0)
    assert pd.Timestamp(fy) == pd.Timestamp("2023-12-31")
    assert inp["pe"] == pytest.approx(20.0) and inp["eps"] == pytest.approx(0.12)
    assert inp["pbv"] == pytest.approx(240 / 100)                        # capitalizare 240 / capital 100
    assert inp["net_debt"] == pytest.approx(20.0) and inp["fcf"] == pytest.approx(16.0)   # 20 − 4
    assert inp["piotroski"]["year"] == pd.Timestamp("2023-12-31")
    assert inp["piotroski"]["prior_year"] == pd.Timestamp("2022-12-31")
    row = bf.score_row("TEST", "2024-06-30", inp, fy)
    assert row["fund_score"] == fundamental_score(inp)["score"]
    # E/P 5% / AAA 5% = 1× → componenta ia 0
    assert row["c_E/P față de AAA"] == 0.0


def test_fara_informatie_din_viitor():
    inc, bal, cf = _stmts()
    base, _ = bf.inputs_at("2025-03-30", 2.4, inc, bal, cf, "Technology", "TEST", 0.04, 1.0, 5.0)
    inc2 = inc.copy()
    inc2.loc["Net Income", pd.Timestamp("2024-12-31")] = -999.0          # anul 2024 devine cunoscut abia la 31.03.2025
    same, _ = bf.inputs_at("2025-03-30", 2.4, inc2, bal, cf, "Technology", "TEST", 0.04, 1.0, 5.0)
    assert same["pe"] == base["pe"] and same["piotroski"]["passed"] == base["piotroski"]["passed"]


def test_banca_si_date_lipsa():
    inc, bal, cf = _stmts()
    inp, _ = bf.inputs_at("2024-06-30", 2.4, inc, bal, cf, "Financial Services", "BANK", 0.04, 1.2, 5.0)
    assert inp["financial"] and inp["roe"] == pytest.approx(0.12) and inp["ke"] == pytest.approx(0.04 + 1.2 * 0.05)
    inp, why = bf.inputs_at("2022-01-31", 2.4, inc, bal, cf, "Technology", "TEST", 0.04, 1.0, 5.0)
    assert inp is None and "nicio situație" in why
    # Fără rata fără risc: WACC N/A, dar scorul rămâne pe ceilalți piloni (DCF invers N/A, nu 0)
    inp, fy = bf.inputs_at("2024-06-30", 2.4, inc, bal, cf, "Technology", "TEST", None, None, 5.0)
    row = bf.score_row("TEST", "2024-06-30", inp, fy)
    assert inp["wacc"] is None and row["c_DCF invers"] is None and row["fund_score"] is not None


def test_beta_la_data():
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    rng = np.random.default_rng(1)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    asset = bench ** 1.5                                               # randamente ≈ 1,5 × benchmark
    b = bf.beta_as_of(asset, bench, "2024-06-30")
    assert 1.3 < b < 1.7
    assert bf.beta_as_of(asset, bench, "2020-06-30") is None             # sub 52 de săptămâni


# --- Scriptul, cap-coadă -------------------------------------------------------

def _fake_prices(symbol, start, end, refresh=False):
    if symbol == "LIPSA":
        return pd.DataFrame()
    idx = pd.bdate_range("2019-01-01", "2026-06-30")
    rng = np.random.default_rng(sum(map(ord, symbol)))
    if symbol == "^TNX":
        s = pd.Series(4.0, index=idx)
        return pd.DataFrame({"adj": s, "raw": s, "volume": 0.0})
    adj = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx)))), index=idx)
    return pd.DataFrame({"adj": adj, "raw": adj * 1.02, "volume": 1000.0})


def _fake_fund(symbol, refresh=False):
    k = sum(map(ord, symbol)) % 7
    inc, bal, cf = _stmts(ni=tuple(8.0 + k + i for i in range(4)))
    sector = "Financial Services" if symbol == "BANCA" else "Technology"
    return {"income": inc, "balance": bal, "cashflow": cf, "sector": sector, "splits": pd.Series(dtype=float)}


def test_rulare_script(tmp_path, monkeypatch, capsys):
    from tools import backtest_fundamental as tbf
    monkeypatch.setattr(tbf, "download", _fake_prices)
    monkeypatch.setattr(tbf, "download_fundamentals", _fake_fund)
    monkeypatch.setattr(tbf, "download_fred", lambda code, refresh=False: pd.Series(
        5.0, index=pd.date_range("2018-01-01", "2026-06-01", freq="MS")))
    sym = tmp_path / "s.txt"
    sym.write_text("AAA\nBBB\nCCC\nDDD\nEEE\nFFF\nBANCA\nLIPSA\n", encoding="utf-8")
    out = tmp_path / "rezultat_fund.csv"
    code = tbf.main(["--simboluri", str(sym), "--start", "2023-01-01", "--end", "2025-06-30", "--out", str(out)])
    assert code == 0
    df = pd.read_csv(out)
    assert df["symbol"].nunique() == 7
    assert {"fund_score", "p_evaluare", "p_calitate", "p_sanatate", "p_rentabilitate", "fwd_12m", "fiscal_year"} <= set(df.columns)
    # Înainte de 31.03.2023 se folosește anul fiscal 2021, după, 2022
    a = df[df["symbol"] == "AAA"].set_index("date")["fiscal_year"]
    assert a["2023-02-28"] == "2021-12-31" and a["2023-04-28"] == "2022-12-31"
    rez = pd.read_csv(str(out).replace(".csv", "_rezumat.csv"))
    assert {"benzi", "cincimi"} <= set(rez["grupare"]) and "fund_score" in set(rez["scor"])
    printed = capsys.readouterr().out
    assert "Fără date: LIPSA" in printed and "ani fiscali distincți" in printed and "Orientativ" in printed
