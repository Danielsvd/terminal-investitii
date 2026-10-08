"""Teste pentru analytics/risk.py, cu valori calculate de mână."""
import numpy as np
import pandas as pd
import pytest

from analytics.risk import beta_benchmark, beta_weekly


def _prices(returns, start="2026-01-02"):
    """Prețuri săptămânale (vineri) care produc exact randamentele date."""
    idx = pd.date_range(start, periods=len(returns) + 1, freq="W-FRI")
    return pd.Series(100.0 * np.cumprod([1.0] + [1 + r for r in returns]), index=idx)


def test_beta_exemplu_calculat_de_mana():
    # benchmark: 1%, -2%, 3% (medie 0,6667%); activ: 2%, -1%, 3% (medie 1,3333%)
    # cov × 2 = 0,0000222 + 0,0006222 + 0,0003889 = 0,0010333
    # var × 2 = 0,0000111 + 0,0007111 + 0,0005444 = 0,0012667
    # beta = 0,0010333 / 0,0012667 = 31/38 = 0,8158
    out = beta_weekly(_prices([0.02, -0.01, 0.03]), _prices([0.01, -0.02, 0.03]), min_obs=3)
    assert out["beta"] == pytest.approx(31 / 38, abs=1e-6)
    assert out["n"] == 3


def test_beta_activ_care_amplifica_benchmarkul():
    rng = np.random.default_rng(7)
    bench = rng.normal(0.002, 0.02, 120)
    out = beta_weekly(_prices(1.5 * bench), _prices(bench))
    assert out["beta"] == pytest.approx(1.5, abs=1e-9) and out["n"] == 120


def test_beta_date_zilnice_cu_fus_orar_si_fara():
    days = pd.bdate_range("2023-01-02", periods=600)
    rng = np.random.default_rng(3)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))), index=days)
    asset = pd.Series(bench.values ** 2 / 100, index=days.tz_localize("Europe/Bucharest"))
    out = beta_weekly(asset, bench)          # randament activ ≈ 2 × randament benchmark
    assert out["beta"] == pytest.approx(2.0, abs=0.05) and out["n"] >= 100


def test_beta_esantion_mic_sau_date_lipsa_da_none():
    assert beta_weekly(_prices([0.01] * 10), _prices([0.02, -0.01] * 5)) is None   # 10 < 52
    assert beta_weekly(None, _prices([0.01, 0.02])) is None
    assert beta_weekly(pd.Series(dtype=float), pd.Series(dtype=float)) is None
    flat = _prices([0.0] * 60)
    assert beta_weekly(_prices([0.01, -0.01] * 30), flat) is None                  # benchmark fără variație


def test_beta_saptamanile_fara_tranzactii_nu_devin_randament_zero():
    rng = np.random.default_rng(11)
    bench = _prices(rng.normal(0.002, 0.02, 80))
    asset = _prices(1.2 * bench.pct_change().dropna().values)
    asset.iloc[10:15] = np.nan                       # 5 săptămâni fără cotație
    out = beta_weekly(asset, bench)
    assert out["n"] < 80 and out["beta"] == pytest.approx(1.2, abs=1e-9)


def test_beta_benchmark_pe_piete():
    assert beta_benchmark("SNP.RO", "RON")[0] == "TVBETETF.RO"
    assert "proxy" in beta_benchmark("snp.ro", None)[1]
    assert beta_benchmark("SAP.DE", "EUR")[0] == "^GDAXI"
    assert beta_benchmark("MC.PA", "EUR")[0] == "^STOXX50E"
    assert beta_benchmark("AAPL", "USD")[0] == "^GSPC"
    assert beta_benchmark("VOD.L", "GBp") == (None, None)
