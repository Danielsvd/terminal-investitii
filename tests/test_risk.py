"""Teste pentru analytics/risk.py, cu valori calculate de mână."""
import numpy as np
import pandas as pd
import pytest

from analytics.risk import beta_benchmark, beta_weekly, jensen_alpha


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


# --- Alpha Jensen ------------------------------------------------------------

def _line(start_value, end_value, periods=253, end="2026-10-08", tz=None):
    idx = pd.bdate_range(end=end, periods=periods, tz=tz)
    return pd.Series(np.linspace(start_value, end_value, periods), index=idx)


def test_alpha_exemplu_calculat_de_mana():
    # activ +20%, benchmark +10%, beta 1,2, rf 6%
    # alpha = 20% − [6% + 1,2 × (10% − 6%)] = 20% − 10,8% = 9,2%
    out = jensen_alpha(_line(100, 120), _line(100, 110), beta=1.2, risk_free=0.06)
    assert out["alpha"] == pytest.approx(0.092, abs=1e-9)
    assert out["asset_return"] == pytest.approx(0.20) and out["bench_return"] == pytest.approx(0.10)


def test_alpha_foloseste_doar_ultimul_an_si_ignora_fusul_orar():
    # 3 ani de date: doar ultimul an contează (primii doi ani sunt plați la 50, apoi 100 -> 120)
    asset = pd.concat([_line(50, 50, periods=500, end="2025-10-01"), _line(100, 120)])
    asset.index = asset.index.tz_localize("Europe/Bucharest")
    out = jensen_alpha(asset, _line(100, 110, periods=800), beta=1.0, risk_free=0.0)
    assert out["asset_return"] == pytest.approx(0.20, abs=0.01)
    assert (out["end"] - out["start"]).days <= 366


def test_alpha_fara_beta_rf_sau_istoric_este_none():
    asset, bench = _line(100, 120), _line(100, 110)
    assert jensen_alpha(asset, bench, beta=None, risk_free=0.06) is None      # beta lipsă NU se presupune 1
    assert jensen_alpha(asset, bench, beta=1.0, risk_free=None) is None
    assert jensen_alpha(asset, None, beta=1.0, risk_free=0.06) is None
    assert jensen_alpha(_line(100, 120, periods=80), bench, beta=1.0, risk_free=0.06) is None   # ~4 luni
    assert jensen_alpha(pd.Series(dtype=float), bench, beta=1.0, risk_free=0.06) is None


# --- Risc pe randamentele portofoliului ----------------------------------------

def test_performance_risk_valori_calculate_de_mana():
    import numpy as np
    import pytest
    from analytics.risk import performance_risk
    idx = pd.bdate_range("2024-01-01", periods=5)
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=idx)
    out = performance_risk(r, rf_annual=0.0, min_obs=5)
    # media 0,006 ; abateri pătrate: 1,6e-5 + 6,76e-4 + 5,76e-4 + 2,56e-4 + 1,96e-4 = 1,72e-3 ; /4 -> std 0,0207364
    assert out["vol_ann"] == pytest.approx(0.0207364 * np.sqrt(252), rel=1e-5)
    assert out["sharpe"] == pytest.approx(0.006 / 0.0207364 * np.sqrt(252), rel=1e-5)
    # deviația în jos: (0,02² + 0,01²) / 5 = 1e-4 -> 0,01
    assert out["sortino"] == pytest.approx(0.006 / 0.01 * np.sqrt(252), rel=1e-9)
    # curba: 1,01 ; 0,9898 ; ... -> cea mai mare scădere de la vârf e 0,9898 / 1,01 - 1 = -2%
    assert out["max_drawdown"] == pytest.approx(-0.02)
    # percentila 5 din [-0,02 ; -0,01 ; 0,01 ; 0,02 ; 0,03]: -0,02 + 0,2 × 0,01 = -0,018 ; sub ea doar -0,02
    assert out["var95_hist"] == pytest.approx(-0.018)
    assert out["cvar95"] == pytest.approx(-0.02)
    assert out["var95_param"] == pytest.approx(0.006 - 1.6448536 * 0.0207364, rel=1e-5)
    assert out["n"] == 5 and out["start"] == idx[0] and out["end"] == idx[-1]


def test_performance_risk_rf_si_esantion_mic():
    import pytest
    from analytics.risk import performance_risk
    idx = pd.bdate_range("2024-01-01", periods=5)
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=idx)
    assert performance_risk(r, 0.0) is None                                   # sub 60 de observații: fără indicatori
    no_rf = performance_risk(r, None, min_obs=5)
    assert no_rf["sharpe"] is None and no_rf["sortino"] is None and no_rf["vol_ann"] > 0
    with_rf = performance_risk(r, 0.0756, min_obs=5)                          # rf zilnic 0,0003
    assert with_rf["sharpe"] == pytest.approx((0.006 - 0.0003) / 0.0207364 * 252 ** 0.5, rel=1e-5)
    # pierdere din prima zi: drawdown măsurat față de punctul de plecare
    loss = performance_risk(pd.Series([-0.10, 0.01, 0.01, 0.01, 0.01], index=idx), 0.0, min_obs=5)
    assert loss["max_drawdown"] == pytest.approx(-0.10)
