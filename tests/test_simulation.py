"""Teste pentru Monte Carlo prin bootstrap și optimizarea cu plafon. Date construite, fără rețea."""
import numpy as np
import pandas as pd
import pytest

from analytics import simulation as S

IDX = pd.bdate_range("2023-01-02", periods=301)


def test_bootstrap_cu_randament_constant_e_determinist():
    curve = pd.Series(100.0 * 1.001 ** np.arange(301), index=IDX)          # +0,1% în fiecare zi
    paths = S.bootstrap_paths(curve, days_ahead=252, simulations=50)
    assert paths.shape == (253, 50)
    last = curve.iloc[-1]
    assert np.allclose(paths[0], last)
    # orice reeșantionare a unui randament constant dă aceeași traiectorie: ultima × 1,001^252
    assert np.allclose(paths[-1], last * 1.001 ** 252)
    summary = S.summarize_paths(paths)
    assert summary["prob_profit"] == 100.0
    assert summary["change_p5_pct"] == pytest.approx((1.001 ** 252 - 1) * 100)
    assert summary["simulations"] == 50 and summary["days"] == 252
    assert list(summary["bands"].columns) == ["p5", "p50", "p95"] and len(summary["bands"]) == 253
    # fără media istorică, un randament constant nu mai mișcă nimic
    flat = S.bootstrap_paths(curve, simulations=50, demean=True)
    assert np.allclose(flat[-1], last)


def test_bootstrap_reproductibil_semn_si_esantion_mic():
    rng = np.random.default_rng(7)
    curve = pd.Series(100.0 * np.cumprod(1 + rng.normal(-0.001, 0.02, 301)), index=IDX)   # tendință de scădere
    a, b = S.bootstrap_paths(curve), S.bootstrap_paths(curve)
    assert np.array_equal(a, b)                                            # același seed, același rezultat
    summary = S.summarize_paths(a)
    assert summary["p5"] < summary["p50"] < summary["p95"]
    assert summary["change_p5_pct"] < 0                                    # pierderea are semn negativ, o singură dată
    assert 0 <= summary["prob_profit"] <= 100
    assert summary["prob_profit"] == pytest.approx((a[-1] > a[0, 0]).mean() * 100)
    assert S.bootstrap_paths(curve.iloc[:40]) is None                      # sub 60 de randamente
    assert S.bootstrap_paths(pd.Series(dtype="float64")) is None


def _returns(n_assets, seed, days=600, vols=None, means=None):
    rng = np.random.default_rng(seed)
    vols = vols or [0.01] * n_assets
    means = means or [0.0004] * n_assets
    data = {f"A{i}": rng.normal(means[i], vols[i], days) for i in range(n_assets)}
    return pd.DataFrame(data, index=pd.bdate_range("2021-01-04", periods=days))


def test_optimize_weights_plafon_si_varianta_minima():
    returns = _returns(5, 1, means=[0.003, 0.0002, 0.0002, 0.0002, 0.0002])    # A0 mult peste restul
    out = S.optimize_weights(returns, rf_annual=0.0, max_weight=0.25)
    assert out["cap"] == 0.25
    assert out["max_sharpe"].sum() == pytest.approx(1.0)
    assert out["max_sharpe"]["A0"] == pytest.approx(0.25, abs=1e-4)            # fără plafon ar fi luat aproape tot
    assert (out["max_sharpe"] <= 0.25 + 1e-9).all() and (out["max_sharpe"] >= -1e-12).all()
    assert 0.0 <= out["shrinkage"] <= 1.0
    # două active necorelate cu aceeași volatilitate: varianța minimă e jumătate-jumătate
    two = _returns(2, 3, days=4000)
    mv = S.optimize_weights(two, max_weight=1.0)["min_variance"]
    assert mv["A0"] == pytest.approx(0.5, abs=0.03)
    # activul de trei ori mai volatil primește mult mai puțin
    skew = _returns(2, 4, days=4000, vols=[0.01, 0.03])
    assert S.optimize_weights(skew, max_weight=1.0)["min_variance"]["A0"] > 0.85


def test_effective_cap_cu_putine_active():
    assert S.effective_cap(10, 0.25) == 0.25
    assert S.effective_cap(3, 0.25) == pytest.approx(1 / 3)                    # 3 × 25% nu poate ajunge la 100%
    out = S.optimize_weights(_returns(3, 5), max_weight=0.25)
    assert out["max_sharpe"].max() <= 1 / 3 + 1e-6 and out["max_sharpe"].sum() == pytest.approx(1.0)


def test_optimize_portfolio_rezultat_si_in_afara_esantionului():
    closes = (1 + _returns(4, 9, days=700)).cumprod() * 100
    result, message = S.optimize_portfolio(closes, rf_annual=0.03, max_weight=0.4)
    assert message == "Optimizare finalizată."
    assert sum(result["allocation"].values()) == pytest.approx(100.0, abs=0.05)
    assert sum(result["min_variance"].values()) == pytest.approx(100.0, abs=0.05)
    assert max(result["allocation"].values()) <= 40.0 + 0.01 and result["cap"] == 40.0
    assert result["n_days"] == 699
    assert [r["label"] for r in result["out_of_sample"]] == ["Sharpe maxim", "Varianță minimă", "Ponderi egale"]
    assert result["test_days"] == 699 - int(699 * 0.7)
    # statisticile raportate sunt cele ale ponderilor raportate
    returns = closes.pct_change(fill_method=None).dropna()
    check = S.weights_stats(pd.Series(result["allocation"]) / 100, returns, 0.03)
    assert check["return"] == pytest.approx(result["expected_return"], abs=0.05)


def test_optimize_portfolio_date_insuficiente():
    closes = (1 + _returns(3, 2, days=100)).cumprod()
    assert S.optimize_portfolio(closes)[0] is None and "99 zile comune" in S.optimize_portfolio(closes)[1]
    assert S.optimize_portfolio(closes[["A0"]])[0] is None
    gappy = (1 + _returns(2, 2, days=400)).cumprod()
    gappy.iloc[::2, 1] = np.nan                                                # al doilea simbol are preț o zi din două
    assert S.optimize_portfolio(gappy)[0] is None                              # nicio zi cu randament pentru ambele


def test_trim_short_history_scoate_pozitia_listata_recent():
    closes = (1 + _returns(4, 11, days=700)).cumprod() * 100
    closes.iloc[:650, 3] = np.nan                                              # A3 e listat de 50 de zile
    kept, dropped = S.trim_short_history(closes, 250)
    assert dropped == ["A3"] and list(kept.columns) == ["A0", "A1", "A2"]
    result, _ = S.optimize_portfolio(closes, max_weight=0.5)
    assert result["excluded"] == ["A3"] and set(result["allocation"]) == {"A0", "A1", "A2"} and result["n_days"] == 699
    # doar două simboluri, unul prea scurt: nu se scoate nimic, iar mesajul spune câte zile comune sunt
    two = closes[["A0", "A3"]]
    assert S.trim_short_history(two, 250)[1] == []
    out, message = S.optimize_portfolio(two)
    assert out is None and "49 zile comune" in message
    # toate cu istoric complet: nimic scos
    assert S.trim_short_history(closes[["A0", "A1", "A2"]], 250)[1] == []
