"""Simulare Monte Carlo prin bootstrap și optimizare de portofoliu cu plafon pe poziție.

Funcții pure: fără Streamlit, fără rețea. Ambele sunt scenarii construite din trecut,
nu prognoze: rezultatele se prezintă împreună cu ipotezele și cu o verificare în afara
eșantionului.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


# --- Monte Carlo ---------------------------------------------------------------

def bootstrap_paths(curve, days_ahead=TRADING_DAYS, simulations=1000, seed=42, demean=False, min_obs=60):
    """Traiectorii viitoare obținute prin reeșantionarea randamentelor zilnice istorice.

    Fiecare zi simulată primește un randament tras la întâmplare (cu revenire) din cele
    observate, deci cozile și asimetria reale se păstrează, fără ipoteza de distribuție normală.
    `demean=True` scade media istorică: arată doar efectul volatilității, fără presupunerea că
    randamentul mediu din fereastra aleasă continuă. `seed` fix: aceleași date dau același rezultat.
    Întoarce un tablou (days_ahead + 1, simulations) care pornește de la ultima valoare, sau
    None sub `min_obs` randamente.
    """
    series = pd.Series(curve, dtype="float64").dropna()
    returns = series.pct_change(fill_method=None).dropna().to_numpy()
    if len(returns) < min_obs or series.iloc[-1] <= 0:
        return None
    if demean:
        returns = returns - returns.mean()
    rng = np.random.default_rng(seed)
    sample = returns[rng.integers(0, len(returns), size=(days_ahead, simulations))]
    growth = np.vstack([np.ones((1, simulations)), np.cumprod(1.0 + sample, axis=0)])
    return float(series.iloc[-1]) * growth


def summarize_paths(paths):
    """Rezumatul simulării: percentilele finale, variația în scenariul pesimist, șansa de profit.

    - p5 / p50 / p95 : percentilele valorii la sfârșitul orizontului.
    - change_p5_pct  : (p5 / valoarea de azi − 1) × 100, cu semn (negativ = pierdere).
    - prob_profit    : procentul traiectoriilor care se termină peste valoarea de azi.
    - bands          : DataFrame cu percentilele 5, 50 și 95 pentru fiecare zi a orizontului.
    """
    start = float(paths[0, 0])
    final = paths[-1]
    p5, p50, p95 = (float(np.percentile(final, q)) for q in (5, 50, 95))
    bands = pd.DataFrame(np.percentile(paths, [5, 50, 95], axis=1).T, columns=["p5", "p50", "p95"])
    return {"start": start, "p5": p5, "p50": p50, "p95": p95, "change_p5_pct": (p5 / start - 1.0) * 100,
            "change_p50_pct": (p50 / start - 1.0) * 100, "prob_profit": float((final > start).mean() * 100),
            "simulations": int(paths.shape[1]), "days": int(paths.shape[0] - 1), "bands": bands}


# --- Optimizare ------------------------------------------------------------------

def shrunk_covariance(returns):
    """Matricea de covarianță zilnică Ledoit-Wolf: (matrice, intensitatea shrinkage-ului).

    Covarianța de eșantion pe puține date are valori extreme care fac optimizarea să
    parieze totul pe câteva perechi; Ledoit-Wolf o trage spre o structură simplă.
    """
    from sklearn.covariance import LedoitWolf
    model = LedoitWolf().fit(np.asarray(returns, dtype="float64"))
    return model.covariance_, float(model.shrinkage_)


def effective_cap(n_assets, max_weight):
    """Plafonul aplicat efectiv: cu puține active, plafonul cerut poate fi imposibil (4 × 20% < 100%)."""
    return max(float(max_weight), 1.0 / n_assets)


def _solve(objective, n_assets, cap):
    start = np.full(n_assets, 1.0 / n_assets)
    result = minimize(objective, start, method="SLSQP", bounds=[(0.0, cap)] * n_assets,
                      constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},), options={"maxiter": 500, "ftol": 1e-12})
    if not result.success:
        return None
    weights = np.clip(result.x, 0.0, cap)
    return weights / weights.sum()


def optimize_weights(returns, rf_annual=0.0, max_weight=0.25):
    """Ponderile de Sharpe maxim și de varianță minimă, cu plafon pe poziție și covarianță Ledoit-Wolf.

    returns: DataFrame de randamente zilnice, fără valori lipsă. Întoarce dict cu
    "max_sharpe" și "min_variance" (Series pe simboluri), "cap" și "shrinkage", sau None
    dacă optimizarea nu converge.
    """
    n = returns.shape[1]
    cap = effective_cap(n, max_weight)
    mean = returns.mean().to_numpy() * TRADING_DAYS
    cov_daily, shrinkage = shrunk_covariance(returns)
    cov = cov_daily * TRADING_DAYS

    def variance(w):
        return float(w @ cov @ w)

    def negative_sharpe(w):
        vol = variance(w) ** 0.5
        return -(float(w @ mean) - rf_annual) / vol if vol > 0 else 0.0

    w_sharpe, w_minvar = _solve(negative_sharpe, n, cap), _solve(variance, n, cap)
    if w_sharpe is None or w_minvar is None:
        return None
    return {"max_sharpe": pd.Series(w_sharpe, index=returns.columns), "min_variance": pd.Series(w_minvar, index=returns.columns),
            "cap": cap, "shrinkage": shrinkage}


def weights_stats(weights, returns, rf_annual=0.0):
    """Randamentul, volatilitatea și Sharpe anualizate ale unui portofoliu cu ponderi fixe, pe `returns`."""
    daily = returns.to_numpy() @ np.asarray(weights, dtype="float64")
    ret, vol = float(daily.mean() * TRADING_DAYS), float(daily.std(ddof=1) * TRADING_DAYS ** 0.5)
    return {"return": ret * 100, "volatility": vol * 100, "sharpe": (ret - rf_annual) / vol if vol > 0 else None}


def trim_short_history(closes, min_days):
    """Scoate simbolurile cu istoric prea scurt, dacă rămân cel puțin două cu istoric suficient.

    O poziție listată recent ar reduce tot calculul la câteva zeci de zile comune. Un simbol
    e „scurt” dacă are cel mult `min_days` prețuri. Dacă nu rămân două simboluri lungi, nu se
    scoate nimic: problema e a întregului set, nu a unui simbol. Întoarce (tabel, simboluri scoase).
    """
    counts = closes.count()
    short = [str(c) for c in counts[counts <= min_days].sort_values().index]
    if not short or (len(counts) - len(short)) < 2:
        return closes, []
    return closes.drop(columns=short), short


def optimize_portfolio(closes, rf_annual=0.0, max_weight=0.25, min_days=250, test_share=0.3):
    """Optimizare pe tot istoricul comun, plus verificare în afara eșantionului.

    - Randamentele se calculează doar pe zilele în care toate simbolurile au preț.
    - "allocation" (Sharpe maxim) și "min_variance" sunt în procente pe simbol.
    - "out_of_sample": ponderile găsite pe primele (1 − test_share) din zile, evaluate pe restul,
      alături de ponderile egale. Dacă pe datele nevăzute optimizarea nu bate ponderile egale,
      „optimul” e în mare parte potrivire pe trecut.
    Întoarce (rezultat, mesaj); rezultatul e None când datele nu ajung.
    """
    frame, excluded = trim_short_history(pd.DataFrame(closes), min_days)
    if frame.shape[1] < 2:
        return None, "Sunt necesare cel puțin 2 active cu istoric de preț pentru optimizare."
    returns = frame.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < min_days:
        return None, (f"Doar {len(returns)} zile comune de tranzacționare; sunt necesare cel puțin {min_days} "
                      "(un an) pentru o optimizare cu sens.")
    full = optimize_weights(returns, rf_annual, max_weight)
    if full is None:
        return None, "Optimizarea nu a convers pe aceste date."
    stats = weights_stats(full["max_sharpe"], returns, rf_annual)
    minvar_stats = weights_stats(full["min_variance"], returns, rf_annual)

    split = int(len(returns) * (1.0 - test_share))
    train, test = returns.iloc[:split], returns.iloc[split:]
    out_of_sample = []
    trained = optimize_weights(train, rf_annual, max_weight) if len(train) >= 120 and len(test) >= 60 else None
    if trained is not None:
        equal = np.full(returns.shape[1], 1.0 / returns.shape[1])
        for label, weights in (("Sharpe maxim", trained["max_sharpe"]), ("Varianță minimă", trained["min_variance"]), ("Ponderi egale", equal)):
            row = weights_stats(weights, test, rf_annual)
            row["label"] = label
            out_of_sample.append(row)

    result = {
        "allocation": (full["max_sharpe"] * 100).round(2).to_dict(),
        "min_variance": (full["min_variance"] * 100).round(2).to_dict(),
        "expected_return": stats["return"], "expected_volatility": stats["volatility"], "sharpe_ratio": stats["sharpe"],
        "min_variance_stats": minvar_stats, "cap": full["cap"] * 100, "shrinkage": full["shrinkage"],
        "n_days": int(len(returns)), "start": returns.index[0], "end": returns.index[-1],
        "out_of_sample": out_of_sample, "excluded": excluded,
        "test_start": test.index[0] if len(test) else None, "test_days": int(len(test)),
    }
    return result, "Optimizare finalizată."
