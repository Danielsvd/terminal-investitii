import pandas as pd
import numpy as np
import math

from analytics import scanner as scanner_calc
from analytics.scanner import MIN_AVG_SHARES, RVOL_DEFAULT, liquidity_ok


def test_prag_bvb_in_lei_ca_in_rezumat():
    assert liquidity_ok("TBM.RO", 0.31, 178_264, 99_839)            # 55.262 lei ≥ 50.000
    assert not liquidity_ok("RMAH.RO", 0.81, 48_043, 23_903)        # 38.915 lei < 50.000
    assert not liquidity_ok("SPX.RO", 0.29, 26_936, 6_744)          # 7.811 lei
    assert liquidity_ok("EBS.RO", 611.0, 500, 300)                  # 305.500 lei, deși sub 5.000 de acțiuni
    assert liquidity_ok("X.RO", 1.0, 50_000, 0)                     # exact pragul intră
    assert not liquidity_ok("X.RO", None, 50_000, 0) and not liquidity_ok("X.RO", 1.0, math.nan, 0)


def test_sua_si_europa_pastreaza_regula_in_actiuni():
    assert liquidity_ok("AAPL", 200.0, 1, MIN_AVG_SHARES)
    assert not liquidity_ok("SAP.DE", 200.0, 10**6, MIN_AVG_SHARES - 1)
    assert not liquidity_ok("AAPL", 200.0, 1, None)
    assert RVOL_DEFAULT == 1.2


def _frame(closes, vols):
    idx = pd.bdate_range("2026-09-01", periods=len(closes))
    return pd.DataFrame({"Close": closes, "Volume": vols}, index=idx)


def test_rvol_media_pe_20_de_sedinte_cu_volum_fara_ziua_evaluata():
    vols = [1000.0] * 20 + [3000.0]
    rs = scanner_calc.rvol_stats(_frame([10.0] * 20 + [10.5], vols))
    assert rs["avg_volume"] == 1000.0 and rs["rvol"] == 3.0 and rs["n_avg"] == 20
    assert abs(rs["change_pct"] - 5.0) < 1e-12


def test_rvol_zilele_cu_volum_zero_nu_intra_in_medie():
    # 21 de ședințe cu 1.000, intercalate cu 14 zile fără tranzacții (BVB): media rămâne 1.000, nu ~600
    vols = [1000.0, 0.0, 1000.0, 1000.0, 0.0] * 7 + [2000.0]
    rs = scanner_calc.rvol_stats(_frame([10.0] * 35 + [10.0], vols))
    assert rs["avg_volume"] == 1000.0 and rs["rvol"] == 2.0


def test_rvol_pret_si_volum_aliniate_pe_aceeasi_zi():
    # Ultima zi are volum dar fără preț: ședința evaluată e penultima (preț și volum din aceeași zi)
    closes = [10.0] * 20 + [11.0, np.nan]
    vols = [1000.0] * 20 + [1500.0, 9999.0]
    rs = scanner_calc.rvol_stats(_frame(closes, vols))
    assert rs["volume"] == 1500.0 and abs(rs["change_pct"] - 10.0) < 1e-12
    assert rs["date"] == pd.bdate_range("2026-09-01", periods=22)[20]


def test_rvol_fara_istoric_suficient_e_none():
    assert scanner_calc.rvol_stats(_frame([10.0] * 15, [1000.0] * 15)) is None
    assert scanner_calc.rvol_stats(None) is None
    assert scanner_calc.rvol_stats(pd.DataFrame({"Close": [1.0]})) is None


def test_rvol_sedinta_partiala_implicit_ultima_incheiata():
    # 21 de ședințe; ultima (în curs) are doar 400 din ~1.000: implicit se evaluează ședința de ieri (2.000 → RVOL 2)
    closes = [10.0] * 20 + [10.2, 10.1]
    vols = [1000.0] * 20 + [2000.0, 400.0]
    df = _frame(closes, vols)
    today = df.index[-1]
    is_partial = lambda d: pd.Timestamp(d) == today
    rs = scanner_calc.rvol_session(df, is_partial)
    assert rs["date"] == df.index[-2] and rs["rvol"] == 2.0 and rs["partial"] is False
    rs_live = scanner_calc.rvol_session(df, is_partial, include_partial=True)
    assert rs_live["date"] == today and rs_live["partial"] is True
    assert abs(rs_live["rvol"] - 400.0 / ((19 * 1000.0 + 2000.0) / 20)) < 1e-12
    # bursa închisă: ultima bară e completă
    rs_closed = scanner_calc.rvol_session(df, lambda d: False)
    assert rs_closed["date"] == today and rs_closed["partial"] is False


def test_hours_key():
    assert scanner_calc.hours_key("TLV.RO") == "BVB" and scanner_calc.hours_key("AAPL") == "NYSE"
    assert scanner_calc.hours_key("BRK-B") == "NYSE" and scanner_calc.hours_key("SAP.DE") == "XETRA"
    assert scanner_calc.hours_key("ULVR.L") == "XETRA"


def test_status_label_netestat():
    sl = scanner_calc.status_label
    assert sl(2.5, 2.0) == "🚀 BREAKOUT (netestat)"
    assert sl(2.5, -2.0) == "⚠️ PANIC SELL (netestat)"
    assert sl(2.5, 1.0) == "✅ ACUMULARE (netestat)"       # RVOL mare, mișcare mică pozitivă
    assert sl(2.0, 2.0) == "✅ ACUMULARE (netestat)"       # pragul 2 e strict
    assert sl(1.2, 0.5) == "Normal" and sl(1.5, -0.5) == "Normal"
    assert sl(None, 1.0) == "N/A" and sl(2.5, float("nan")) == "N/A"


def _anomaly_frame(last_vol, last_close, n=60, seed=1):
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.005, n))
    vols = rng.normal(1_000_000, 50_000, n)
    closes = np.append(closes, closes[-1] * last_close)
    vols = np.append(vols, last_vol)
    return _frame(list(closes), list(vols))


def test_anomalie_ziua_evaluata_nu_intra_in_antrenare():
    # volum de 10× și +8% într-o zi: anomalie; o zi obișnuită: nu
    assert scanner_calc.volume_anomaly(_anomaly_frame(10_000_000, 1.08)) is True
    assert scanner_calc.volume_anomaly(_anomaly_frame(1_000_000, 1.0)) is False


def test_anomalie_fara_istoric_e_none():
    df = _anomaly_frame(1_000_000, 1.0, n=15)
    assert scanner_calc.volume_anomaly(df) is None          # 10 zile de antrenare < 20
    assert scanner_calc.volume_anomaly(None) is None
    feats = scanner_calc.anomaly_features(_anomaly_frame(1_000_000, 1.0, n=30))
    assert list(feats.columns) == ["Return", "Volume", "Volatility"] and len(feats) == 31 - 5
