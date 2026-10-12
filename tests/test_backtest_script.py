"""Rularea cap-coadă a tools/backtest_scor.py pe date sintetice (fără rețea)."""
import numpy as np
import pandas as pd

from tools import backtest_scor


def _fake_download(symbol, start, end, refresh=False):
    if symbol == "LIPSA":
        return pd.DataFrame()
    seed = sum(map(ord, symbol))
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-11-01", "2021-12-31")
    adj = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx)))), index=idx)
    raw = adj * 1.05
    raw[raw.index >= "2018-06-15"] = adj[adj.index >= "2018-06-15"]   # un „dividend" pe 15 iun. 2018
    vol = pd.Series(1000.0, index=idx)
    return pd.DataFrame({"adj": adj, "raw": raw, "volume": vol})


def test_rulare_bvb_cu_simbol_lipsa(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(backtest_scor, "download", _fake_download)
    sym = tmp_path / "simboluri.txt"
    sym.write_text("# test\nAAA\nbbb.ro\nCCC\nDDD\nEEE\nFFF\nLIPSA\nAAA\n", encoding="utf-8")
    out = tmp_path / "rezultat.csv"
    code = backtest_scor.main(["--piata", "bvb", "--simboluri", str(sym), "--start", "2016-01-01",
                               "--end", "2019-12-31", "--out", str(out)])
    assert code == 0
    df = pd.read_csv(out)
    assert df["symbol"].nunique() == 6                     # AAA dublat o singură dată, LIPSA raportat
    assert df["date"].nunique() == 48                      # ian. 2016 – dec. 2019
    assert {"tech_score", "tech_score_unadj", "dividend_20d", "fwd_12m"} <= set(df.columns)
    assert df.loc[df["date"] == "2018-06-29", "dividend_20d"].all()
    rez = pd.read_csv(str(out).replace(".csv", "_rezumat.csv"))
    assert {"tech_score", "trend", "momentum", "rsi_c", "tech_score_unadj"} <= set(rez["scor"])
    # Rezumatul trebuie să fie exact tabelul calculat direct din observații (câmpurile IC nu suprascriu nimic).
    from analytics.backtest import bucket_summary
    direct = bucket_summary(df, "tech_score", "fwd_6m").reset_index()
    part = rez[(rez["scor"] == "tech_score") & (rez["orizont_luni"] == 6)].reset_index(drop=True)
    for col in ("N", "exces_mediu", "pct_pozitiv", "nr_date"):
        assert np.allclose(part[col].astype(float), direct[col].astype(float), equal_nan=True), col
    assert {"ic_mediu", "ic_t", "ic_nr_date", "ic_pct_pozitiv"} <= set(rez.columns)
    printed = capsys.readouterr().out
    assert "Fără date: LIPSA" in printed and "IC mediu" in printed


def test_citire_simboluri(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("aapl  # comentariu\n\nMSFT\n", encoding="utf-8")
    b.write_text("msft\nTLV.RO\n", encoding="utf-8")
    assert backtest_scor.read_symbols([a, b]) == ["AAPL", "MSFT", "TLV.RO"]
