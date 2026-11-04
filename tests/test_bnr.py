"""Cursurile BNR: parsare XML (cu multiplicator), unirea surselor, variațiile pe orizonturi."""
import numpy as np
import pandas as pd
import pytest

from data.bnr import FX_CURRENCIES, fx_table, merge_histories, parse_bnr_xml


def _rate(c, v, m):
    mult = ' multiplier="%s"' % m if m else ""
    return '<Rate currency="%s"%s>%s</Rate>' % (c, mult, v)


def _xml(cubes):
    body = "".join('<Cube date="%s">' % d + "".join(_rate(*r) for r in rates) + "</Cube>" for d, rates in cubes)
    return ('<?xml version="1.0" encoding="utf-8"?><DataSet xmlns="http://www.bnr.ro/xsd">'
            '<Header><Publisher>National Bank of Romania</Publisher><PublishingDate>2026-11-04</PublishingDate></Header>'
            f'<Body><Subject>Reference rates</Subject><OrigCurrency>RON</OrigCurrency>{body}</Body></DataSet>')


def test_parsare_cu_multiplicator_si_namespace():
    df = parse_bnr_xml(_xml([("2026-11-03", [("EUR", "5.0800", None), ("HUF", "1.3100", "100")]),
                             ("2026-11-04", [("EUR", "5.0866", None), ("HUF", "1.3122", "100"), ("USD", "4.4000", None)])]))
    assert list(df.index) == [pd.Timestamp("2026-11-03"), pd.Timestamp("2026-11-04")]
    assert df.loc["2026-11-04", "EUR"] == pytest.approx(5.0866)
    assert df.loc["2026-11-04", "HUF"] == pytest.approx(0.013122)       # 100 HUF = 1,3122 lei
    assert np.isnan(df.loc["2026-11-03", "USD"])                         # lipsă rămâne NaN


def test_parsare_invalida_sau_goala():
    assert parse_bnr_xml("").empty
    assert parse_bnr_xml("<nu-e-xml").empty
    assert parse_bnr_xml(_xml([("2026-11-04", [("EUR", "abc", None), ("USD", "0", None)])])).empty


def test_unire_ultima_sursa_castiga():
    a = parse_bnr_xml(_xml([("2026-11-03", [("EUR", "5.0", None)]), ("2026-11-04", [("EUR", "5.1", None)])]))
    b = parse_bnr_xml(_xml([("2026-11-04", [("EUR", "5.2", None)])]))
    m = merge_histories(a, None, pd.DataFrame(), b)
    assert m["EUR"].tolist() == [5.0, 5.2]
    assert merge_histories().empty


def _history():
    idx = pd.to_datetime(["2025-10-31", "2025-11-04", "2025-12-31", "2026-10-02", "2026-10-05", "2026-11-03", "2026-11-04"])
    return pd.DataFrame({"EUR": [4.98, 5.00, 5.05, 5.06, 5.07, 5.08, 5.10],
                         "HUF": [0.0130, 0.0131, 0.0132, 0.0133, 0.0134, 0.0135, 0.0136]}, index=idx)


def test_tabel_variatii():
    t = fx_table(_history(), [("EUR", "Euro", 1), ("HUF", "Forint", 100), ("USD", "Dolar", 1)]).set_index("Cod")
    eur = t.loc["EUR"]
    assert eur["Curs BNR"] == pytest.approx(5.10) and eur["Data"] == pd.Timestamp("2026-11-04")
    assert eur["Zi %"] == pytest.approx((5.10 / 5.08 - 1) * 100)
    # o lună: ultima publicare ≤ 04.10.2026 = 02.10 (5,06)
    assert eur["1 lună %"] == pytest.approx((5.10 / 5.06 - 1) * 100)
    # un an: ≤ 04.11.2025 = 04.11.2025 (5,00)
    assert eur["1 an %"] == pytest.approx((5.10 / 5.00 - 1) * 100)
    # de la 1 ianuarie: ultimul curs din 2025 = 31.12.2025 (5,05)
    assert eur["De la 1 ian. %"] == pytest.approx((5.10 / 5.05 - 1) * 100)
    assert t.loc["HUF", "Curs BNR"] == pytest.approx(1.36)              # afișat pe 100 HUF
    usd = t.loc["USD"]
    assert pd.isna(usd["Curs BNR"]) and pd.isna(usd["Zi %"])             # lipsă = N/A, nu 0


def test_orizont_neacoperit_e_na():
    h = _history().loc["2026-10-05":]
    eur = fx_table(h, [("EUR", "Euro", 1)]).iloc[0]
    # istoricul începe la 05.10.2026, după limita de o lună (04.10.2026): N/A, nu variația față de 05.10
    assert pd.isna(eur["1 lună %"])
    assert pd.isna(eur["1 an %"]) and pd.isna(eur["De la 1 ian. %"])
    assert not pd.isna(eur["Zi %"])


def test_lista_valute_unica():
    codes = [c for c, _, _ in FX_CURRENCIES]
    assert len(codes) == len(set(codes)) and codes[:2] == ["EUR", "USD"]
