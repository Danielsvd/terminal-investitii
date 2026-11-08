"""Filtrul știrilor pe emitent (data/news.py). Titluri sintetice, inspirate din cazurile reale."""
from data.news import company_terms, filter_company_news, mentions_company


def test_termeni_sap():
    words, acronyms = company_terms("SAP.DE", ("SAP SE",))
    assert acronyms == {"SAP"} and words == set()


def test_termeni_petrom_fara_cuvinte_generice():
    words, acronyms = company_terms("SNP.RO", ("OMV Petrom S.A.",))
    assert words == {"petrom", "omv petrom"} and acronyms == {"SNP", "OMV"}


def test_termeni_banca_transilvania():
    words, _ = company_terms("TLV.RO", ("Banca Transilvania S.A.",))
    assert words == {"transilvania", "banca transilvania"}        # „Banca” singur e generic


def test_termeni_apple():
    words, acronyms = company_terms("AAPL", ("Apple Inc.",))
    assert words == {"apple"} and acronyms == {"AAPL"}


def test_simbol_scurt_numele_intreg():
    words, acronyms = company_terms("T", ("AT&T Inc.",))
    assert acronyms == set() and words == {"at&t"}   # „T” și „AT” sub 3 litere; rămâne numele întreg
    assert mentions_company("AT&T adds 500k subscribers", (words, acronyms))
    assert not mentions_company("Telecom stocks rise", (words, acronyms))


def test_acronimul_se_compara_cu_majuscule():
    terms = company_terms("ALL", ())
    assert not mentions_company("Stocks rally all day", terms)
    assert mentions_company("Allstate (ALL) beats estimates", terms)


def test_cuvant_intreg():
    terms = company_terms("AAPL", ("Apple Inc.",))
    assert mentions_company("Apple unveils new iPhone", terms)
    assert not mentions_company("Pineapple prices surge", terms)


def test_filtru_sap_scoate_ibm():
    items = [
        {"title": "IBM and ServiceNow expand partnership", "summary": "Cloud deal."},
        {"title": "SAP raises cloud outlook", "summary": ""},
        {"title": "European tech stocks fall", "summary": "SAP shares dropped 2%."},
    ]
    kept = filter_company_news(items, "SAP.DE", ("SAP SE",))
    assert [k["title"] for k in kept] == ["SAP raises cloud outlook", "European tech stocks fall"]


def test_filtru_snp_scoate_bancile_franceze():
    items = [{"title": "French Bank Stocks slide on budget worries"},
             {"title": "OMV Petrom Q3 profit beats forecast"}]
    kept = filter_company_news(items, "SNP.RO", ("OMV Petrom S.A.",))
    assert [k["title"] for k in kept] == ["OMV Petrom Q3 profit beats forecast"]


def test_fara_termeni_lista_goala():
    assert filter_company_news([{"title": "Markets rally"}], "T", ()) == []


def test_fara_stiri():
    assert filter_company_news([], "AAPL", ("Apple Inc.",)) == []
    assert filter_company_news(None, "AAPL", ("Apple Inc.",)) == []


# --- Rezumatul Zilei D: știrile pe pozițiile din registru ----------------------------------------------------------
from datetime import datetime

from data.news import holdings_news, news_time


def test_stiri_pe_pozitii_filtrate_recente_fara_dubluri():
    now = datetime(2026, 11, 5, 18, 0)
    general = [
        {"title": "Banca Transilvania raportează profit record", "link": "a", "source": "ZF", "date_str": "2026-11-05 09:00"},
        {"title": "BET închide pe plus", "link": "b", "source": "Bursa", "date_str": "2026-11-05 17:50"},   # nu menționează
        {"title": "TLV anunță dividende", "link": "c", "source": "Profit", "date_str": "2026-11-01 10:00"},  # prea veche
        {"title": "OMV Petrom investește în gaze", "link": "d", "source": "ZF", "date_str": "2026-11-04 20:00"},
        {"title": "Petrom: fără dată", "link": "e", "source": "ZF", "date_str": "n/a"},                     # fără dată
    ]
    own = {"AAPL": [{"title": "Apple unveils new chip", "link": "f", "publisher": "Yahoo Finance",
                     "date_str": "2026-11-05 15:30"}]}
    holdings = {"TLV.RO": ("Banca Transilvania S.A.",), "SNP.RO": ("OMV Petrom S.A.",), "AAPL": ("Apple Inc.",)}
    out = holdings_news(holdings, general, own, now=now)
    assert [(x["symbol"], x["link"]) for x in out] == [("AAPL", "f"), ("TLV.RO", "a"), ("SNP.RO", "d")]
    assert out[0]["source"] == "Yahoo Finance" and out[0]["time"] == datetime(2026, 11, 5, 15, 30)


def test_stiri_limita_pe_simbol_si_aceeasi_stire_o_data():
    now = datetime(2026, 11, 5, 18, 0)
    general = [{"title": f"Apple news {i}", "link": str(i), "date_str": f"2026-11-05 1{i}:00"} for i in range(5)]
    general.append({"title": "Apple and Microsoft sign deal", "link": "x", "date_str": "2026-11-05 16:30"})
    out = holdings_news({"AAPL": ("Apple Inc.",), "MSFT": ("Microsoft Corporation",)}, general, now=now, per_symbol=3)
    aapl = [x for x in out if x["symbol"] == "AAPL"]
    msft = [x for x in out if x["symbol"] == "MSFT"]
    assert len(aapl) == 3 and [x["link"] for x in aapl] == ["x", "4", "3"]   # cele mai noi 3
    assert msft == []                                                           # știrea comună: o singură dată, la AAPL
    assert news_time({"date_str": "bad"}) is None


def test_termeni_stricti_fara_cuvinte_generice_din_nume():
    from data.news import strict_terms
    now = datetime(2026, 11, 5, 21, 0)
    assert strict_terms("BAC", ("Bank of America Corporation",)) == ({"bank of america"}, {"BAC"})
    assert strict_terms("NVO", ("Novo Nordisk A/S",)) == ({"novo nordisk"}, {"NVO"})
    assert strict_terms("CRWV", ("CoreWeave, Inc.",)) == ({"coreweave"}, {"CRWV"})
    assert strict_terms("SNP.RO", ("OMV PETROM SA",)) == ({"omv petrom"}, {"SNP"})
    general = [
        {"title": "eToro se extinde în America Latină prin achiziția fintech-ului argentinian LB Finanzas", "date_str": "2026-11-05 19:47"},
        {"title": "After US$474m gamble flops, what happens now to Elon Musk's Tesla?", "summary": "The first time ...",
         "date_str": "2026-11-05 17:00"},
        {"title": "ETFs Are Net Sellers of Bank of America (BAC) on Nov. 3", "date_str": "2026-11-05 13:02"},
        {"title": "First Majestic Silver raises output", "date_str": "2026-11-05 12:00"},
    ]
    out = holdings_news({"BAC": ("Bank of America Corporation",), "AG": ("First Majestic Silver Corp.",)}, general, now=now)
    assert [(x["symbol"], x["title"][:10]) for x in out] == [("BAC", "ETFs Are N"), ("AG", "First Maje")]


def test_stiri_cu_data_in_viitor_marcate_si_puse_la_coada():
    from data.news import future_note, mark_future
    now = datetime(2026, 11, 7, 0, 53)
    items = [
        {"title": "Bursa: ediția de mâine", "source": "Bursa", "date_str": "2026-11-07 18:47"},   # +18 ore
        {"title": "Bursa: alta", "source": "Bursa", "date_str": "2026-11-07 17:30"},
        {"title": "ZF: reală", "source": "ZF", "date_str": "2026-11-07 00:40"},
        {"title": "Profit: decalaj de 10 minute", "source": "Profit", "date_str": "2026-11-07 01:03"},  # sub toleranță
        {"title": "Economica: ieri", "source": "Economica", "date_str": "2026-11-06 22:00"},
    ]
    out = mark_future(items, now)
    assert [x["title"][:7] for x in out] == ["Profit:", "ZF: rea", "Economi", "Bursa: ", "Bursa: "]
    assert [x["future"] for x in out] == [False, False, False, True, True]
    assert out[3]["date_str"] == "2026-11-07 18:47"            # data din sursă nu se schimbă
    assert future_note(out[3]) == " ⚠️ dată din sursă în viitor" and future_note(out[0]) == ""
    assert "future" not in items[0]                              # lista primită nu se modifică
