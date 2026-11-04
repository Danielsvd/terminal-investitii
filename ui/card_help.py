"""Textele „?” (help=) pentru cardurile din „Indicatori Fundamentali”: cum se calculează,
ce înseamnă și de unde vine valoarea afișată (Yahoo, foaia BVB sau situațiile financiare).

Funcție pură: primește dicționarul `info` deja completat de aplicație și întoarce text.
"""

FUND_HELP = {
    "trailingPE": ("Preț / profitul pe acțiune din ultimele 12 luni (EPS TTM). Arată câți ani de profit actual "
                   "plătești. Se compară doar cu companii din același sector; la pierderi nu are sens (N/A)."),
    "forwardPE": ("Preț / profitul pe acțiune estimat de analiști pentru următoarele 12 luni (consensul din Yahoo). "
                  "Sub P/E-ul curent înseamnă că analiștii așteaptă creșterea profitului. Estimare, nu rezultat raportat."),
    "dividendRate": ("Dividendul anual pe acțiune (ultimul declarat, anualizat) și randamentul = dividend / prețul curent. "
                     "Nu include impozitul reținut."),
    "priceToBook": ("Preț / valoarea contabilă pe acțiune (capital propriu / numărul de acțiuni). Sub 1 = acțiunea costă mai "
                    "puțin decât capitalul din bilanț. Când capitalul propriu e micșorat de răscumpărări sau pierderi "
                    "acumulate, P/BV devine foarte mare și nu mai măsoară evaluarea."),
    "gn": ("P/E × P/BV. Criteriul lui Graham cere cel mult 22,5. Informativ: la SUA/UE nu intră în scor "
           "(evaluarea folosește E/P față de AAA)."),
    "trailingEps": "Profitul net atribuibil acționarilor / numărul mediu de acțiuni, pe ultimele 12 luni.",
    "bookValue": "Capitalul propriu / numărul de acțiuni, din ultimul bilanț.",
    "returnOnAssets": "Profit net / active totale. Cât profit face compania din tot ce deține, indiferent cum e finanțată.",
    "returnOnEquity": ("Profit net / capital propriu. Cât profit pe banii acționarilor. Iese foarte mare când capitalul "
                       "propriu e mic (răscumpărări, datorii mari), deci se citește împreună cu Datorii/Capital."),
    "profitMargins": "Profit net / venituri, pe ultimele 12 luni. Cât rămâne din fiecare leu (dolar) vândut după toate cheltuielile.",
    "operatingMargins": "Profit operațional / venituri. Rentabilitatea activității de bază, înainte de dobânzi și impozit.",
    "debtToEquity": ("Datorie totală / capital propriu, în procente (71% = 0,71 lei datorie la 1 leu capital). Peste 100% "
                     "= mai multă datorie decât capital propriu; valori mari sunt obișnuite la utilități și la bănci."),
    "currentRatio": ("Active curente / datorii curente. Sub 1 = datoriile scadente într-un an depășesc activele care "
                     "devin bani într-un an."),
    "quickRatio": ("(Active curente − stocuri) / datorii curente. Ca Current Ratio, dar fără stocuri, care se transformă "
                   "mai greu în bani."),
}

# Indicatorii calculați din alte valori: sursa e a componentelor.
COMPONENTS = {"gn": ("trailingPE", "priceToBook")}


def value_source(info, key):
    """De unde vine valoarea afișată pentru `key` (o propoziție)."""
    info = info or {}
    keys = COMPONENTS.get(key, (key,))
    repriced = set(info.get("_bvb_repriced") or [])
    sheet = set(info.get("_from_bvb_sheet") or [])
    statements = set(info.get("_from_statements") or [])
    parts = []
    for k in keys:
        if k in repriced:
            parts.append("foaia BVB, recalculat la prețul curent")
        elif k in sheet:
            period = info.get("_bvb_period")
            parts.append(f"foaia BVB (raportare {period})" if period else "foaia BVB")
        elif k in statements:
            parts.append("calculat de aplicație din situațiile financiare (Yahoo nu a trimis valoarea)")
        else:
            parts.append("Yahoo Finance")
    unique = list(dict.fromkeys(parts))
    return "Sursă: " + "; ".join(unique) + "."


def fund_help(info, key):
    """Textul complet pentru „?”: definiția + sursa valorii."""
    return f"{FUND_HELP[key]} {value_source(info, key)}"
