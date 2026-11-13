"""Textele „?” (help=) pentru cardurile din „Indicatori Fundamentali”: cum se calculează,
ce înseamnă și de unde vine valoarea afișată (Yahoo, foaia BVB sau situațiile financiare).

Funcție pură: primește dicționarul `info` deja completat de aplicație și întoarce text.
"""

RULE_NOTE = "Praguri orientative (reguli practice din literatură, nu standard oficial și netestate pe datele tale)"


def _h(ce, calcul, praguri=None, atentie=None):
    """Explicația în patru părți scurte, pe rânduri separate (se citește pe telefon)."""
    parts = [f"**Ce e:** {ce}", f"**Calcul:** {calcul}"]
    if praguri:
        parts.append(f"**{RULE_NOTE}:** {praguri}")
    if atentie:
        parts.append(f"**Atenție:** {atentie}")
    return "\n\n".join(parts)


FUND_HELP = {
    "trailingPE": _h("câți ani de profit actual plătești pentru acțiune.",
                     "preț / profitul pe acțiune din ultimele 12 luni (EPS TTM).",
                     "sub 5 foarte ieftin · 5–10 ieftin · 10–15 corect (Graham: maximum 15) · 15–25 scump · peste 25 foarte scump.",
                     "sub 5 e adesea o capcană (profit nerepetabil, ciclu la vârf, risc mare); peste 25 poate fi normal la "
                     "creștere rapidă (vezi PEG). Se compară cu sectorul și cu dobânzile (E/P față de AAA). La pierdere: N/A."),
    "forwardPE": _h("P/E-ul pe profitul estimat de analiști pentru următoarele 12 luni.",
                    "preț / EPS estimat (consensul din Yahoo).",
                    "aceleași benzi ca P/E: sub 10 ieftin · 10–15 corect · 15–25 scump · peste 25 foarte scump. Forward P/E sub "
                    "P/E-ul curent = analiștii așteaptă creșterea profitului.",
                    "e o estimare, nu un rezultat raportat; estimările se revizuiesc des."),
    "peg_hist": _h("P/E raportat la ritmul de creștere a profitului (indicatorul lui Peter Lynch).",
                   "P/E ÷ creșterea anuală compusă a EPS pe ultimii 3 ani, în procente (P/E 20 și EPS +20%/an → PEG 1). "
                   "EPS din situațiile financiare raportate (la BVB: profit net / acțiuni din fișierul „Analiză companii”).",
                   "sub 1 ieftin față de creștere · 1–2 corect · peste 2 scump.",
                   "creșterea trecută nu garantează creșterea viitoare. N/A la pierdere, la EPS în scădere sau când numărul "
                   "de acțiuni s-a schimbat mult (split, majorare). Creșteri de peste 50%/an fac PEG-ul instabil."),
    "peg_analysts": _h("PEG-ul calculat de Yahoo cu creșterea estimată de analiști.",
                       "P/E ÷ creșterea anuală a EPS așteptată de analiști pe următorii ~5 ani (Yahoo, „trailingPegRatio”).",
                       "sub 1 ieftin față de creștere · 1–2 corect · peste 2 scump.",
                       "estimare, informativă; lipsește la majoritatea acțiunilor BVB și la multe europene."),
    "dividendRate": _h("dividendul anual pe acțiune și randamentul lui la prețul curent.",
                       "dividend anual / preț curent. Fără impozitul reținut.",
                       "sub 2% mic · 2–4% moderat · 4–7% ridicat · peste 7% foarte ridicat.",
                       "peste 7%: verifică dacă e sustenabil (cât din profit sau din FCF se distribuie, dividend special, "
                       "preț prăbușit)."),
    "priceToBook": _h("cât plătești pentru fiecare leu (dolar) de capital propriu din bilanț.",
                      "preț / valoarea contabilă pe acțiune (capital propriu / numărul de acțiuni).",
                      "sub 1 sub valoarea contabilă · 1–1,5 corect (Graham) · 1,5–3 moderat · peste 3 scump.",
                      "sub 1 poate însemna active supraevaluate în bilanț sau rentabilitate slabă (ROE). Când capitalul "
                      "propriu e micșorat de răscumpărări sau pierderi, P/BV devine foarte mare și nu mai măsoară "
                      "evaluarea. Băncile se judecă pe P/BV împreună cu ROE."),
    "gn": _h("criteriul lui Benjamin Graham care leagă prețul de profit și de activul net.",
             "P/E × P/BV.",
             "cel mult 22,5 trece criteriul (P/E 15 × P/BV 1,5) · 22,5–50 peste · peste 50 departe.",
             "Graham folosea profitul mediu pe 3 ani; aici e P/E pe 12 luni. Informativ la SUA/UE (evaluarea din scor "
             "folosește E/P față de AAA); neinformativ când P/BV e foarte mare."),
    "trailingEps": _h("profitul care revine unei acțiuni.",
                      "profit net atribuibil acționarilor / numărul mediu de acțiuni, pe ultimele 12 luni.",
                      None, "se urmărește evoluția în timp, nu valoarea absolută; negativ = pierdere."),
    "bookValue": _h("capitalul propriu care revine unei acțiuni.",
                    "capital propriu / numărul de acțiuni, din ultimul bilanț.",
                    None, "se compară cu prețul prin P/BV."),
    "returnOnAssets": _h("cât profit face compania din tot ce deține, indiferent cum e finanțată.",
                         "profit net / active totale.",
                         "sub 2% slab · 2–5% acceptabil · 5–10% bun · peste 10% excelent.",
                         "băncile au normal 1–1,5%; depinde de sector."),
    "returnOnEquity": _h("cât profit face compania din banii acționarilor.",
                         "profit net / capital propriu.",
                         "sub 8% slab (sub costul capitalului) · 8–15% acceptabil · 15–20% bun · peste 20% excelent.",
                         "iese foarte mare când capitalul propriu e mic (răscumpărări, datorii mari): citește-l împreună "
                         "cu Datorii/Capital. Negativ = pierdere."),
    "profitMargins": _h("cât rămâne profit din fiecare leu (dolar) vândut, după toate cheltuielile.",
                        "profit net / venituri, pe ultimele 12 luni.",
                        "sub 5% subțire · 5–10% medie · 10–20% bună · peste 20% foarte bună.",
                        "depinde mult de sector: în retail 2–4% e normal, în software 20%+ e obișnuit."),
    "operatingMargins": _h("rentabilitatea activității de bază, înainte de dobânzi și impozit.",
                           "profit operațional / venituri.",
                           "sub 5% slabă · 5–15% medie · 15–25% bună · peste 25% foarte bună.",
                           "depinde de sector; compară cu firmele din același domeniu."),
    "debtToEquity": _h("câtă datorie are compania față de capitalul acționarilor.",
                       "datorie totală / capital propriu, în procente (71% = 0,71 lei datorie la 1 leu capital).",
                       "sub 50% conservator · 50–100% moderat · 100–200% ridicat · peste 200% foarte ridicat.",
                       "utilitățile și băncile funcționează normal cu valori mari."),
    "currentRatio": _h("dacă activele care devin bani într-un an acoperă datoriile scadente într-un an.",
                       "active curente / datorii curente.",
                       "sub 1 risc de lichiditate · 1–1,5 strâns · 1,5–3 confortabil · peste 3 posibil numerar nefolosit.",
                       "nu se aplică băncilor."),
    "quickRatio": _h("ca Current Ratio, dar fără stocuri, care se transformă mai greu în bani.",
                     "(active curente − stocuri) / datorii curente.",
                     "sub 0,5 slab · 0,5–1 acceptabil · peste 1 bun.",
                     "nu se aplică băncilor."),
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
    return "**Sursă:** " + "; ".join(unique) + "."


def fund_help(info, key):
    """Textul complet pentru „?”: definiția + sursa valorii."""
    return f"{FUND_HELP[key]}\n\n{value_source(info, key)}"
