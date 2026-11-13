"""Textele „?” (help=) pentru cardurile din „Indicatori Fundamentali”: cum se calculează,
ce înseamnă și de unde vine valoarea afișată (Yahoo, foaia BVB sau situațiile financiare).

Funcție pură: primește dicționarul `info` deja completat de aplicație și întoarce text.
"""

RULE_NOTE = "Praguri orientative (reguli practice din literatură, nu standard oficial și netestate pe datele tale)"


TEST_NOTE_BVB = "Testat pe datele tale (BVB)"


def _h(ce, calcul, praguri=None, atentie=None, test=None, praguri_titlu=None):
    """Explicația în părți scurte, pe rânduri separate (se citește pe telefon).

    `praguri` = reguli practice (titlul spune că nu sunt testate), `test` = rezultatul testului istoric
    pe datele lui Daniel, `praguri_titlu` înlocuiește titlul pragurilor când ele sunt ale scorului
    (definite de aplicație, nu reguli din literatură)."""
    parts = [f"**Ce e:** {ce}", f"**Calcul:** {calcul}"]
    if praguri:
        parts.append(f"**{praguri_titlu or RULE_NOTE}:** {praguri}")
    if test:
        parts.append(f"**{TEST_NOTE_BVB}:** {test}")
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


# --- Analiză companie: Decizie BVB în doi pași (pragurile scorului și ale momentului sunt testate) ----

SCORE_BANDS = "Pragurile scorului (definite de aplicație)"

BVB_HELP = {
    "scor": _h("poziția companiei față de celelalte companii BVB din fișierele tale, pe evaluare și calitate.",
               "media percentilelor a patru indicatori: P/E, P/BV, randament dividend, Piotroski simplificat "
               "(100 = cea mai bună pe toți, 0 = cea mai slabă).",
               "treimea de sus FAVORABIL · mijlocul NEUTRU · treimea de jos NEFAVORABIL.",
               "e un scor relativ: „favorabil” înseamnă mai bun decât celelalte companii BVB urmărite, nu ieftin în absolut.",
               test="2017–2026: treimea de sus a bătut media BVB cu +6,0% la 6 luni (56% cazuri pozitive); "
                    "treimea de jos a rămas cu −7,0% în urmă (33% pozitive).", praguri_titlu=SCORE_BANDS),
    "pe": _h("câți ani de profit actual plătești pentru acțiune.",
             "capitalizare (preț curent × acțiuni din ultimul raport) / profit net pe ultimele 12 luni, din cifrele "
             "cumulate din fișier. Sub valoare: percentila față de celelalte companii BVB.",
             "sub 5 foarte ieftin · 5–10 ieftin · 10–15 corect · 15–25 scump · peste 25 foarte scump.",
             "sub 5 e adesea o capcană (profit nerepetabil, ciclu la vârf). La pierdere: N/A.",
             test="P/E mic a bătut P/E mare (IC +0,17)."),
    "pbv": _h("cât plătești pentru fiecare leu de capital propriu.",
              "capitalizare / capital propriu din ultimul raport.",
              "sub 1 sub valoarea contabilă · 1–1,5 corect (Graham) · 1,5–3 moderat · peste 3 scump.",
              "sub 1 poate însemna active supraevaluate în bilanț sau rentabilitate slabă.",
              test="P/BV mic a bătut P/BV mare (IC +0,13), mai slab decât P/E."),
    "dy": _h("randamentul dividendului la prețul de azi.",
             "profitul distribuit din ultimul an / capitalizarea de azi.",
             "sub 2% mic · 2–4% moderat · 4–7% ridicat · peste 7% foarte ridicat.",
             "peste 7%: verifică dacă dividendul se poate repeta (profit în scădere, dividend special).",
             test="dividendul mare a bătut dividendul mic (IC +0,13)."),
    "pio": _h("calitatea și direcția rezultatelor financiare.",
              "6 criterii, ultimul raport față de același raport de acum un an: ROA pozitiv, ROA în creștere, "
              "lichiditate curentă în creștere, datorii pe termen lung / active în scădere, fără emisiune de acțiuni, "
              "rotația activelor în creștere. Fără criteriile de flux de numerar (nu sunt în fișiere).",
              "0–2 slab · 3–4 mediu · 5–6 puternic (echivalentul benzilor 0–3 / 4–6 / 7–9 ale Piotroski complet).",
              test="cel mai puternic indicator din test (IC +0,28)."),
    "gn": _h("criteriul lui Graham care leagă prețul de profit și de activul net.",
             "P/E × P/BV.",
             "cel mult 22,5 trece criteriul (P/E 15 × P/BV 1,5) · peste 22,5 nu trece.",
             "nu intră în scorul BVB (e informație).",
             test="GN ≤ 22,5 → +2,9% față de media BVB la 6 luni (50% pozitive); GN > 22,5 → −3,6% (39% pozitive)."),
    "moment": _h("dacă tehnicul susține o intrare acum (eticheta de moment, separată de scorul fundamental).",
                 "media dintre trend (preț peste SMA 50 și SMA 200, SMA 50 peste SMA 200) și momentum 12-1 luni "
                 "scalat (−30% → 0, +30% → 100).",
                 "80–100 FAVORABIL · 40–80 NEUTRU · sub 40 NEFAVORABIL.",
                 "nu se adună cu scorul fundamental.",
                 test="2012–2025: ≥ 80 → +6,1% față de media BVB la 6 luni (54% pozitive); < 40 → −7,2% (32% pozitive). "
                      "RSI nu intră: în test „supravândut” a continuat să scadă.", praguri_titlu=SCORE_BANDS),
    "trend": _h("direcția trendului pe termen mediu și lung.",
                "câte din: preț > SMA 50, preț > SMA 200, SMA 50 > SMA 200 (sub 200 de ședințe: doar preț > SMA 50).",
                "3 din 3 trend ascendent clar · 2 mixt pozitiv · 1 mixt negativ · 0 trend descendent.",
                "e o componentă a momentului, nu un semnal separat."),
    "momentum": _h("cât a crescut acțiunea în ultimul an, fără ultima lună.",
                   "randamentul de acum 12 luni până acum o lună, pe date calendaristice (ultima lună exclusă, "
                   "pentru că pe termen scurt prețurile tind să se întoarcă).",
                   "peste +20% puternic · 0 – +20% pozitiv · −20% – 0 negativ · sub −20% slab.",
                   "e o componentă a momentului."),
    "rsi": _h("cât de repede a urcat sau a coborât prețul în ultimele 14 ședințe.",
              "RSI 14 cu netezire Wilder.",
              "sub 30 supravândut · 30–70 neutru · peste 70 supracumpărat.",
              "doar descriptiv: în trend puternic RSI rămâne mult peste 70 sau sub 30.",
              test="nu a dat semnal de intrare: „supravândut” a continuat să scadă."),
    "roe": _h("cât profit face compania din banii acționarilor.",
              "profit net pe 12 luni / capital propriu.",
              "sub 8% slab · 8–15% acceptabil · 15–20% bun · peste 20% excelent.",
              "iese mare și când capitalul propriu e mic (datorii mari).",
              test="nu a prezis randamentul (IC ≈ 0): informație, fără puncte în scor."),
    "marja": _h("cât rămâne profit din fiecare leu vândut.",
                "profit net / cifră de afaceri, pe 12 luni.",
                "sub 5% subțire · 5–10% medie · 10–20% bună · peste 20% foarte bună.",
                "depinde mult de sector.",
                test="nu a prezis randamentul: informație."),
    "indatorare": _h("câte datorii are compania față de capitalul propriu.",
                     "datorii TOTALE (inclusiv furnizori, provizioane, venituri în avans) / capital propriu, din ultimul raport. "
                     "Diferă de „Datorii/Capital” din Yahoo, care ia doar datoria financiară.",
                     "sub 0,5 conservator · 0,5–1 moderat · 1–2 ridicat · peste 2 foarte ridicat.",
                     "include și datoriile comerciale, deci iese mai mare decât datoria financiară.",
                     test="nu a prezis randamentul: informație de risc."),
}


# --- Analiză companie: scorul fundamental SUA/UE (descriptiv), risc, îndatorare -----------------

PILLAR_HELP = {
    "evaluare": _h("cât de ieftină e acțiunea față de ce produce compania.",
                   "media componentelor cu date. Nefinanciare: DCF invers (creșterea FCF pe care o cere prețul: ≤ 0% → 100, "
                   "≥ 20% → 0) și randamentul profitului E/P față de randamentul AAA (≤ 1× → 0, ≥ 2× → 100). "
                   "Bănci și asigurători: P/BV față de P/BV justificat (ROE − g) / (ke − g): raport ≤ 0,7 → 100, ≥ 1,3 → 0.",
                   "peste 67 bun · 33–67 mediu · sub 33 slab.",
                   "GN apare doar informativ.", praguri_titlu=SCORE_BANDS),
    "calitate": _h("calitatea și direcția rezultatelor financiare (Piotroski F-Score).",
                   "criterii îndeplinite / criterii evaluabile (minimum 6 din 9), ultimul an fiscal față de cel precedent.",
                   "peste 67 puternic (echivalent 7–9 din 9) · 33–67 mediu · sub 33 slab.", praguri_titlu=SCORE_BANDS),
    "sanatate": _h("riscul de dificultate financiară.",
                   "media: zona Altman (sigură 100, gri 50, dificultate 0), datorie netă / EBITDA (≤ 1 → 100, ≥ 4 → 0), "
                   "acoperirea dobânzii EBIT / dobânzi (≤ 1,5 → 0, ≥ 8 → 100).",
                   "peste 67 solidă · 33–67 medie · sub 33 fragilă.", praguri_titlu=SCORE_BANDS),
    "rentabilitate": _h("dacă rentabilitatea capitalului depășește costul lui.",
                        "ROE față de costul capitalului propriu ke = rf + beta × 5%: −5 pp → 0, +5 pp → 100.",
                        "peste 50 ROE peste costul capitalului (creează valoare) · sub 50 sub costul capitalului.",
                        praguri_titlu=SCORE_BANDS),
}
PILLAR_TEST_NOTE = ("\n\n**Test istoric SUA 2022–2026:** fără putere de predicție; scorul descrie compania, "
                    "pragurile sunt orientative.")

RISK_HELP = {
    "beta": _h("cât de mult se mișcă acțiunea când se mișcă piața ei.",
               "covarianța randamentelor acțiunii cu ale indicelui / varianța indicelui.",
               "sub 0,8 defensivă · 0,8–1,2 ca piața · peste 1,2 agresivă (amplifică mișcările pieței).",
               "beta mic nu înseamnă risc mic: acțiunea poate avea riscuri proprii mari."),
    "alpha": _h("cât a câștigat acțiunea peste ce justifica riscul ei de piață (alpha Jensen).",
                "randamentul acțiunii − [rf + beta × (randamentul indicelui − rf)], pe un an.",
                "peste 0 a bătut piața ajustat la risc · sub 0 a rămas în urmă.",
                "pe un singur an e mult zgomot; nu prezice anul următor."),
    "nd_ebitda": _h("în câți ani ar plăti compania datoria netă din EBITDA.",
                    "(datorie totală − numerar) / EBITDA, din situațiile financiare.",
                    "sub 0 numerar net · 0–1 foarte mică · 1–2 confortabilă · 2–3 moderată · 3–4 ridicată · peste 4 riscantă.",
                    "nu se aplică băncilor. N/A când EBITDA e negativ sau lipsește."),
    "int_cov": _h("de câte ori acoperă profitul operațional dobânzile.",
                  "EBIT / cheltuielile cu dobânzile raportate.",
                  "sub 1,5 risc · 1,5–3 slabă · 3–8 bună · peste 8 solidă.",
                  "la unele companii „dobânzile” raportate includ și alte costuri financiare."),
    "target": _h("media prețurilor țintă ale analiștilor, de obicei la 12 luni.",
                 "Yahoo `targetMeanPrice`.",
                 None,
                 "e opinia analiștilor, nu o evaluare a aplicației; ținta se corectează des după preț. La BVB lipsește des."),
}


# --- Analiză companie: opțiuni ---------------------------------------------------------------------

OPT_HELP = {
    "pc_oi": _h("raportul dintre pozițiile deschise pe put (protecție / pariu pe scădere) și pe call.",
                "Σ open interest put / Σ open interest call, pe expirarea aleasă.",
                "sub 0,7 optimism · 0,7–1,1 neutru · peste 1,1 pesimism sau protecție (aceleași praguri ca verdictul de mai jos).",
                "pe o singură acțiune e zgomotos și se citește adesea invers (contrarian). N/A = fără open interest la call-uri."),
    "pc_vol": _h("același raport, pe volumul tranzacționat azi.",
                 "Σ volum put / Σ volum call, pe expirarea aleasă, în ședința curentă.",
                 "sub 0,7 optimism · 0,7–1,1 neutru · peste 1,1 pesimism sau protecție.",
                 "se mișcă mai repede decât open interest-ul; în timpul ședinței e parțial. N/A = fără volum la call-uri."),
    "max_pain": _h("strike-ul la care opțiunile acestei expirări ar valora cel mai puțin la expirare.",
                   "pentru fiecare strike K: Σ OI call × max(K − strike, 0) + Σ OI put × max(strike − K, 0); Max Pain = K-ul "
                   "cu suma minimă.",
                   None, "putere de predicție slabă: e un reper, nu o țintă de preț."),
    "expirare": _h("expirarea pe care sunt calculați indicatorii.",
                   "expirarea cu 20–45 de zile rămase cea mai apropiată de 30.",
                   None, "expirările de câteva zile au IV deformat de timpul scurt."),
    "iv_rank": _h("unde e volatilitatea implicită de azi între minimul și maximul ultimului an.",
                  "(IV de azi − minim) / (maxim − minim) pe ultimul an, din istoricul IV la bani salvat zilnic (foaia IV_istoric).",
                  "sub 20 IV mică (opțiuni relativ ieftine) · 20–50 normală · 50–80 ridicată · peste 80 extremă (opțiuni scumpe).",
                  "cu istoric scurt e orientativ; sub numărul minim de zile: N/A."),
    "iv_pct": _h("în câte zile din ultimul an IV-ul a fost mai mic decât azi.",
                 "procentul zilelor din ultimul an cu IV la bani sub cel de azi (același istoric ca IV Rank).",
                 "sub 20 IV mică · 20–50 normală · 50–80 ridicată · peste 80 extremă.",
                 "mai robust decât IV Rank la un singur vârf extrem în istoric."),
    "iv_hv": _h("cât de multă mișcare așteaptă opțiunile față de mișcarea reală recentă.",
                "IV la bani / volatilitatea realizată pe ultimele 30 de zile calendaristice (randamente log, √252).",
                "sub 0,8 IV sub mișcarea reală (opțiuni ieftine) · 0,8–1,2 normal · peste 1,2 piața plătește primă mare.",
                "nu e IV Rank: compară cu trecutul recent al acțiunii, nu cu istoricul IV."),
    "delta": _h("cât se schimbă prețul opțiunii când acțiunea urcă cu 1 USD.",
                "Black-Scholes cu IV la bani, rata fără risc și randamentul dividendului din antet.",
                "call 0 – 1, put −1 – 0; la bani ≈ ±0,5. În valoare absolută ≈ probabilitatea aproximativă de a expira în bani.",
                "se schimbă cu prețul și cu timpul (vezi gamma)."),
    "gamma": _h("cât se schimbă delta când acțiunea urcă cu 1 USD.",
                "Black-Scholes: φ(d1) / (S × σ × √T); aceeași pentru call și put.",
                None, "e cea mai mare la bani și aproape de expirare: acolo poziția se schimbă cel mai repede."),
    "theta": _h("cât pierde opțiunea din valoare într-o zi, dacă restul rămâne neschimbat.",
                "theta anual Black-Scholes / 365, pe o acțiune suport (un contract = 100 de acțiuni).",
                None, "negativ pentru cumpărător (timpul lucrează împotriva lui), câștig pentru vânzător; crește spre expirare."),
    "vega": _h("cât se schimbă prețul opțiunii dacă volatilitatea implicită urcă cu 1 punct procentual.",
               "Black-Scholes; aceeași pentru call și put.",
               None, "contează mult înainte de rezultate financiare, când IV-ul scade brusc după anunț."),
    "expected_move": _h("intervalul de preț până la expirare pe care îl implică opțiunile.",
                        "preț × IV la bani × √(zile / 365) = ±1 abatere standard.",
                        "în aproximativ 68% din cazuri prețul ar rămâne în interval (sub ipoteza de normalitate).",
                        "distribuția normală subestimează mișcările extreme."),
}
