"""Secțiunea „Glosar economic”: teorie macroeconomică și reguli de investiții, pe capitole.

Conținut static (fără rețea). Surse: documentele lui Daniel „Minighid de Macroeconomie” și „Hartă în
investiții” (capitolele marcate „din notițele tale”), completate cu teoria uzuală de piață. Corelațiile
sunt tendințe istorice, nu legi: se schimbă cu regimul economic, iar textul o spune unde contează.
"""
import unicodedata

INTRO = (
    "Teorie pe scurt, pentru deciziile de portofoliu. Corelațiile de mai jos sunt **tendințe istorice, nu legi**: "
    "se pot inversa luni sau ani la rând când se schimbă regimul (inflație ridicată, crize, politici ale băncilor "
    "centrale). Folosește-le ca întrebări de verificat în date, nu ca semnale de tranzacționare."
)

CHAPTERS = [
    {
        "title": "Cum citești o corelație",
        "icon": "🧭",
        "sections": [
            ("Ce înseamnă corelația", """
- **Corelația** (−1 … +1) arată cât de des două active se mișcă în aceeași direcție. +1 = mereu împreună, −1 = mereu invers, 0 = fără legătură liniară.
- Se calculează pe **randamente** (variații), nu pe prețuri: două prețuri care cresc amândouă în timp par corelate chiar dacă nu au legătură.
- Peste ~0,7 în valoare absolută = legătură puternică; 0,3–0,7 moderată; sub 0,3 slabă.
"""),
            ("Ce NU înseamnă", """
- **Corelația nu e cauzalitate**: aurul și dolarul nu se „mișcă unul pe altul”, ci reacționează amândouă la aceiași factori (dobânzi reale, frică).
- **Nu e stabilă**: corelația acțiuni–obligațiuni a fost negativă aproape tot intervalul 2000–2021 și a devenit pozitivă în 2022, când inflația a lovit ambele.
- **În crize, corelațiile dintre activele riscante urcă spre 1**: diversificarea între acțiuni ajută cel mai puțin exact când ai mai mare nevoie de ea.
"""),
        ],
    },
    {
        "title": "Dobânzile și băncile centrale",
        "icon": "🏦",
        "sections": [
            ("Dobânda de politică monetară", """
- **Fed** (SUA), **BCE** (zona euro) și **BNR** (România) stabilesc dobânda la care se împrumută băncile pe termen foarte scurt. De ea depind dobânzile la credite, la depozite și randamentele titlurilor de stat.
- **Dobânzi în creștere** → creditul se scumpește, economia încetinește, inflația scade, acțiunile scad de obicei (mai ales cele de creștere), dolarul / moneda locală se întărește.
- **Dobânzi în scădere** → efectul invers. Piața reacționează la **așteptări**, nu la decizia în sine: o reducere deja anticipată mișcă puțin prețurile.
- **QE / QT** (relaxare / înăsprire cantitativă): banca centrală cumpără sau lasă să expire obligațiuni, adăugând sau retrăgând lichiditate din piață.
"""),
            ("Prețul obligațiunilor și randamentul", """
- **Prețul unei obligațiuni scade când randamentul crește**, și invers. Sensibilitatea se numește **duration**: o obligațiune cu duration 7 pierde ~7% din preț la +1 punct procentual de randament.
- Obligațiunile lungi (10–30 de ani) sunt mult mai sensibile la dobânzi decât cele scurte. Titlurile de stat ținute până la scadență nu pierd nominal, dar pierd cost de oportunitate.
"""),
            ("Curba randamentelor", """
- Normal, randamentul pe 10 ani e **peste** cel pe 3 luni sau 2 ani (îți iei banii înapoi mai târziu, ceri mai mult).
- **Curba inversată** (10 ani − 3 luni < 0) a precedat aproape toate recesiunile din SUA din ultimii 50 de ani, cu un decalaj de 6–24 de luni. Inversarea lungă din 2022–2024 nu a fost urmată (încă) de recesiune: semnalul nu e infailibil. Semnalul de alarmă vine adesea la **re-pantare** (curba redevine pozitivă), când banca centrală începe să taie dobânzile.
- În aplicație: Piață globală → curba randamentelor.
"""),
            ("Dobânzile și evaluarea acțiunilor", """
- Valoarea unei acțiuni = profiturile viitoare actualizate cu o rată de scont. **Dobânda mai mare = rată de scont mai mare = valoare de azi mai mică**.
- Efectul e cel mai puternic la companiile cu profituri departe în viitor (tehnologie, creștere rapidă) și mai slab la cele cu profit stabil azi (value, dividende).
- Comparația simplă: **randamentul profiturilor (E/P = 1 / P/E)** față de randamentul titlurilor de stat. Dacă un titlu de stat dă mai mult decât E/P-ul pieței, prima pentru riscul acțiunilor e mică. În aplicație: Piață globală → prima de risc.
"""),
        ],
    },
    {
        "title": "Inflația",
        "icon": "📈",
        "sections": [
            ("Cum se măsoară", """
- **CPI** (indicele prețurilor de consum) an la an = cât s-au scumpit prețurile față de aceeași lună de anul trecut. **Core CPI** exclude energia și alimentele (mai stabil, urmărit de bănci).
- În SUA, Fed țintește **PCE core** ~2%. BCE țintește 2%, BNR 2,5% ± 1 pp.
- **Dobânda reală** = dobânda nominală − inflația. Contează pentru aur, pentru economisire și pentru evaluarea acțiunilor.
"""),
            ("Inflația și activele", """
- **Inflație moderată și stabilă (2–3%)**: bună pentru acțiuni (companiile își cresc prețurile).
- **Inflație ridicată sau în accelerare**: rău pentru obligațiunile cu cupon fix și pentru acțiunile scumpe; bine pentru mărfuri, energie, companii cu putere de preț. Din notițele tale: în perioade de inflație, mărfurile (inclusiv aurul) tind să se aprecieze.
- **Stagflația** (inflație + creștere slabă): cel mai greu regim pentru portofoliile clasice acțiuni + obligațiuni (vezi 1970 și 2022).
- **Deflația / recesiunea**: câștigă titlurile de stat și monedele de refugiu; mărfurile și aurul pot scădea temporar.
"""),
        ],
    },
    {
        "title": "Aurul",
        "icon": "🥇",
        "sections": [
            ("Rolul aurului (din notițele tale)", """
- **Activ de refugiu** în incertitudine economică, inflație sau crize geopolitice. O creștere rapidă a aurului indică adesea temeri economice sau financiare.
- **Corelație negativă cu dolarul**: când dolarul slăbește, aurul crește (e cotat în USD; devine mai ieftin pentru restul lumii).
- Protecție împotriva pierderii puterii de cumpărare când inflația e ridicată.
- 1 uncie troy = **31,1 grame**.
"""),
            ("Ce mai mișcă aurul", """
- **Dobânzile reale**: aurul nu plătește dobândă sau dividend, deci costul de a-l ține crește când dobânzile reale cresc. Istoric, aurul a avut o corelație negativă puternică cu randamentul real pe 10 ani al SUA (TIPS).
- Din 2022, legătura cu dobânzile reale s-a slăbit: **băncile centrale** (China, Polonia, India, Turcia) cumpără aur în cantități record pentru a-și diversifica rezervele, iar aurul a crescut chiar cu dobânzi reale mari.
- În **panica de lichiditate** (martie 2020, octombrie 2008) aurul poate scădea câteva zile, pentru că se vinde ce se poate vinde ca să acoperi pierderi în altă parte.
- Raportul **aur / argint** (de obicei 60–90): argintul e și metal industrial, deci crește mai mult în expansiuni și scade mai mult în recesiuni.
- În portofoliu: PPFB.DE / EGLN (ETC pe aur fizic) urmărește prețul aurului în EUR, deci câștigul tău depinde și de cursul EUR/USD.
"""),
        ],
    },
    {
        "title": "Petrolul, gazele și celelalte mărfuri",
        "icon": "🛢️",
        "sections": [
            ("Petrolul (din notițele tale)", """
- Resursă strategică: influențează costurile de producție și transport, deci **inflația** și **politicile monetare**.
- **Corelație în general inversă cu dolarul**: un dolar slab face petrolul mai ieftin în alte valute și stimulează cererea; un dolar puternic o reduce. Când investitorii fug în USD într-o criză, petrolul scade de obicei, pentru că scade și cererea globală.
"""),
            ("Ce mai trebuie știut despre petrol", """
- Din 2020 SUA e **exportator net** de petrol și produse petroliere: relația petrol–dolar e mai slabă și uneori pozitivă (petrol scump → mai multe dolari intră în SUA).
- **OPEC+** controlează o mare parte din ofertă: deciziile de reducere a producției mișcă prețul mai mult decât datele economice.
- Monede legate de petrol: **CAD** (Canada), **NOK** (Norvegia). Petrol în creștere → aceste monede se întăresc.
- Câștigători la petrol scump: companiile de energie (la BVB: OMV Petrom, Romgaz). Perdanți: companiile aeriene, chimia, transportul, consumatorii.
- Futures: **contango** (contractele viitoare mai scumpe) = ofertă abundentă; **backwardation** (viitoarele mai ieftine) = ofertă strânsă acum. ETF-urile pe petrol pierd bani în contango la fiecare rostogolire a contractelor.
"""),
            ("Gaze, metale industriale, agricole (din notițele tale)", """
- **Gaze naturale**: legate de sezon (cerere mai mare iarna) și de geopolitică; afectează balanțele comerciale ale exportatorilor și importatorilor. Prețul european (TTF) diferă mult de cel american (Henry Hub).
- **Cupru, aluminiu**: indică sănătatea economiei globale (construcții, producție, electrificare). Cuprul e numit „Dr. Copper”: crește înaintea expansiunilor, scade înaintea recesiunilor.
- **Grâne și produse agricole** (grâu, porumb, soia): depind de climă, cererea globală și politicile comerciale.
"""),
        ],
    },
    {
        "title": "Valute și monede de refugiu",
        "icon": "💱",
        "sections": [
            ("Monedele de refugiu (din notițele tale)", """
- **USD, CHF, JPY**: economii stabile, rezerve mari. În crize, investitorii se mută spre ele și le apreciază.
- Dolarul are adesea o relație inversă cu petrolul și aurul. Yenul și francul tind să se aprecieze în volatilitatea globală.
"""),
            ("Ce mișcă cursul pe termen mediu", """
- **Diferența de dobânzi**: moneda țării cu dobânzi mai mari tinde să atragă capital (strategia **carry trade**: împrumuți în yeni ieftini, plasezi în dolari mai scumpi). Când frica crește, carry trade-urile se închid brusc și yenul urcă puternic.
- **Dolar „smile”**: dolarul se întărește în două situații — când economia SUA merge mult mai bine decât restul lumii și când lumea e în criză (refugiu). Slăbește când creșterea globală e largă și sincronizată.
- **Dolar puternic = presiune pe piețele emergente** (datorii în USD mai scumpe, capital care pleacă). Dolar slab = vânt din spate pentru emergente și mărfuri.
- **AUD, NZD, CAD, NOK** sunt monede „de risc” / de mărfuri: cresc când apetitul pentru risc și mărfurile cresc.
"""),
            ("Leul și investitorul din România", """
- **EUR/RON** e ținut stabil de BNR (flotare administrată): mișcări mici, cu o depreciere lentă a leului pe termen lung. **USD/RON** urmează practic EUR/USD.
- Randamentul tău în lei = (1 + randamentul în valută) × (1 + variația cursului) − 1. O acțiune americană +10% cu dolarul −8% față de leu îți aduce ~+1,2% în lei. În aplicație: Piață globală → randamentul în lei.
- Dacă ai cheltuieli în lei, expunerea mare la USD adaugă un risc pe care acțiunile nu-l compensează. Acoperirea (hedging) costă diferența de dobânzi dintre cele două monede.
"""),
        ],
    },
    {
        "title": "Ciclul economic și rotația sectorială",
        "icon": "🔄",
        "sections": [
            ("Fazele ciclului", """
| Fază | Ce se vede | Sectoare care merg de obicei bine |
|---|---|---|
| Revenire | dobânzi jos, PMI urcă de sub 50 | financiar, consum discreționar, imobiliar, companii mici |
| Expansiune | PMI peste 50, profituri în creștere | tehnologie, industrie, materiale |
| Vârf / încetinire | inflație sus, băncile urcă dobânzile, curba se aplatizează | energie, materiale, sănătate |
| Contracție | PMI sub 50, șomaj în creștere | utilități, consum de bază, sănătate (defensive), titluri de stat |
"""),
            ("Semnale de urmărit", """
- **PMI / ISM** peste 50 = activitate în expansiune; sub 50 = contracție. Direcția contează mai mult decât nivelul.
- **Regula Sahm**: media pe 3 luni a șomajului urcă cu 0,5 pp peste minimul ultimelor 12 luni → semnalul a însoțit toate recesiunile din SUA din 1970 încoace; în 2024 a dat și o alarmă fără recesiune.
- **Spread-urile de credit** cresc înaintea și în timpul recesiunilor.
- **Câștigurile pe acțiune** ale indicilor scad de obicei în recesiune cu 15–30%; piața le anticipează cu 6–9 luni.
- În aplicație: Piață globală → harta sectoarelor (Money Flow) și barometrul economiei reale.
"""),
        ],
    },
    {
        "title": "Risc, frică și sentiment",
        "icon": "🌡️",
        "sections": [
            ("Ce se întâmplă într-o zi de „risk-off”", """
| Activ | Direcție tipică |
|---|---|
| Acțiuni (mai ales cele mici și emergente) | ↓ |
| VIX | ↑ |
| Titluri de stat SUA / Germania | ↑ prețul (randamentul ↓) |
| USD, JPY, CHF | ↑ |
| Aur | ↑ de obicei (↓ în panica de lichiditate) |
| Petrol, cupru | ↓ |
| Spread-uri de credit | ↑ |
| Monede emergente (inclusiv leul față de USD) | ↓ |
"""),
            ("Indicatorii de sentiment", """
- **VIX**: sub 15 calm, 15–20 normal, 20–30 tensiune, peste 30 frică. Vârfurile extreme (peste 40) au coincis istoric cu zone bune de cumpărare pe termen lung, dar nu se știe dinainte unde e vârful.
- **Put/Call**: valori foarte mari arată frică (adesea citit contrarian).
- **Lățimea pieței**: o creștere dusă de 5–10 giganți e mai fragilă decât una în care cresc majoritatea companiilor.
- Din notițele tale: uneori e avantajos să cumperi în pesimism generalizat, când acțiunile se vând sub valoarea reală; piețele reacționează exagerat la știri negative pe termen scurt.
"""),
        ],
    },
    {
        "title": "Corelații tipice — tabel de buzunar",
        "icon": "🧮",
        "sections": [
            ("Tabelul", """
| Pereche | Corelație tipică | Când se strică |
|---|---|---|
| Aur ↔ dolar (DXY) | negativă | crize de lichiditate, cumpărări masive ale băncilor centrale |
| Aur ↔ dobânda reală SUA | negativă puternică | 2022–2024 (cumpărări ale băncilor centrale) |
| Petrol ↔ dolar | negativă | din 2020 (SUA exportator net) e mai slabă |
| Petrol ↔ CAD, NOK | pozitivă | rar |
| Petrol ↔ inflație (cu decalaj de 1–3 luni) | pozitivă | — |
| Acțiuni ↔ titluri de stat | negativă (2000–2021) | pozitivă când inflația e problema principală (1970, 2022) |
| Acțiuni ↔ VIX | negativă puternică | aproape niciodată |
| Acțiuni ↔ spread-uri de credit | negativă | — |
| Cupru ↔ creșterea economică / acțiuni | pozitivă | șocuri de ofertă (mine închise) |
| Piețe emergente ↔ dolar | negativă | — |
| JPY, CHF ↔ acțiuni | negativă (refugiu) | perioade de carry trade stabil |
| BET ↔ S&P 500 | moderată (piață de frontieră/emergentă, lichiditate mică) | crize globale: urcă |
"""),
        ],
    },
    {
        "title": "Evaluarea pieței și a companiilor",
        "icon": "⚖️",
        "sections": [
            ("Multiplii și reperele lor", """
- **P/E** — câți ani de profit plătești. Graham: maximum 15. **P/BV** — maximum 1,5. **GN = P/E × P/BV** — maximum 22,5.
- **PEG = P/E ÷ creșterea profitului (%)** — sub 1 ieftin față de creștere (Peter Lynch).
- **CAPE (Shiller P/E)** = prețul ÷ media profiturilor reale pe 10 ani; istoric, valori mari au fost urmate de randamente mai mici în următorii 10 ani (nu în următorul an).
- **E/P față de titlurile de stat** (prima de risc a acțiunilor): cu cât e mai mică, cu atât acțiunile sunt mai scumpe față de obligațiuni.
- **DCF** — valoarea intrinsecă din fluxurile de numerar viitoare; sensibil la ipoteze, deci se citește cu tabelul de sensibilitate.
- În aplicație: Analiză companie → Indicatori fundamentali (fiecare „?” are pragurile).
"""),
        ],
    },
    {
        "title": "Harta ta în investiții — când cumperi",
        "icon": "🗺️",
        "sections": [
            ("Principii (din notițele tale)", """
1. **Analiza fundamentală**: model de afaceri solid, venituri și profituri în creștere, avantaj competitiv, conducere bună.
2. **Prețul față de valoare**: cumperi sub valoarea intrinsecă (DCF, multipli).
3. **Sentimentul pieței**: pesimismul generalizat poate da prețuri sub valoarea reală.
4. **Subevaluare** față de fundamentale: P/E, valoare contabilă, creșterea veniturilor și a profiturilor.
5. **Trenduri și tehnic** pentru momentul intrării (nu infailibil).
6. **Anunțuri pozitive**: produse noi, contracte mari.
7. **Reacții exagerate la știri negative**: oportunități pentru termen lung, dacă fundamentele rămân.
8. **Corecții de piață** de 10–20%, cu fundamente solide.
9. **Condițiile macro** (dobânzi, inflație) pot crea oportunități pe sectoare întregi.
10. **Achiziții periodice** (cost mediu în timp) reduc riscul de a intra într-un singur moment prost.
11. **Evită timing-ul pieței** pe termen scurt.
12. Investiția minimă: **1.000 lei / ordin** (comisioanele fixe cântăresc mult la ordine mici).
"""),
            ("Criteriile tale minime de cumpărare", """
| Criteriu | Prag | Unde îl vezi în aplicație |
|---|---|---|
| Marjă de profit | minimum 10% | Indicatori fundamentali → Marjă netă |
| Cifra de afaceri | în creștere | grafic Venit / Venit net |
| Profit net | în creștere | grafic Venit / Venit net |
| Grad de îndatorare | maximum 70% | Decizie BVB → Datorii totale / capital |
| P/E | maximum 15 | Indicatori fundamentali / Decizie BVB |
| P/BV | maximum 1,5 | Indicatori fundamentali / Decizie BVB |
| GN (P/E × P/BV) | maximum 22,5 | Indicatori fundamentali / Decizie BVB |
| ROA | minimum 5% | Indicatori fundamentali |
| ROE | minimum 15% | Indicatori fundamentali |
| Capitalizare | minimum 1 mld. lei | antetul Analizei companiei |

Notă: pe BVB, testul istoric din aplicație a găsit putere de predicție la P/E, P/BV, dividend și Piotroski, dar **nu** la ROE, marjă și îndatorare — acestea rămân filtre de calitate și risc, nu semnale de randament.
"""),
        ],
    },
    {
        "title": "Harta ta în investiții — când vinzi",
        "icon": "🚪",
        "sections": [
            ("Motive de vânzare (din notițele tale)", """
1. **Ținta de preț** stabilită la cumpărare a fost atinsă.
2. **Fundamentele se deteriorează** (venituri în scădere, probleme de conducere).
3. **Reechilibrare**: o poziție a devenit prea mare în portofoliu.
4. **Nevoie de lichiditate** — planificată din timp, ca să nu vinzi forțat într-un moment prost.
5. **Macro sau sentiment** nefavorabil sectorului (de ex. dobânzi în creștere pentru tehnologie).
6. **Performanță mult sub piață sau sub companiile similare**, pe termen lung — verifici dacă motivul e temporar sau fundamental.
7. **Optimizare fiscală**: vânzarea pozițiilor pe pierdere pentru a compensa câștiguri (cu verificarea regulilor fiscale în vigoare).
8. Regula ta: nu vinde la minim, așteaptă câteva zile o revenire (cazul AROBS). Atenție: revenirea nu e garantată — dacă motivul scăderii e fundamental, așteptarea poate costa.
"""),
            ("Cât lași să scadă / să crească", """
- **Stop-loss** la un procent fix (10%, 15%, 20%) după toleranța la risc — dar decizia finală se ia pe motivul scăderii: fundamental → ieși; zgomot de piață → poate fi oportunitate de cumpărare.
- La scădere, **reevaluezi premisele inițiale**: dacă sunt valide, menții sau cumperi mai mult la preț mai mic.
- La creștere: vinzi la ținta din evaluarea intrinsecă, dacă fundamentele nu s-au îmbunătățit cât să justifice o evaluare mai mare; fii flexibil la informații noi; compară cu indicele și cu companiile similare; vinde dacă analiza arată supraevaluare.
- O regulă simplă cu ATR (în aplicație, Analiză companie): un stop la 2–3 × ATR sub preț se adaptează la volatilitatea fiecărei acțiuni, în loc de același procent pentru toate.
"""),
        ],
    },
    {
        "title": "Construcția portofoliului",
        "icon": "🧱",
        "sections": [
            ("Diversificare și risc (din notițele tale)", """
- Diversifică; nu investi prea mult într-o singură acțiune sau sector; revizuiește regulat portofoliul.
- **Costurile și impozitele** reduc direct randamentul net: comisioane mici și planificare fiscală.
- Urmărește dobânzile, inflația, creșterea economică și politica monetară.
"""),
            ("Exemplul tău de alocare", """
| Categorie | Procent | Exemple |
|---|---|---|
| Acțiuni locale BVB | 50% | companii blue-chip |
| Obligațiuni | 25% | titluri de stat |
| ETF-uri | 15% | VUAA (S&P 500) |
| Mărfuri (metale prețioase) | 5% | aur, argint |
| Lichiditate | 5% | numerar |
"""),
            ("Ce adaug eu", """
- **Diversificarea se măsoară prin corelația dintre poziții**, nu prin numărul lor: 10 bănci românești se mișcă aproape ca una singură.
- **Rebalansare** la un interval fix (de ex. anual) sau când o clasă iese cu peste 5 pp din ținta ei: vinzi ce a crescut, cumperi ce a scăzut — disciplină anti-emoțională.
- **Dimensionarea poziției** după risc: o acțiune cu volatilitate dublă primește aproximativ jumătate din pondere, ca să contribuie cu același risc.
- **50% BVB** înseamnă concentrare pe o singură țară, monedă și o piață cu lichiditate mică: în crize locale, acțiunile, leul și titlurile de stat românești pot scădea împreună.
- **Titlurile de stat** sunt reperul fără risc în lei: un portofoliu de acțiuni care nu le bate pe termen lung nu și-a plătit riscul.
"""),
        ],
    },
    {
        "title": "Psihologia investitorului",
        "icon": "🧠",
        "sections": [
            ("Capcanele comune", """
- **Efectul de turmă** (din notițele tale): cumperi doar pentru că și alții cumpără.
- **Ancorarea** (din notițele tale): te fixezi pe un preț (cel de cumpărare, maximul vechi) fără să ții cont de schimbările fundamentale.
- **Aversiunea la pierdere**: o pierdere doare de ~2 ori mai mult decât bucuria unui câștig egal, deci ții pierzătorii prea mult și vinzi câștigătorii prea devreme (efectul de dispoziție).
- **Confirmarea**: cauți doar informațiile care îți susțin decizia. Antidot: scrie dinainte ce te-ar face să vinzi.
- **Recența**: crezi că ultimele luni vor continua la nesfârșit (în bine sau în rău).
- **Excesul de încredere**: tranzacționezi prea des; fiecare tranzacție costă comision și impozit.
"""),
        ],
    },
]


def _norm(text):
    """Litere mici, fără diacritice: „Dobânzi” → „dobanzi”, ca să găsească și fără diacritice."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(text).lower()) if not unicodedata.combining(c))


def search(chapters, query):
    """Capitolele cu secțiunile care conțin toate cuvintele căutate (în titlu sau text). Interogare goală → tot."""
    words = [w for w in _norm(query).split() if w]
    if not words:
        return chapters
    out = []
    for ch in chapters:
        hits = [(t, body) for t, body in ch["sections"]
                if all(w in _norm(ch["title"] + " " + t + " " + body) for w in words)]
        if hits:
            out.append({**ch, "sections": hits})
    return out


def render(st):
    """Pagina „Glosar economic”."""
    st.title("📚 Glosar economic")
    st.caption(INTRO)
    query = st.text_input("Caută în glosar", placeholder="de ex. aur, dobânzi, yen, stop-loss", key="glosar_cautare")
    chapters = search(CHAPTERS, query)
    if not chapters:
        st.info("Nimic găsit. Încearcă un singur cuvânt (fără diacritice merge și el).")
        return
    for ch in chapters:
        st.markdown(f"### {ch['icon']} {ch['title']}")
        for title, body in ch["sections"]:
            with st.expander(title, expanded=bool(query.strip())):
                st.markdown(body.strip())
    st.caption("Capitolele marcate „din notițele tale” vin din documentele tale „Minighid de Macroeconomie” și "
               "„Hartă în investiții”; restul e teorie uzuală de piață. Nu e consultanță financiară.")
