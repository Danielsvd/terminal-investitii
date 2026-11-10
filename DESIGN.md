# DESIGN.md — Terminal Investiții PRO

Direcția: **Bleumarin + Revolut** (aleasă de Daniel pe 10.11.2026), cu **paleta „Indigo profund”** (schimbată de Daniel pe 10.11, 19:12; PR #169). Valorile curente sunt în §1.0; tabelele §1.1–§1.4 descriu rolurile.
Baza e Bleumarin: un terminal sobru, cu margini fine și densitate mare pe tabele. Din stilul Revolut se iau doar cifra mare de sus, colțurile mari și cardurile fără margine pe ecranele de sinteză, plus butonul principal complet rotunjit.
Starea documentului: **aprobat** (10.11.2026). Pe test1: #165 (`config.toml`), #166 (tokenuri light-dark), #167 (banner mobil), #168 (iconuri).

> Direcția nu vine din comparația cu site-uri reale (UXKIN nu e conectat). Vine din cele 7 machete arătate în conversație, din CSS-ul actual din `app.py` și din constrângerile Streamlit 1.65.

---

## 0. Principii

1. **Cifrele sunt conținutul.** Tot restul (culori, umbre, iconuri) le servește. Fără gradiente decorative, efecte de sticlă sau strălucire.
2. **O singură culoare de acțiune.** Albastrul se folosește pentru butoane, linkuri, selecție și focus. Verdele și roșul înseamnă doar câștig/pierdere și verdict, niciodată decor.
3. **Două densități.**
   - **Sinteză**, în stil Revolut: Rezumatul zilei, antetul Portofoliului, cardurile de verdict.
   - **Date**, în stil Bleumarin: tabele, matrici, opțiuni, registru, Watchlist.
4. **Ambele moduri sunt egale.** Orice culoare are pereche luminos/întunecat. Nicio culoare nu se scrie direct în cod: se folosesc doar tokenurile de mai jos.
5. **Contrast minim 4,5:1** pentru text, verificat pentru fiecare pereche din §1.4.

---

## 1. Culoare

### 1.0 Paleta curentă — „Indigo profund” (#169, sursa de adevăr: `ui/theme.py` TOKENS = `.streamlit/config.toml`)

| Token | Întunecat | Luminos | Rol |
|---|---|---|---|
| `bg` | `#141230` | `#F4F4FB` | fundalul paginii, cu strălucire discretă `glow` în capul paginii |
| `surface` / `surface-top` | `#1C1940` / `#221F4C` | `#FFFFFF` / `#FFFFFF` | carduri: gradient fin de sus în jos |
| `surface-2` | `#242150` | `#F8F8FD` | antet de tabel, interiorul unui card |
| `border` / `border-strong` | `#2E2A5C` / `#3D3874` | `#E3E1F3` / `#CFCCE8` | contur de 1px, hover |
| `text` / `muted` | `#ECEAFB` / `#A09BC8` | `#16133A` / `#5F5A8A` | text și etichete |
| `accent` / `accent-text` | `#818CF8` / `#A5B4FC` | `#4F46E5` / `#4338CA` | selecție, bare / linkuri; butonul principal: `#6366F1` (întunecat), `#4F46E5` (luminos) |
| `up` / `down` / `warn` | `#34D399` / `#F87171` / `#FBBF24` | `#047857` / `#B91C1C` / `#B45309` | câștig / pierdere / atenție |
| `highlight`, `shadow`, `glow` | `#FFFFFF0F`, `#05041473`, `#818CF81A` | `#FFFFFFE6`, `#16133A12`, `#4F46E50F` | linia luminoasă sus, umbra moale, strălucirea paginii |

Contrast (calculat): text / muted / up / down / warn ≥ 5,0 pe bg, surface și surface-2 în ambele moduri; text alb pe butonul principal 4,47 (întunecat) și 6,3 (luminos).
Carduri premium (`ui_theme.premium_css()`): gradient `surface-top → surface`, contur `border`, colțuri 14px, umbră `inset 0 1px 0 highlight, 0 1px 2px shadow, 0 10px 28px shadow`. Se aplică pe `.fin-card`, `.news-card`, `st.metric`, expandere și pe cardurile HTML vechi cu fundal `surface`.

### 1.1 Suprafețe și text (paleta inițială Bleumarin, înlocuită de §1.0 — rolurile rămân)

| Token | Întunecat | Luminos | Rol |
|---|---|---|---|
| `--bg` | `#0B1220` | `#F3F5F9` | fundalul paginii (`backgroundColor`) |
| `--surface` | `#111A2E` | `#FFFFFF` | carduri, metrici, expandere (`secondaryBackgroundColor`) |
| `--surface-2` | `#18233B` | `#F8FAFC` | rând evidențiat, antet de tabel, interiorul unui card |
| `--border` | `#1F2A44` | `#DCE2EC` | margini fine (0,5–1px) pe densitatea „Date” |
| `--border-strong` | `#2E3B5C` | `#C4CDDB` | hover și separatoare importante |
| `--text` | `#E5ECF6` | `#0F172A` | text principal și cifre |
| `--text-muted` | `#8A97AD` | `#526079` | etichete, unități, surse, „?” |
| `--text-faint` | `#5B6880` | `#8792A6` | doar text decorativ sau dezactivat, **niciodată** cifre |

### 1.2 Acțiune (accent)

| Token | Întunecat | Luminos | Rol |
|---|---|---|---|
| `--accent` | `#3B82F6` | `#1D4ED8` | fundalul butonului principal, bara de scor, selecția (`primaryColor`) |
| `--accent-text` | `#60A5FA` | `#1D4ED8` | linkuri și text albastru pe card |
| `--accent-bg` | `#142A52` | `#DBEAFE` | cip informativ, rândul selectat |
| `--on-accent` | `#FFFFFF` | `#FFFFFF` | textul de pe butonul principal |

Pe butonul principal din modul întunecat textul alb stă pe `#2563EB` (fundalul de apăsare/hover), cu contrast 5,2:1. Pe `#3B82F6` textul alb are doar 3,7:1, ceea ce e sub prag pentru text mic. De aceea butonul folosește `#2563EB` ca fundal de bază, iar `--accent` rămâne pentru bare și selecție.

### 1.3 Semantică financiară

| Token | Întunecat | Luminos | Rol |
|---|---|---|---|
| `--up` | `#22C55E` | `#15803D` | câștig, variație pozitivă |
| `--down` | `#F87171` | `#B91C1C` | pierdere, variație negativă |
| `--up-bg` | `#0F2F1E` | `#DCFCE7` | cip/fond pentru FAVORABIL, „peste prag” |
| `--down-bg` | `#3A1518` | `#FEE2E2` | cip/fond pentru NEFAVORABIL, alertă 2σ |
| `--warn` | `#FBBF24` | `#B45309` | neutru/atenție, „netestat”, date vechi |
| `--warn-bg` | `#3A2A08` | `#FEF3C7` | cip „netestat”, „proxy”, „eșantion mic” |
| `--na` | `--text-muted` | `--text-muted` | „N/A”: fără culoare semantică, ca lipsa de date să nu pară un semnal |

Reguli:
- Semnul se scrie mereu (`+1,24%`, `−0,85%` cu minus tipografic), ca sensul să nu depindă doar de culoare.
- Verdictele apar ca cip: fundal `--*-bg`, text în culoarea semantică, eticheta și scorul numeric (de ex. „Moment: FAVORABIL 100”).
- O valoare de 0 exact se colorează cu `--text`, nu cu verde sau roșu.

### 1.4 Contrast verificat (WCAG, raport față de fundalul pe care stă)

| Pereche | Întunecat | Luminos |
|---|---|---|
| text / bg | 15,8 | 16,4 |
| text-muted / surface | 5,9 | 6,4 |
| accent-text / surface | 6,8 | 6,7 |
| up / surface | 7,6 | 5,0 |
| down / surface | 6,3 | 6,5 |
| warn / surface | 10,4 | 5,0 |
| on-accent / buton | 5,2 (pe `#2563EB`) | 6,7 |

### 1.5 Grafice (Plotly)

- Fundalul graficului e transparent (`paper_bgcolor` și `plot_bgcolor` = `rgba(0,0,0,0)`), deci graficul preia culoarea cardului.
- Grila e `--border`, axele și etichetele sunt `--text-muted`, iar fontul e cel din §2.
- Seriile categoriale, în ordine: `#3B82F6` albastru, `#F59E0B` chihlimbar, `#14B8A6` turcoaz, `#A78BFA` violet, `#F472B6` roz, `#94A3B8` gri. În modul luminos sunt aceleași nuanțe, cu un pas mai închise: `#1D4ED8`, `#B45309`, `#0F766E`, `#7C3AED`, `#BE185D`, `#475569`.
- Lumânări: `--up` / `--down`. Benchmark: gri `#94A3B8`, linie punctată.
- Pentru seria „Depus net” din registru se păstrează chihlimbarul existent (`#C98500` → `#F59E0B` / `#B45309`).
- Corelații și heatmap-uri: scală divergentă `--down` → `--surface-2` → `--accent`, cu centrul în 0.
- Gradientul de la bara analiștilor (verde → galben → roșu) rămâne, fiindcă are sens: e o scală. Se refac doar culorile, cu tokenurile de mai sus.

Implementat (#166): Streamlit aplică deja tema lui peste șablonul figurii, așa că `plot_ro()` corectează doar culorile scrise pentru fundal închis, prin `ui_theme.theme_figure_dict()` (§7.2).

---

## 2. Tipografie

- **Familie (decis 10.11):** fontul de sistem (`font = "sans-serif"` în tema Streamlit). Nu se adaugă fonturi externe acum: Google Fonts ar fi o sursă externă nouă. Poate fi discutat separat.
- **Cifre tabulare peste tot:** `font-variant-numeric: tabular-nums;` pe metrici, tabele și cardurile HTML, ca zecimalele să stea aliniate.
- **Format numeric românesc:** neschimbat (`ro()`, `separators=",."`).

| Nivel | Mărime / înălțime rând | Greutate | Unde |
|---|---|---|---|
| Hero (Revolut) | 32px / 1,15 | 600 | valoarea portofoliului, scorul principal; doar pe Sinteză |
| Metrică | 24px / 1,2 | 600 | `st.metric` (`metricValueFontSize`) |
| H2 secțiune | 20px / 1,3 | 600 | titlurile de tab și secțiune |
| H3 card | 16px / 1,4 | 600 | titlul unui card |
| Text | 15px / 1,55 | 400 | paragrafe, ghidul macro (`baseFontSize = 15`) |
| Tabel | 14px / 1,4 | 400 | `st.dataframe`, tabele HTML |
| Etichetă | 13px / 1,3 | 500 | etichete de metrică, cipuri |
| Notă | 12px / 1,4 | 400 | sursă, dată, „proxy”, număr de observații |

Nu se folosește text sub 12px. Eticheta de 10px de la bara analiștilor urcă la 12px.

---

## 3. Spațiere

Scara pe 4px: `4 · 8 · 12 · 16 · 20 · 24 · 32`.

| Situație | Spațiu interior | Spațiu între elemente |
|---|---|---|
| Card de date (Bleumarin) | 12–16px | 12px |
| Card de sinteză (Revolut) | 20px | 16px |
| Rând de tabel | 8px vertical | — |
| Între secțiuni | — | 32px |

Pe telefon (sub 640px), spațiul interior al cardurilor coboară cu un pas (20 → 16, 16 → 12).

---

## 4. Formă și adâncime

| Token | Valoare | Unde |
|---|---|---|
| `--radius-sm` | 6px | cipuri pătrățoase, câmpuri de formular, celule |
| `--radius` | 10px | carduri de date, metrici, expandere (`baseRadius`) |
| `--radius-lg` | 20px | carduri de sinteză (Revolut) |
| `--radius-pill` | 999px | butonul principal, cipurile de verdict (`buttonRadius = "full"`) |

- **Umbre:** niciuna în modul întunecat, unde suprafața face separarea. În modul luminos, doar pe cardurile de sinteză: `0 1px 2px rgba(15,23,42,.06), 0 4px 12px rgba(15,23,42,.05)`.
- **Margini:** cardurile de date au `1px solid var(--border)`. Cardurile de sinteză nu au margine (Revolut) și se separă prin `--surface` pe `--bg`.
- **Fără `transform` la hover** pe carduri: mișcarea nu ajută la citit cifre. La hover, marginea trece pe `--border-strong`.
- **Bara colorată din stânga** (știri): se păstrează, cu `border-radius: 0` pe acea latură.

---

## 5. Componente

### 5.1 Card de sinteză (Revolut)
- `--surface`, `--radius-lg`, fără margine, spațiu interior 20px.
- Sus: etichetă 13px `--text-muted`. Dedesubt: cifra hero de 32px și variația cu semn în `--up`/`--down`, 15px/600.
- Dedesubt, opțional, o linie de context de 12px (sursă, dată, curs BNR).
- Se folosește pe: Rezumatul zilei (valoare totală, rezultatul zilei pe piețe), antetul Portofoliului (valoare consolidată RON, TWR, XIRR) și cardul principal de verdict din Analiză.

### 5.2 Card de date (Bleumarin)
- `--surface`, `--radius`, `1px solid var(--border)`, spațiu interior 12–16px.
- Titlul H3 și „?”-ul (`help=`) stau pe același rând.
- Se folosește pe: fundamentale, risc, DCF, macro și tot restul.

### 5.3 Metrică (`st.metric`)
- Stilul vine din temă (`metricValueFontSize`, `metricValueFontWeight`) și din CSS-ul de mai jos. Fundal `--surface`, `--radius`, margine `--border`.
- Delta folosește culorile native Streamlit, legate prin temă de `greenColor`/`redColor` = `--up`/`--down`.

### 5.4 Cip de verdict și de stare
- `--radius-pill`, spațiu 3×10px, 13px/500.
- FAVORABIL: `--up-bg` + `--up`. NEFAVORABIL: `--down-bg` + `--down`. NEUTRU, „netestat”, „proxy”: `--warn-bg` + `--warn`. Informativ: `--accent-bg` + `--accent-text`.
- Textul conține mereu eticheta și scorul numeric.

### 5.5 Butoane
- **Principal:** `--radius-pill`, fundal `--accent` (în modul întunecat `#2563EB`), text `--on-accent`. Cel mult unul pe secțiune („Analizează”, „Importă”, „Scrie în Sheets”).
- **Secundar:** stilul Streamlit implicit, cu raza din temă.
- **Focus:** contur de 2px `--accent`, decalat cu 2px. Nu se elimină.
- Acțiunile care scriu în Sheets rămân secundare până la previzualizare. Doar butonul de confirmare e principal.

### 5.6 Tabele (`st.dataframe` și HTML)
- Tema: `dataframeBorderColor = --border`, `dataframeHeaderBackgroundColor = --surface-2`, `dataframeHeaderTextColor = --text-muted`.
- Cifrele se aliniază la dreapta, cu `tabular-nums`. Textul se aliniază la stânga.
- Câștig/pierdere doar prin culoarea textului, nu prin fundalul celulei. Excepție: heatmap-urile.
- „N/A” în `--text-muted`.

### 5.7 Taburi și navigare
- Tabul activ are subliniere de 2px `--accent` și text `--text`. Taburile inactive au `--text-muted`.
- Comutatorul luminos/întunecat e cel nativ Streamlit (meniul ⋮ → Settings → Theme), care apare automat când sunt definite ambele teme. Nu se construiește alt comutator.

### 5.8 Știri
- Card de date cu bară de 4px în stânga: `--up` / `--down` / `--border-strong` după impact. Titlul e în `--accent-text`, metadatele în `--text-muted` la 12px.

### 5.9 Stări
- **Încărcare:** `st.spinner` cu text concret („Descarc prețurile…”).
- **Lipsă date:** un card de date cu „N/A” și motivul pe o linie (`--text-muted`), nu un card gol.
- **Eroare:** `st.error` scurt, ce s-a întâmplat și ce poți face. Fără text de excepție brut.

---

## 6. Iconuri și imagini

- **Decis 10.11:** meniul lateral și taburile fără piață trec pe iconuri Material native (`:material/…:`), fără dependență nouă. Etichetele din meniu se afișează prin `format_func`; valorile interne („1. Agregator Știri” etc.) rămân neschimbate.
- Meniu: Știri `newspaper` · Analiză companie `query_stats` · Portofoliu `account_balance_wallet` · Piață globală `public` · Import date `upload_file` · Rezumatul zilei `today` · Scanner volum `radar` · Watchlist `visibility`. Numerele „1.”–„8.” dispar din etichete.
- Steagurile 🇷🇴 🇪🇺 🇺🇸 rămân pe taburile de piață și valută (sunt informație). „Wall Street” → „SUA”. Dispozitive: Mac și Android, ambele afișează steagurile corect.
- Fără imagini decorative.

---

## 7. Implementare în Streamlit 1.65 (de făcut după aprobare)

Verificat în pachetul `streamlit==1.65.0`, versiunea fixată în `requirements.txt`: există secțiunile `[theme.light]` și `[theme.dark]` și `st.context.theme.type` (`"light"`/`"dark"`).

### 7.1 `.streamlit/config.toml` (fișier nou)
```toml
[theme]
font = "sans-serif"
baseFontSize = 15
baseRadius = "10px"
buttonRadius = "full"
metricValueFontSize = "24px"
metricValueFontWeight = 600
showWidgetBorder = true

[theme.dark]
primaryColor = "#3B82F6"
backgroundColor = "#0B1220"
secondaryBackgroundColor = "#111A2E"
textColor = "#E5ECF6"
borderColor = "#1F2A44"
linkColor = "#60A5FA"
greenColor = "#22C55E"
redColor = "#F87171"
yellowColor = "#FBBF24"
blueColor = "#3B82F6"
dataframeBorderColor = "#1F2A44"
dataframeHeaderBackgroundColor = "#18233B"
dataframeHeaderTextColor = "#8A97AD"
chartCategoricalColors = ["#3B82F6", "#F59E0B", "#14B8A6", "#A78BFA", "#F472B6", "#94A3B8"]

[theme.light]
primaryColor = "#1D4ED8"
backgroundColor = "#F3F5F9"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#0F172A"
borderColor = "#DCE2EC"
linkColor = "#1D4ED8"
greenColor = "#15803D"
redColor = "#B91C1C"
yellowColor = "#B45309"
blueColor = "#1D4ED8"
dataframeBorderColor = "#DCE2EC"
dataframeHeaderBackgroundColor = "#F8FAFC"
dataframeHeaderTextColor = "#526079"
chartCategoricalColors = ["#1D4ED8", "#B45309", "#0F766E", "#7C3AED", "#BE185D", "#475569"]
```
Verificat la rulare cu `streamlit==1.65.0`: cheile sunt valide. `base` NU se pune în `[theme.light]` / `[theme.dark]` (avertisment „not a valid config option”).

### 7.2 Tokenuri CSS pentru cardurile HTML proprii (făcut, PR #166)
- `ui/theme.py` (pur, cu teste în `tests/test_theme.py`): `TOKENS` (luminos, întunecat) și `root_css()`, care scrie `:root { --tk-*: light-dark(luminos, întunecat); }`.
- Streamlit pune `color-scheme: light|dark` pe containerul aplicației, deci `light-dark()` alege singur valoarea, și imediat după schimbarea temei din meniu, fără rerulare. `st.context.theme.type` NU se folosește: documentația Streamlit spune că poate fi greșit la prima încărcare și la schimbarea temei.
- `themed_html()` înlocuiește la afișare culorile paletei vechi (GitHub dark: `#161B22`, `#8B949E`, `#3FB950`...) cu `var(--tk-*)`, iar `rgba(...)` vechi cu `color-mix(...)`. `install()` trece prin ea tot ce se afișează cu `markdown(..., unsafe_allow_html=True)`. Codul care produce HTML-ul nu se schimbă.
- Tabelele (`st.dataframe` cu Styler) se desenează pe canvas, unde `var()` nu merge: folosesc culori de mijloc `DF_UP #3A8854`, `DF_DOWN #E82C2C`, cu contrast ≈ 4:1 pe ambele fundaluri (sub 4,5, asumat), iar rândul evidențiat e `rgba(59,130,246,0.14)`.
- Graficele Plotly: `plot_ro()` trece figura prin `theme_figure_dict()`. Fundalul `#0E1117` devine transparent, grila `#30363D` devine gri translucid, iar galbenul și cyanul pur trec pe chihlimbar și turcoaz. Restul stilului vine din tema Streamlit.
- Browsere: `light-dark()` cere Chrome 123+ / Safari 17.5+ (Mac și Android actualizate).

### 7.3 Ordinea PR-urilor (fiecare verificat pe test1)
1. ✅ #165 — `config.toml` cu ambele teme, plus eliminarea `background-color` forțat din CSS-ul global (`.stApp { background-color: #0E1117 }` ar bloca modul luminos).
2. ✅ #166 — `ui/theme.py`: carduri HTML, metrici, tabele și grafice lizibile în ambele moduri.
3. ✅ inclus în #166 — graficele prin `theme_figure_dict()` în `plot_ro()` (în loc de două șabloane Plotly: Streamlit aplică deja tema lui peste șablon).
4. Cardul de sinteză (Revolut) pe Rezumatul zilei și pe antetul Portofoliului.
5. Cardurile HTML rămase (verdicte, scor macro, ghid), pe rând, câte un tab pe PR.
6. ✅ #168 — iconuri Material în meniu și pe taburile fără piață (doar etichete, prin `format_func`; §6).

În plus: #167 — bannerul macro pe telefon (verdictul nu se mai rupe în cuvânt, `clamp()` + `flex-wrap`).

**NU se atinge:** calculele, textele, ordinea secțiunilor, numele taburilor și emoji-urile (excepție: PR 6, decis pe 10.11), formatul numeric românesc, `hub.service` / copia fixată a botului Telegram.

### 7.4 Verificare independentă (pentru fiecare PR)
- Pe test1: ⋮ → Settings → Theme, comuți „Light” / „Dark” și faci câte o captură pe același ecran, pe PC și pe telefon.
- În fiecare mod nu trebuie să rămână text alb pe alb sau negru pe negru. Cardurile, tabelele și graficele trebuie să-și schimbe fundalul.
- Câștigul e verde și pierderea roșie în ambele moduri. „N/A” e gri.
- Contrastul se poate verifica pe orice pereche cu un calculator WCAG (de ex. webaim.org/resources/contrastchecker) folosind valorile din §1.
