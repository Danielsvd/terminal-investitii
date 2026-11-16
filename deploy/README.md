# Rulare pe serverul propriu (Ubuntu)

Aplicația oficială rulează din `~/apps/terminal-investitii` (branch `main`), ca serviciu
`terminal-investitii.service`, accesibilă doar prin Tailscale. Cheia (`.streamlit/secrets.toml`) stă
doar pe server și nu intră niciodată în repo.

## Jobul IV_istoric

- Cod: `tools/iv_snapshot.py` (aceleași calcule ca aplicația: expirarea 20–45 de zile, IV la bani).
- Simboluri: cele SUA din foaia `Portofoliu` (Currency USD) și din `watchlist`, citite la fiecare rulare.
- Programare: `systemd/iv-snapshot.timer`, luni–vineri 15:30 ora New York. Scrie cel mult un rând pe
  (zi, simbol); dacă aplicația a salvat deja în ziua respectivă, sare. În afara ședinței nu scrie nimic.
- Probă fără scriere: `.venv/bin/python -m tools.iv_snapshot --dry-run`.
- Jurnal: `journalctl -u iv-snapshot -n 50 --no-pager`. Următoarea rulare: `systemctl list-timers iv-snapshot.timer`.

## Jobul Pilon II (deținerile fondurilor de pensii)

- Cod: `tools/pilon2/` (`fetch.py` descarcă rapoartele Anexa 12, `parse.py` le citește, `run.py` scrie foile
  `Pilon2_detineri` și `ISIN_map`). Folder dedicat: `~/apps/pilon2/` (mediul `.venv`, arhiva `pdf/`,
  `surse.json` opțional pentru adresele paginilor de rapoarte).
- Mediul separat, cu același Python ca aplicația:
  ```bash
  ~/apps/terminal-investitii/.venv/bin/python -m venv ~/apps/pilon2/.venv
  ~/apps/pilon2/.venv/bin/pip install -r ~/apps/terminal-investitii/tools/pilon2/requirements.txt
  ```
- Probă fără scriere: `cd ~/apps/terminal-investitii && ~/apps/pilon2/.venv/bin/python -m tools.pilon2.run --data-dir ~/apps/pilon2 --proba`.
- Istoricul (o singură dată): `... -m tools.pilon2.run --data-dir ~/apps/pilon2 --luni 30 --istoric`.
- Programare: `systemd/pilon2-snapshot.timer`, pe 5 și pe 20 ale lunii. Un raport cu probleme nu se scrie (apare
  „RESPINS” în jurnal). Jurnal: `journalctl -u pilon2-snapshot -n 80 --no-pager`.
- Timerul se instalează doar după ce codul e în `main` (rulează din checkout-ul aplicației).
- NN publică pe pagină doar raportul curent; lunile vechi se scot din arhiva `.zip` a fondului Pilon II
  („…fond pensii private 2021 - 2026.zip”; pagina are și arhivele NN Activ/Optim, excluse prin `filtru_zip`).
  Arhiva se păstrează în `pdf/nn/_arhiva.zip` + `_arhiva.url` și se descarcă din nou cel mult o dată la 30 de zile.
