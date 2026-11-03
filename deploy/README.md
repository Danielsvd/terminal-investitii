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
