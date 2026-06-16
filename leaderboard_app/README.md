# Time Attack Leaderboard

App separata per gestire una classifica manuale top 10.

## Avvio

Doppio click su:

```bat
run_leaderboard.bat
```

Oppure da terminale:

```bat
python leaderboard_app\app.py
```

## Uso

- Inserisci il nome del pilota.
- Inserisci il tempo in uno di questi formati:
  - `1:42.315`
  - `1:42:315`
  - `102.315`
- Premi `AGGIUNGI`.

Se lo stesso pilota inserisce un tempo migliore, il record viene aggiornato. Se il tempo e peggiore, la classifica resta invariata.

I dati vengono salvati in `leaderboard_data.json`.
