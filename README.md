# worldcup-2026-model

Modello predittivo del **Mondiale FIFA 2026** (48 squadre). Stima ogni giorno, con
100.000 simulazioni Monte Carlo, la probabilita di ciascuna nazionale di vincere il
girone, qualificarsi e arrivare a ogni fase fino alla finale. Dataset, modello e
dashboard sono aperti e si aggiornano in automatico.

**Dashboard live:** vedi GitHub Pages di questo repo (cartella `docs/`).

## Come funziona

1. **Elo** point-in-time calcolato sull'intera storia delle partite internazionali
   (dal 1872, fonte aperta [martj42/international_results](https://github.com/martj42/international_results)),
   con vantaggio campo e moltiplicatore per lo scarto di gol.
2. **Poisson bivariato Dixon-Coles** guidato dalla differenza Elo: stima i gol attesi
   di ogni partita e quindi le probabilita 1 / X / 2 e la distribuzione dei risultati.
3. **Monte Carlo** dell'intero torneo: gironi, selezione delle 8 migliori terze con
   allocazione ufficiale Annex C, tabellone R32 -> finale. 100.000 iterazioni.
   Le partite gia giocate usano il risultato reale, le altre sono simulate.

## Validazione (backtest point-in-time 2002-2022)

Testato sui 6 Mondiali usando solo i dati disponibili prima di ogni torneo. Su 384
partite:

| Metrica | Modello | Uniforme |
|---|---|---|
| Log loss | 0,976 | 1,099 |
| RPS | 0,200 | - |
| Brier | 0,576 | - |
| Accuracy | 56,8% | 33% |

Valori in linea con lo stato dell'arte per il calcio internazionale. Dettaglio per
anno in `data/processed/backtest.json`.

## Struttura

```
src/
  config.py      parametri, gironi 2026 ufficiali, normalizzazione nomi
  ingest.py      download dati + costruzione tabella match + struttura 2026
  elo.py         Elo point-in-time
  poisson.py     Poisson Dixon-Coles (fit MLE, predizione)
  metrics.py     log loss, Brier, RPS, calibrazione
  backtest.py    backtest 2002-2022
  bracket.py     tabellone 2026 + allocazione Annex C delle terze
  simulate.py    Monte Carlo vettorizzato del torneo
  run_daily.py   orchestratore: snapshot datato + indice + serie storica
sims/2026/       snapshot giornalieri immutabili + index.json + history.json
docs/            dashboard statica (GitHub Pages) + mirror dati
.github/workflows/daily.yml   aggiornamento automatico giornaliero
```

## Uso locale

```bash
pip install -r requirements.txt
python src/ingest.py          # scarica e prepara i dati
python src/backtest.py        # backtest 2002-2022
python src/run_daily.py       # snapshot di oggi (100k simulazioni)
```

## Dati e licenza

Solo dati sportivi pubblici. Fonte risultati:
martj42/international_results (licenza aperta). Struttura torneo: dati ufficiali FIFA.

Simulazione a scopo informativo e divulgativo. Le probabilita non sono garanzie ne
consigli di scommessa.
