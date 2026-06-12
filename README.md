# worldcup-2026-model

Open, daily-updated probabilistic model of the **2026 FIFA World Cup** (48 teams).
Consensus of six models + 100,000 Monte Carlo tournament simulations.

🇬🇧 [English](#english) · 🇮🇹 [Italiano](#italiano)

**Live dashboard:** [umbertobertonelli.it/macro/mondiale-2026](https://www.umbertobertonelli.it/macro/mondiale-2026/) · mirror on [GitHub Pages](https://drelegantia.github.io/worldcup-2026-model/)

---

## English

Predictive model of the **2026 FIFA World Cup** (48 teams). Every day it estimates,
with 100,000 Monte Carlo simulations of the whole tournament, each nation's
probability of winning its group, qualifying, and reaching every stage up to the
final. Dataset, model and dashboard are open and update automatically.

### How it works

The forecast is a **consensus of six models with different assumptions** (Market /
bookmaker consensus, Massey ratings, multinomial logistic, Elo-Poisson, recent Form,
Attack/Defense), weighted by leave-one-World-Cup-out validation.

1. **Point-in-time Elo** computed over the full history of international matches
   (since 1872, open dataset [martj42/international_results](https://github.com/martj42/international_results)),
   with home advantage and a goal-difference multiplier.
2. **Dixon-Coles bivariate Poisson** driven by the Elo gap: expected goals per match,
   hence 1 / X / 2 probabilities and the score distribution.
3. **Monte Carlo** of the whole tournament: groups, selection of the 8 best
   third-placed teams via the **official FIFA Annexe C** allocation (all 495
   combinations), R32-to-final bracket per the official FIFA structure. 100,000
   iterations. Matches already played use the real result; the rest are simulated.

### Validation (point-in-time backtest 2002-2022)

Tested on the 6 World Cups using only data available before each tournament, over 384
matches:

| Metric | Model | Uniform |
|---|---|---|
| Log loss | 0.976 | 1.099 |
| RPS | 0.200 | - |
| Brier | 0.576 | - |
| Accuracy | 56.8% | 33% |

In line with the state of the art for international football. Per-year detail in
`data/processed/backtest.json`.

### Local use

```bash
pip install -r requirements.txt
python src/ingest.py          # download and prepare data
python src/backtest.py        # backtest 2002-2022
python src/run_daily.py       # today's snapshot (100k simulations)
```

### Data and license

Public sports data only. Match results: martj42/international_results (open license).
Tournament structure: official FIFA data. For information and outreach only, not
betting advice.

---

## Italiano

Modello predittivo del **Mondiale FIFA 2026** (48 squadre). Stima ogni giorno, con
100.000 simulazioni Monte Carlo, la probabilità di ciascuna nazionale di vincere il
girone, qualificarsi e arrivare a ogni fase fino alla finale. Dataset, modello e
dashboard sono aperti e si aggiornano in automatico.

### Come funziona

La previsione è il **consenso di sei modelli con ipotesi diverse** (Mercato /
consenso bookmaker, Massey, Logistica, Elo-Poisson, Forma, Attacco/Difesa), pesati
con validazione leave-one-Mondiale-out.

1. **Elo** point-in-time calcolato sull'intera storia delle partite internazionali
   (dal 1872, fonte aperta [martj42/international_results](https://github.com/martj42/international_results)),
   con vantaggio campo e moltiplicatore per lo scarto di gol.
2. **Poisson bivariato Dixon-Coles** guidato dalla differenza Elo: stima i gol attesi
   di ogni partita e quindi le probabilità 1 / X / 2 e la distribuzione dei risultati.
3. **Monte Carlo** dell'intero torneo: gironi, selezione delle 8 migliori terze con
   allocazione ufficiale FIFA Annexe C (tutte le 495 combinazioni), tabellone dai
   sedicesimi alla finale secondo la struttura ufficiale FIFA. 100.000 iterazioni.
   Le partite già giocate usano il risultato reale, le altre sono simulate.

### Validazione (backtest point-in-time 2002-2022)

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

### Struttura

```
src/
  config.py      parametri, gironi 2026 ufficiali, normalizzazione nomi
  ingest.py      download dati + costruzione tabella match + struttura 2026
  elo.py         Elo point-in-time
  poisson.py     Poisson Dixon-Coles (fit MLE, predizione)
  metrics.py     log loss, Brier, RPS, calibrazione
  backtest.py    backtest 2002-2022
  bracket.py     tabellone 2026 + allocazione Annexe C delle terze
  simulate.py    Monte Carlo vettorizzato del torneo
  consensus.py   consenso dei sei modelli
  run_daily.py   orchestratore: snapshot datato + indice + serie storica
data/bracket/    tabella ufficiale FIFA Annexe C (495 combinazioni)
sims/2026/       snapshot giornalieri immutabili + index.json + history.json
docs/            dashboard statica (GitHub Pages) + mirror dati
.github/workflows/daily.yml   aggiornamento automatico giornaliero
```

### Uso locale

```bash
pip install -r requirements.txt
python src/ingest.py          # scarica e prepara i dati
python src/backtest.py        # backtest 2002-2022
python src/run_daily.py       # snapshot di oggi (100k simulazioni)
```

### Dati e licenza

Solo dati sportivi pubblici. Fonte risultati: martj42/international_results (licenza
aperta). Struttura torneo: dati ufficiali FIFA. Simulazione a scopo informativo e
divulgativo: le probabilità non sono garanzie né consigli di scommessa.
