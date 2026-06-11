"""Modello-mercato (bookmaker consensus) vs il mio consenso a 5 modelli.

Dataset: probabilita di consenso bookmaker (Leitner/Zeileis/Hornik) per WC
2010/2014/2018/2022, ricostruito in data/market/market_winprob.csv.

Test: per ogni Mondiale confronto il rank e il log-score del campione reale sotto
(a) il mio modello, (b) il mercato, (c) il blend a vari pesi. Se un blend batte
sia il mio modello sia il mercato, il mercato aggiunge informazione.
"""
import csv
import sys
import numpy as np
import pandas as pd

from config import DATA_PROC, WC_START_DATES
from elo import compute_elo, ratings_as_of
from poisson import fit
import retro as R

WEIGHTS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]   # w = peso MIO modello; (1-w) = mercato


def load_market():
    path = DATA_PROC.parent / "market" / "market_winprob.csv"
    M = {}
    for row in csv.DictReader(open(path)):
        M.setdefault(int(row["year"]), {})[row["team"]] = float(row["market_prob_pct"]) / 100.0
    return M


def my_champ_probs(mm, tl, year):
    rec = R.reconstruct(mm, year)
    cutoff = WC_START_DATES[year]
    elo = ratings_as_of(tl, pd.Timestamp(cutoff))
    train = mm[(mm.date < pd.Timestamp(cutoff)) & (mm.date >= pd.Timestamp(cutoff) - pd.Timedelta(days=365 * 16))]
    params = fit(train, cutoff)
    cons_ko, cons_group = R._build_consensus(mm, rec, elo, params, cutoff)
    probs = R.simulate_past(rec, elo, params, n=40000, cons_ko=cons_ko, cons_group=cons_group)
    teams = [t for g in rec["groups"] for t in g]
    return rec["actual"]["champion"], teams, {p["team"]: p["p_champion"] for p in probs}


def run():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, tl = compute_elo(m)
    MK = load_market()

    per_w = {w: {"rank": [], "logscore": [], "prob": []} for w in WEIGHTS}
    print(f"{'anno':5}{'campione':14}" + "".join(f"w={w:<6}" for w in WEIGHTS) + "  (rank#, prob%)")
    for year in sorted(MK):
        champ, teams, mine = my_champ_probs(mm, tl, year)
        mk = MK[year]
        miss = [t for t in teams if t not in mk]
        if miss:
            print(f"[avviso] {year}: market manca {miss}", file=sys.stderr)
        line = f"{year:5}{champ[:13]:14}"
        for w in WEIGHTS:
            blend = {t: w * mine.get(t, 0.0) + (1 - w) * mk.get(t, 0.0) for t in teams}
            s = sum(blend.values()) or 1.0
            blend = {t: v / s for t, v in blend.items()}
            order = sorted(blend, key=lambda t: -blend[t])
            rank = order.index(champ) + 1
            pc = max(blend[champ], 1e-9)
            per_w[w]["rank"].append(rank); per_w[w]["logscore"].append(-np.log(pc)); per_w[w]["prob"].append(pc)
            line += f"#{rank:<2}{round(100*pc,1):<4}"
        print(line)

    print(f"\n{'peso mio':10}{'(mercato)':10}{'rank medio':>11}{'logscore':>10}{'prob media%':>12}")
    for w in WEIGHTS:
        r = np.mean(per_w[w]["rank"]); ls = np.mean(per_w[w]["logscore"]); pp = 100 * np.mean(per_w[w]["prob"])
        tag = "  <- solo mercato" if w == 0 else ("  <- solo mio" if w == 1 else "")
        print(f"{w:<10}{round(1-w,2):<10}{r:>11.2f}{ls:>10.3f}{pp:>12.1f}{tag}")
    print("\nLog-score piu basso = meglio prevede il campione reale. Se il minimo e a 0<w<1,")
    print("il blend col mercato batte sia il mio modello (w=1) sia il mercato (w=0).")


if __name__ == "__main__":
    run()
