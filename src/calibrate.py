"""Idea presa dai concorrenti (best practice mancante): layer di CALIBRAZIONE.

Temperature scaling sulle probabilita 1X2 del modello: p_cal ~ softmax(log p / T).
T si stima sui Mondiali 2002-2014 (train) minimizzando il log loss, si valuta su
2018-2022 (test). Si adotta solo se migliora out-of-sample.

Confronta anche un secondo iperparametro: il denominatore Elo->gol (400 vs altri),
come fa Hicruben.
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from config import BACKTEST_WORLD_CUPS, WC_START_DATES, DATA_PROC
from elo import compute_elo
from poisson import fit, match_probs
import metrics as M

TRAIN = [2002, 2006, 2010, 2014]
TEST = [2018, 2022]


def collect():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, _ = compute_elo(m)
    rows = []
    for year in BACKTEST_WORLD_CUPS:
        cutoff = pd.Timestamp(WC_START_DATES[year])
        train = mm[(mm.date < cutoff) & (mm.date >= cutoff - pd.Timedelta(days=365 * 18))]
        params = fit(train, cutoff)
        wc = mm[(mm.tournament == "FIFA World Cup") & (mm.date >= cutoff)
                & (mm.date < cutoff + pd.Timedelta(days=60))]
        for _, r in wc.iterrows():
            (p1, px, p2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, params)
            rows.append({"year": year, "p": [p1, px, p2],
                         "o": M.outcome_index(r.home_score, r.away_score)})
    return pd.DataFrame(rows)


def temp(probs, T):
    logp = np.log(np.clip(probs, 1e-12, 1))
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run():
    df = collect()
    P = np.array(df["p"].tolist()); O = df["o"].values
    trm = df["year"].isin(TRAIN).values; tem = df["year"].isin(TEST).values

    res = minimize_scalar(lambda T: M.log_loss(temp(P[trm], T), O[trm]),
                          bounds=(0.5, 2.5), method="bounded")
    Tbest = res.x
    print(f"Temperatura ottimale (train 2002-2014): T = {Tbest:.3f}  (T>1 = modello troppo sicuro)\n",
          file=sys.stderr)

    def show(label, probs, mask):
        print(f"{label:28}{M.log_loss(probs[mask],O[mask]):9.4f}"
              f"{M.rps(probs[mask],O[mask]):8.4f}{M.brier(probs[mask],O[mask]):8.4f}"
              f"{M.accuracy(probs[mask],O[mask]):7.3f}{M.calibration(probs[mask],O[mask]):8.4f}")

    print(f"{'modello':28}{'logloss':>9}{'rps':>8}{'brier':>8}{'acc':>7}{'ece':>8}")
    print("--- TEST 2018-2022 ---")
    show("base", P, tem); show(f"calibrato (T={Tbest:.2f})", temp(P, Tbest), tem)
    print("--- POOLED 2002-2022 ---")
    full = np.ones(len(O), bool)
    show("base", P, full); show(f"calibrato (T={Tbest:.2f})", temp(P, Tbest), full)
    return Tbest


if __name__ == "__main__":
    run()
