"""Taratura iperparametri Elo/Poisson sul backtest WC (point-in-time).

Grid su: home_adv (Elo), k_scale (Elo), half_life (Poisson).
Selezione sul TRAIN (Mondiali 2002-2014), verifica sul TEST (2018-2022).
Adotta una combinazione solo se migliora il test rispetto all'attuale.
"""
import itertools
import sys

import numpy as np
import pandas as pd

from config import WC_START_DATES, DATA_PROC, ELO_HOME_ADV
from elo import compute_elo
from poisson import fit, match_probs
import metrics as M

TRAIN_WC = [2002, 2006, 2010, 2014]
TEST_WC = [2018, 2022]


def eval_combo(m, home_adv, k_scale, half_life):
    mm, _, _ = compute_elo(m, k_scale=k_scale, home_adv=home_adv)
    res = {}
    for year in TRAIN_WC + TEST_WC:
        cutoff = pd.Timestamp(WC_START_DATES[year])
        train = mm[(mm.date < cutoff) & (mm.date >= cutoff - pd.Timedelta(days=365 * 18))]
        params = fit(train, cutoff, half_life_days=half_life)
        wc = mm[(mm.tournament == "FIFA World Cup") & (mm.date >= cutoff)
                & (mm.date < cutoff + pd.Timedelta(days=60))]
        probs, outs = [], []
        for _, r in wc.iterrows():
            (p1, px, p2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, params)
            probs.append([p1, px, p2]); outs.append(M.outcome_index(r.home_score, r.away_score))
        res[year] = (np.array(probs), np.array(outs))
    return res


def pooled(res, years):
    P = np.vstack([res[y][0] for y in years]); O = np.concatenate([res[y][1] for y in years])
    return M.log_loss(P, O), M.rps(P, O), M.accuracy(P, O)


def run():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    grid = list(itertools.product([45, 65, 85, 110], [0.85, 1.0, 1.2], [600, 900, 1400]))
    rows = []
    for ha, ks, hl in grid:
        res = eval_combo(m, ha, ks, hl)
        tr_ll, tr_rps, _ = pooled(res, TRAIN_WC)
        te_ll, te_rps, te_acc = pooled(res, TEST_WC)
        rows.append((ha, ks, hl, tr_ll, te_ll, te_rps, te_acc))
        print(f"ha={ha:3} k={ks:.2f} hl={hl:4}  train_ll={tr_ll:.4f}  "
              f"test_ll={te_ll:.4f} test_rps={te_rps:.4f} test_acc={te_acc:.3f}",
              file=sys.stderr)
    df = pd.DataFrame(rows, columns=["home_adv", "k_scale", "half_life",
                                     "train_ll", "test_ll", "test_rps", "test_acc"])
    best = df.sort_values("train_ll").iloc[0]
    cur = df[(df.home_adv == 65) & (df.k_scale == 1.0) & (df.half_life == 900)].iloc[0]
    print("\n=== ATTUALE (ha=65,k=1.0,hl=900) ===")
    print(f"test_ll={cur.test_ll:.4f} test_rps={cur.test_rps:.4f} test_acc={cur.test_acc:.3f}")
    print("=== MIGLIORE su train ===")
    print(f"ha={int(best.home_adv)} k={best.k_scale} hl={int(best.half_life)}  "
          f"test_ll={best.test_ll:.4f} test_rps={best.test_rps:.4f} test_acc={best.test_acc:.3f}")
    df.to_csv(DATA_PROC / "tune.csv", index=False)
    return best, cur


if __name__ == "__main__":
    run()
