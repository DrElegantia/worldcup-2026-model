"""Idea dei concorrenti (lbenz730/0xNadr): modello ATTACCO/DIFESA per squadra.

Invece di derivare i gol da un Elo scalare, si stimano per ogni nazionale un
parametro di attacco e uno di difesa (rappresentazione 2D, asimmetrica), via
regressione di Poisson con shrinkage L2 e pesatura temporale. E' la forma classica
Dixon-Coles ed e' piu' espressiva del mio Elo->gol.

Confronto sul backtest WC con il baseline Elo-Poisson, pooled e test 2018-2022.
"""
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson as pois

from config import BACKTEST_WORLD_CUPS, WC_START_DATES, DATA_PROC
from elo import compute_elo
from poisson import fit, match_probs
import metrics as M

TRAIN = [2002, 2006, 2010, 2014]; TEST = [2018, 2022]


def fit_attack_defense(train, half_life=900.0):
    teams = sorted(set(train.home_team) | set(train.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    H = train.home_team.map(idx).values; A = train.away_team.map(idx).values
    hs = train.home_score.values.astype(float); as_ = train.away_score.values.astype(float)
    neutral = train.neutral.values
    # peso temporale
    cut = train.date.values.max()
    age = (cut - train.date.values) / np.timedelta64(1, "D")
    w1 = train.weight.values * 0.5 ** (age / half_life)

    rows = len(train)
    # due osservazioni per match: gol casa e gol trasferta
    # design: [attacco(n)] [difesa(n)] [home_flag]
    def block(att_team, def_team, home):
        r = np.arange(rows)
        att = csr_matrix((np.ones(rows), (r, att_team)), shape=(rows, n))
        dfn = csr_matrix((np.ones(rows), (r, def_team)), shape=(rows, n))
        hf = csr_matrix(home.reshape(-1, 1).astype(float))
        return hstack([att, dfn, hf]).tocsr()
    Xh = block(H, A, np.where(neutral, 0.0, 1.0))
    Xa = block(A, H, np.zeros(rows))
    X = csr_matrix(np.vstack([Xh.toarray(), Xa.toarray()])) if rows < 1 else _vstack(Xh, Xa)
    y = np.concatenate([hs, as_]); w = np.concatenate([w1, w1])
    m = PoissonRegressor(alpha=1.0, max_iter=400, fit_intercept=True)
    m.fit(X, y, sample_weight=w)
    coef = m.coef_
    att = coef[:n]; dfn = coef[n:2*n]; hadv = coef[2*n]
    return {"idx": idx, "att": att, "dfn": dfn, "hadv": hadv, "b0": m.intercept_}


def _vstack(a, b):
    from scipy.sparse import vstack as sv
    return sv([a, b]).tocsr()


def ad_probs(model, h, a, neutral, rho=-0.05, max_goals=10):
    idx = model["idx"]
    if h not in idx or a not in idx:
        return None
    lh = np.exp(model["b0"] + model["att"][idx[h]] + model["dfn"][idx[a]] + (0 if neutral else model["hadv"]))
    la = np.exp(model["b0"] + model["att"][idx[a]] + model["dfn"][idx[h]])
    lh = min(lh, 12); la = min(la, 12)
    ph = pois.pmf(np.arange(max_goals + 1), lh); pa = pois.pmf(np.arange(max_goals + 1), la)
    mat = np.outer(ph, pa)
    mat[0, 0] *= 1 - lh * la * rho; mat[0, 1] *= 1 + lh * rho
    mat[1, 0] *= 1 + la * rho; mat[1, 1] *= 1 - rho
    mat = np.clip(mat, 0, None); mat /= mat.sum()
    return float(np.tril(mat, -1).sum()), float(np.trace(mat)), float(np.triu(mat, 1).sum())


def run():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, _ = compute_elo(m)
    pe, pad, outs, yrs = [], [], [], []
    for year in BACKTEST_WORLD_CUPS:
        cutoff = pd.Timestamp(WC_START_DATES[year])
        train = mm[(mm.date < cutoff) & (mm.date >= cutoff - pd.Timedelta(days=365 * 12))].copy()
        params = fit(train, cutoff)
        admodel = fit_attack_defense(train)
        wc = mm[(mm.tournament == "FIFA World Cup") & (mm.date >= cutoff)
                & (mm.date < cutoff + pd.Timedelta(days=60))]
        for _, r in wc.iterrows():
            (p1, px, p2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, params)
            adp = ad_probs(admodel, r.home_team, r.away_team, True)
            if adp is None: continue
            pe.append([p1, px, p2]); pad.append(list(adp))
            outs.append(M.outcome_index(r.home_score, r.away_score)); yrs.append(year)
        print(f"[attdef] {year} fatto", file=sys.stderr)
    pe = np.array(pe); pad = np.array(pad); O = np.array(outs); Y = np.array(yrs)
    tem = np.isin(Y, TEST); full = np.ones(len(O), bool)
    # ensemble Elo-Poisson + att/def
    ens = (pe + pad) / 2; ens /= ens.sum(1, keepdims=True)

    def show(label, P, mask):
        print(f"{label:26}{M.log_loss(P[mask],O[mask]):9.4f}{M.rps(P[mask],O[mask]):8.4f}{M.accuracy(P[mask],O[mask]):7.3f}")
    print(f"\n{'modello':26}{'logloss':>9}{'rps':>8}{'acc':>7}")
    print("--- TEST 2018-2022 ---")
    show("Elo-Poisson (mio)", pe, tem); show("Attacco/Difesa", pad, tem); show("ensemble dei due", ens, tem)
    print("--- POOLED 2002-2022 ---")
    show("Elo-Poisson (mio)", pe, full); show("Attacco/Difesa", pad, full); show("ensemble dei due", ens, full)


if __name__ == "__main__":
    run()
