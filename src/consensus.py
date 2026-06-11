"""Consenso di modelli: la 'maggioranza' di piu' modelli validi decide.

Per ogni partita si mediano le probabilita 1X2 di tre modelli:
  - Elo-Poisson Dixon-Coles  (peso 0.6)
  - XGBoost su feature        (peso 0.2)
  - Attacco/Difesa per squadra (peso 0.2)
Pesi validati out-of-sample sui Mondiali 2018/2022 (vedi eval_consensus.py).

Per il Monte Carlo:
  - gironi: si invertono le probabilita di consenso in lambda di gol (per GD/GF);
  - knockout: tabella 48x48 di P(vince) di consenso, usata direttamente.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize_scalar

from config import ELO_HOME_ADV
from poisson import match_probs, score_matrix, outcome_probs
from features import build_match_features, FEATURE_COLS
from eval_attdef import fit_attack_defense, ad_probs
import metrics as M

W_ELO, W_XGB, W_AD = 0.6, 0.2, 0.2


def train(mm, cutoff):
    cutoff = pd.Timestamp(cutoff)
    fm = build_match_features(mm)
    tr = fm[(fm.date < cutoff) & (fm.date >= cutoff - pd.Timedelta(days=365 * 16))].dropna(subset=FEATURE_COLS)
    y = tr.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, max_depth=3,
                            n_estimators=300, learning_rate=0.04, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                            eval_metric="mlogloss", verbosity=0)
    clf.fit(tr[FEATURE_COLS].values, y, sample_weight=tr["weight"].values.astype(float))
    ad = fit_attack_defense(mm[(mm.date < cutoff) & (mm.date >= cutoff - pd.Timedelta(days=365 * 12))])
    return {"xgb": clf, "ad": ad}


def _xgb_vec(snap, h, a, neutral):
    sh, sa = snap[h], snap[a]
    adv = 0.0 if neutral else ELO_HOME_ADV
    return [sh["elo"] - sa["elo"] + adv, sh["ppg"] - sa["ppg"], sh["gf"] - sa["gf"],
            sh["ga"] - sa["ga"], sh["rest"] - sa["rest"], sh["m90"] - sa["m90"],
            sh["mom"] - sa["mom"], 0.0 if neutral else 1.0, 1.0,
            sh["gf"], sh["ga"], sa["gf"], sa["ga"]]


def consensus_1x2(h, a, neutral, params, models, snap, alt_pen_h=0.0, alt_pen_a=0.0):
    (e1, ex, e2), (lh, la), _ = match_probs(snap[h]["elo"], snap[a]["elo"], neutral, params,
                                            alt_pen_h=alt_pen_h, alt_pen_a=alt_pen_a)
    base = np.array([e1, ex, e2])
    out = W_ELO * base
    try:
        out = out + W_XGB * models["xgb"].predict_proba(np.array([_xgb_vec(snap, h, a, neutral)]))[0]
    except Exception:
        out = out + W_XGB * base
    adp = ad_probs(models["ad"], h, a, neutral)
    out = out + W_AD * (np.array(adp) if adp else base)
    return out / out.sum(), (lh, la)


def consensus_lambda(h, a, neutral, params, models, snap, alt_pen_h=0.0, alt_pen_a=0.0):
    """Lambda di gol che riproducono il consenso 1X2 (per il girone)."""
    c, (lh0, la0) = consensus_1x2(h, a, neutral, params, models, snap, alt_pen_h, alt_pen_a)
    rho = params["rho"]

    def loss(t):
        lh = float(np.clip(lh0 * np.exp(t), 1e-3, 12))
        la = float(np.clip(la0 * np.exp(-t), 1e-3, 12))
        p = outcome_probs(score_matrix(lh, la, rho))
        return (p[0] - c[0]) ** 2 + (p[2] - c[2]) ** 2

    t = minimize_scalar(loss, bounds=(-0.8, 0.8), method="bounded").x
    return (float(np.clip(lh0 * np.exp(t), 1e-3, 12)),
            float(np.clip(la0 * np.exp(-t), 1e-3, 12)))


def knockout_table(teams, params, models, snap):
    """48x48: P(i avanza su j) di consenso a campo neutro (pareggi -> rigori split)."""
    n = len(teams)
    # feature XGB per tutte le coppie, una sola predizione batch
    vecs, pairs = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vecs.append(_xgb_vec(snap, teams[i], teams[j], True)); pairs.append((i, j))
    xgp = models["xgb"].predict_proba(np.array(vecs))
    adv = np.full((n, n), 0.5)
    for k, (i, j) in enumerate(pairs):
        (e1, ex, e2), _, _ = match_probs(snap[teams[i]]["elo"], snap[teams[j]]["elo"], True, params)
        base = np.array([e1, ex, e2])
        adp = ad_probs(models["ad"], teams[i], teams[j], True)
        c = W_ELO * base + W_XGB * xgp[k] + W_AD * (np.array(adp) if adp else base)
        c = c / c.sum()
        denom = c[0] + c[2] if (c[0] + c[2]) > 0 else 1.0
        adv[i, j] = c[0] + c[1] * (c[0] / denom)     # vince + meta pareggi pesata
    return adv
