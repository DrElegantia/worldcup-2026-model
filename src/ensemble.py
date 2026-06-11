"""Ensemble di produzione: Elo-Poisson + XGBoost (peso validato sul backtest).

Il peso ENSEMBLE_W e' stato scelto out-of-sample (train 2002-2014, test 2018-2022)
in ensemble_backtest.py. L'ensemble batte il baseline su log loss e RPS.
"""
import numpy as np
import xgboost as xgb

from config import ELO_HOME_ADV
from features import build_match_features, FEATURE_COLS, team_snapshots
from poisson import match_probs
import squad as squadmod
import metrics as M

ENSEMBLE_W = 0.30   # 0 = solo Elo-Poisson, 1 = solo XGBoost (validato sul backtest)
# feature di produzione: base + forza-rosa EA FIFA (validata su WC 2018/2022)
FEATURE_COLS_PROD = FEATURE_COLS + squadmod.SQUAD_COLS


def train_xgb(mm, cutoff):
    import pandas as pd
    fm = build_match_features(mm)
    strength = squadmod.load()
    cols = FEATURE_COLS
    if strength:
        fm = squadmod.attach_squad(fm, strength)
        cols = FEATURE_COLS_PROD
    # finestra con copertura forza-rosa (2014+) se disponibile, altrimenti 18 anni
    lo = pd.Timestamp("2014-01-01") if strength else pd.Timestamp(cutoff) - pd.Timedelta(days=365 * 18)
    train = fm[(fm.date < pd.Timestamp(cutoff)) & (fm.date >= lo)].copy()
    train = train.dropna(subset=cols)
    y = train.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, max_depth=3,
                            n_estimators=300, learning_rate=0.04, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                            eval_metric="mlogloss", verbosity=0)
    clf.fit(train[cols].values, y, sample_weight=train["weight"].values.astype(float))
    clf._cols = cols
    return clf


def _feat_vector(snap, h, a, neutral, sq_h=None, sq_a=None, with_squad=False):
    sh, sa = snap[h], snap[a]
    adv = 0.0 if neutral else ELO_HOME_ADV
    v = [sh["elo"] - sa["elo"] + adv, sh["ppg"] - sa["ppg"], sh["gf"] - sa["gf"],
         sh["ga"] - sa["ga"], sh["rest"] - sa["rest"], sh["m90"] - sa["m90"],
         sh["mom"] - sa["mom"], 0.0 if neutral else 1.0, 1.0,
         sh["gf"], sh["ga"], sa["gf"], sa["ga"]]
    if with_squad:
        N = squadmod.NEUTRAL
        ssh = N if sq_h is None else sq_h
        ssa = N if sq_a is None else sq_a
        dsq = 0.0 if (sq_h is None or sq_a is None) else (sq_h - sq_a)
        v += [dsq, ssh, ssa]
    return [v]


def ensemble_match_probs(h, a, neutral, params, clf, snap, w=ENSEMBLE_W,
                         alt_pen_h=0.0, alt_pen_a=0.0, sq_h=None, sq_a=None):
    """Probabilita 1X2 ensemble + lambda Poisson (per il risultato modale)."""
    (p1, px, p2), (lh, la), mat = match_probs(snap[h]["elo"], snap[a]["elo"], neutral,
                                              params, alt_pen_h=alt_pen_h, alt_pen_a=alt_pen_a)
    if clf is not None and h in snap and a in snap:
        try:
            with_sq = getattr(clf, "_cols", FEATURE_COLS) == FEATURE_COLS_PROD
            fv = np.array(_feat_vector(snap, h, a, neutral, sq_h, sq_a, with_sq))
            px_xgb = clf.predict_proba(fv)[0]
            b = (1 - w) * np.array([p1, px, p2]) + w * px_xgb
            b = b / b.sum()
            return (float(b[0]), float(b[1]), float(b[2])), (lh, la), mat
        except Exception:
            pass
    return (p1, px, p2), (lh, la), mat
