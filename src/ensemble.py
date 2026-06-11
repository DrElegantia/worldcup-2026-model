"""Ensemble di produzione: Elo-Poisson + XGBoost (peso validato sul backtest).

Il peso ENSEMBLE_W e' stato scelto out-of-sample (train 2002-2014, test 2018-2022)
in ensemble_backtest.py. L'ensemble batte il baseline su log loss e RPS.
"""
import numpy as np
import xgboost as xgb

from config import ELO_HOME_ADV
from features import build_match_features, FEATURE_COLS, team_snapshots
from poisson import match_probs
import metrics as M

ENSEMBLE_W = 0.30   # 0 = solo Elo-Poisson, 1 = solo XGBoost (validato sul backtest)


def train_xgb(mm, cutoff):
    fm = build_match_features(mm)
    import pandas as pd
    train = fm[(fm.date < pd.Timestamp(cutoff))
               & (fm.date >= pd.Timestamp(cutoff) - pd.Timedelta(days=365 * 18))].copy()
    train = train.dropna(subset=FEATURE_COLS)
    y = train.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, max_depth=3,
                            n_estimators=300, learning_rate=0.04, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                            eval_metric="mlogloss", verbosity=0)
    clf.fit(train[FEATURE_COLS].values, y, sample_weight=train["weight"].values.astype(float))
    return clf


def _feat_vector(snap, h, a, neutral):
    sh, sa = snap[h], snap[a]
    adv = 0.0 if neutral else ELO_HOME_ADV
    return [[
        sh["elo"] - sa["elo"] + adv,            # d_elo
        sh["ppg"] - sa["ppg"],                  # d_ppg
        sh["gf"] - sa["gf"],                    # d_gf
        sh["ga"] - sa["ga"],                    # d_ga
        sh["rest"] - sa["rest"],                # d_rest
        sh["m90"] - sa["m90"],                  # d_m90
        sh["mom"] - sa["mom"],                  # d_mom
        0.0 if neutral else 1.0,                # home_flag
        1.0,                                    # is_wc
        sh["gf"], sh["ga"], sa["gf"], sa["ga"],
    ]]


def ensemble_match_probs(h, a, neutral, params, clf, snap, w=ENSEMBLE_W,
                         alt_pen_h=0.0, alt_pen_a=0.0):
    """Probabilita 1X2 ensemble + lambda Poisson (per il risultato modale)."""
    (p1, px, p2), (lh, la), mat = match_probs(snap[h]["elo"], snap[a]["elo"], neutral,
                                              params, alt_pen_h=alt_pen_h, alt_pen_a=alt_pen_a)
    if clf is not None and h in snap and a in snap:
        try:
            px_xgb = clf.predict_proba(np.array(_feat_vector(snap, h, a, neutral)))[0]
            b = (1 - w) * np.array([p1, px, p2]) + w * px_xgb
            b = b / b.sum()
            return (float(b[0]), float(b[1]), float(b[2])), (lh, la), mat
        except Exception:
            pass
    return (p1, px, p2), (lh, la), mat
