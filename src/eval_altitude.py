"""Test onesto: le feature di altitudine migliorano la predizione?

Split temporale su TUTTE le partite internazionali (l'altitudine conta soprattutto
nelle qualificazioni sudamericane/CONCACAF, non nei pochi WC ad alta quota).
Confronta XGB senza vs con feature altitudine, su tutto il test e sul sottoinsieme
ad alta quota.
"""
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from config import DATA_PROC
from elo import compute_elo
from features import build_match_features, FEATURE_COLS, FEATURE_COLS_ALT
import metrics as M


def _train_eval(tr, te, cols):
    ytr = tr.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
    yte = te.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, max_depth=3,
                            n_estimators=300, learning_rate=0.04, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                            eval_metric="mlogloss", verbosity=0)
    clf.fit(tr[cols].values, ytr, sample_weight=tr["weight"].values.astype(float))
    p = clf.predict_proba(te[cols].values)
    return clf, p, yte


def run():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, _ = compute_elo(m)
    print("[alt] feature engineering (incl. altitudine)...", file=sys.stderr)
    import geo
    fm = build_match_features(mm)
    fm, _ = geo.add_elevation(fm)
    fm = fm.dropna(subset=FEATURE_COLS_ALT)

    split = pd.Timestamp("2016-01-01")
    tr = fm[fm.date < split]
    te = fm[fm.date >= split]
    print(f"train {len(tr)}  test {len(te)}")

    _, p_base, y = _train_eval(tr, te, FEATURE_COLS)
    clf_alt, p_alt, _ = _train_eval(tr, te, FEATURE_COLS_ALT)

    def rep(name, p, mask=None):
        if mask is None:
            mask = np.ones(len(y), bool)
        ll = M.log_loss(p[mask], y[mask]); rps = M.rps(p[mask], y[mask])
        acc = M.accuracy(p[mask], y[mask])
        print(f"{name:28}{ll:9.4f}{rps:8.4f}{acc:7.3f}  (n={mask.sum()})")

    print(f"\n{'modello':28}{'logloss':>9}{'rps':>8}{'acc':>7}")
    print("--- TUTTO il test (2016+) ---")
    rep("senza altitudine", p_base)
    rep("con altitudine", p_alt)

    hi = (te["venue_alt"].values > 1500)
    print("--- solo partite ad alta quota (>1500m) ---")
    rep("senza altitudine", p_base, hi)
    rep("con altitudine", p_alt, hi)

    # importanza feature
    imp = dict(zip(FEATURE_COLS_ALT, clf_alt.feature_importances_))
    print("\nimportanza feature altitudine:",
          {k: round(float(imp[k]), 4) for k in ["d_alt_pen", "venue_alt"]})
    rank = sorted(imp.items(), key=lambda x: -x[1])
    print("top 6 feature:", [(k, round(float(v), 3)) for k, v in rank[:6]])


if __name__ == "__main__":
    run()
