"""Test onesto: la forza-rosa EA FIFA aggiunge segnale oltre Elo+forma?

Valuta su WC 2018 e WC 2022 (point-in-time), confrontando l'XGB con e senza la
feature differenziale di forza-rosa.
"""
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from config import DATA_PROC, WC_START_DATES
from elo import compute_elo
from poisson import fit, match_probs
from features import build_match_features, FEATURE_COLS
import squad
import metrics as M


def add_squad(fm, strength):
    def s_of(team, year):
        return strength.get(squad.edition_for_year(year), {}).get(team, np.nan)
    yrs = fm["date"].dt.year.values
    sh = np.array([s_of(t, y) for t, y in zip(fm["home_team"].values, yrs)])
    sa = np.array([s_of(t, y) for t, y in zip(fm["away_team"].values, yrs)])
    fm = fm.copy()
    fm["squad_h"] = sh; fm["squad_a"] = sa
    fm["d_squad"] = sh - sa
    return fm


def run():
    strength = squad.load()
    if not strength:
        print("manca squad_strength.json: lancia squad.py", file=sys.stderr); return
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, _ = compute_elo(m)
    fm = build_match_features(mm)
    fm = add_squad(fm, strength)
    fm = fm[fm.date >= pd.Timestamp("2014-01-01")]   # finestra con copertura forza-rosa

    COLS_BASE = FEATURE_COLS
    COLS_SQ = FEATURE_COLS + ["d_squad", "squad_h", "squad_a"]

    def xgb_fit(tr, cols):
        tr = tr.dropna(subset=cols)
        y = tr.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
        clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, max_depth=3,
                                n_estimators=300, learning_rate=0.04, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                                eval_metric="mlogloss", verbosity=0)
        clf.fit(tr[cols].values, y, sample_weight=tr["weight"].values.astype(float))
        return clf

    probs_b, probs_s, outs = [], [], []
    for year in [2018, 2022]:
        cutoff = pd.Timestamp(WC_START_DATES[year])
        tr = fm[fm.date < cutoff]
        clf_b = xgb_fit(tr, COLS_BASE)
        clf_s = xgb_fit(tr, COLS_SQ)
        wc = fm[(fm.tournament == "FIFA World Cup") & (fm.date >= cutoff)
                & (fm.date < cutoff + pd.Timedelta(days=60))].copy()
        wc_b = wc.dropna(subset=COLS_BASE)
        pb = clf_b.predict_proba(wc_b[COLS_BASE].values)
        # per la versione squad, riempi d_squad mancanti con 0 (no info)
        wcs = wc.copy()
        for c in ["d_squad", "squad_h", "squad_a"]:
            wcs[c] = wcs[c].fillna(wcs[c].median() if c != "d_squad" else 0.0)
        ps = clf_s.predict_proba(wcs[COLS_SQ].values)
        params = fit(tr, cutoff)
        for k, (_, r) in enumerate(wc_b.iterrows()):
            (p1, px, p2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, params)
            o = M.outcome_index(r.home_score, r.away_score)
            # ensemble Elo-Poisson + XGB (peso 0.30, come produzione)
            base = 0.7 * np.array([p1, px, p2]) + 0.3 * pb[k]
            sq = 0.7 * np.array([p1, px, p2]) + 0.3 * ps[k]
            probs_b.append(base / base.sum()); probs_s.append(sq / sq.sum()); outs.append(o)
        # feature importance squad
        imp = dict(zip(COLS_SQ, clf_s.feature_importances_))
        print(f"[{year}] importanza d_squad={imp['d_squad']:.3f} squad_h={imp['squad_h']:.3f}",
              file=sys.stderr)

    B = np.array(probs_b); S = np.array(probs_s); O = np.array(outs)
    print(f"\n=== WC 2018+2022 ({len(O)} partite) ===")
    print(f"{'modello':28}{'logloss':>9}{'rps':>8}{'acc':>7}")
    print(f"{'ensemble SENZA forza-rosa':28}{M.log_loss(B,O):9.4f}{M.rps(B,O):8.4f}{M.accuracy(B,O):7.3f}")
    print(f"{'ensemble CON forza-rosa':28}{M.log_loss(S,O):9.4f}{M.rps(S,O):8.4f}{M.accuracy(S,O):7.3f}")


if __name__ == "__main__":
    run()
