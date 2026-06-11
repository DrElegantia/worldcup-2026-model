"""Esperimento: feature di STILE + interazione (non-transitivita) migliorano?

Possesso/PPDA per nazionale point-in-time pre-2018 non sono reperibili da fonti
aperte. Si usano proxy di stile derivate dai risultati (orientamento attacco vs
difesa: 'tiki-taka' = molti gol fatti, 'catenaccio' = pochi gol fatti e subiti)
e, soprattutto, TERMINI DI INTERAZIONE fra gli stili delle due squadre, che e'
cio che l'Elo (forza scalare) non puo' rappresentare.

Confronto su WC 2018/2022: XGB base vs XGB + stile, in ensemble con Elo-Poisson.
"""
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from config import DATA_PROC, WC_START_DATES
from elo import compute_elo
from poisson import fit, match_probs
from features import build_match_features, FEATURE_COLS
import metrics as M

STYLE_COLS = ["s_h", "s_a", "s_clash", "s_dist", "ad_h", "ad_a"]


def add_style(fm):
    fm = fm.copy()
    fm["s_h"] = fm["gf_h"] - fm["ga_h"]          # orientamento: +att / -dif
    fm["s_a"] = fm["gf_a"] - fm["ga_a"]
    fm["s_clash"] = fm["s_h"] * fm["s_a"]         # interazione stili (non-transitiva)
    fm["s_dist"] = (fm["s_h"] - fm["s_a"]).abs()  # distanza di stile
    fm["ad_h"] = fm["gf_h"] * fm["ga_a"]          # attacco A contro difesa B
    fm["ad_a"] = fm["gf_a"] * fm["ga_h"]          # attacco B contro difesa A
    return fm


def run():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, _ = compute_elo(m)
    fm = add_style(build_match_features(mm))
    fm = fm[fm.date >= pd.Timestamp("2014-01-01")]

    BASE = FEATURE_COLS
    STY = FEATURE_COLS + STYLE_COLS

    def xgb_fit(tr, cols):
        tr = tr.dropna(subset=cols)
        y = tr.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
        clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, max_depth=3,
                                n_estimators=300, learning_rate=0.04, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                                eval_metric="mlogloss", verbosity=0)
        clf.fit(tr[cols].values, y, sample_weight=tr["weight"].values.astype(float))
        return clf

    pb, ps, outs = [], [], []
    for year in [2018, 2022]:
        cutoff = pd.Timestamp(WC_START_DATES[year])
        tr = fm[fm.date < cutoff]
        clf_b = xgb_fit(tr, BASE); clf_s = xgb_fit(tr, STY)
        wc = fm[(fm.tournament == "FIFA World Cup") & (fm.date >= cutoff)
                & (fm.date < cutoff + pd.Timedelta(days=60))].dropna(subset=STY)
        b = clf_b.predict_proba(wc[BASE].values)
        s = clf_s.predict_proba(wc[STY].values)
        params = fit(tr, cutoff)
        for k, (_, r) in enumerate(wc.iterrows()):
            (p1, px, p2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, params)
            base = 0.7 * np.array([p1, px, p2]) + 0.3 * b[k]
            sty = 0.7 * np.array([p1, px, p2]) + 0.3 * s[k]
            pb.append(base / base.sum()); ps.append(sty / sty.sum())
            outs.append(M.outcome_index(r.home_score, r.away_score))
        imp = dict(zip(STY, clf_s.feature_importances_))
        print(f"[{year}] importanza stile: " +
              ", ".join(f"{c}={imp[c]:.3f}" for c in STYLE_COLS), file=sys.stderr)

    B = np.array(pb); S = np.array(ps); O = np.array(outs)
    print(f"\n=== WC 2018+2022 ({len(O)} partite) ===")
    print(f"{'modello':30}{'logloss':>9}{'rps':>8}{'acc':>7}")
    print(f"{'ensemble SENZA stile':30}{M.log_loss(B,O):9.4f}{M.rps(B,O):8.4f}{M.accuracy(B,O):7.3f}")
    print(f"{'ensemble CON stile+interazione':30}{M.log_loss(S,O):9.4f}{M.rps(S,O):8.4f}{M.accuracy(S,O):7.3f}")


if __name__ == "__main__":
    run()
