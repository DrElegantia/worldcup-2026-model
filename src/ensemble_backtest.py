"""Fase 2: confronto onesto baseline (Elo-Poisson) vs XGBoost vs ensemble.

Per ogni Mondiale 2002-2022 (point-in-time):
  - feature engineering leakage-free (features.py)
  - XGBoost addestrato sui soli match anteriori al cutoff
  - probabilita Poisson-da-Elo (baseline) sugli stessi match
  - ensemble = blend calibrato delle due

Selezione del peso ensemble OUT-OF-SAMPLE: peso scelto sui Mondiali 2002-2014,
valutato su 2018-2022. Si tiene l'ensemble solo se batte il baseline sul test.
"""
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from config import BACKTEST_WORLD_CUPS, WC_START_DATES, DATA_PROC
from elo import compute_elo
from poisson import fit, match_probs
from features import build_match_features, FEATURE_COLS
import metrics as M


def collect():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, _ = compute_elo(m)
    print("[fase2] feature engineering...", file=sys.stderr)
    fm = build_match_features(mm)

    rows = []
    for year in BACKTEST_WORLD_CUPS:
        cutoff = pd.Timestamp(WC_START_DATES[year])
        train = fm[(fm.date < cutoff) & (fm.date >= cutoff - pd.Timedelta(days=365 * 18))].copy()
        train = train.dropna(subset=FEATURE_COLS)
        y = train.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
        X = train[FEATURE_COLS].values
        w = train["weight"].values.astype(float)

        clf = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, max_depth=3,
            n_estimators=300, learning_rate=0.04, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
            eval_metric="mlogloss", verbosity=0)
        clf.fit(X, y, sample_weight=w)

        params = fit(train, cutoff)
        wc = fm[(fm.tournament == "FIFA World Cup") & (fm.date >= cutoff)
                & (fm.date < cutoff + pd.Timedelta(days=60))].copy()
        Xwc = wc[FEATURE_COLS].values
        pxgb = clf.predict_proba(Xwc)
        for k, (_, r) in enumerate(wc.iterrows()):
            (p1, px, p2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, params)
            rows.append({"year": year,
                         "po": [p1, px, p2], "xg": list(pxgb[k]),
                         "out": M.outcome_index(r.home_score, r.away_score)})
        print(f"[fase2] {year}: {len(wc)} match, train {len(train)}", file=sys.stderr)
    return pd.DataFrame(rows)


def blend(po, xg, w):
    b = (1 - w) * np.array(po) + w * np.array(xg)
    return b / b.sum(axis=1, keepdims=True)


def evaluate(df, idx, probs):
    sub_out = df.loc[idx, "out"].values
    return {"logloss": M.log_loss(probs, sub_out),
            "rps": M.rps(probs, sub_out),
            "acc": M.accuracy(probs, sub_out)}


def run():
    df = collect()
    po = np.array(df["po"].tolist())
    xg = np.array(df["xg"].tolist())
    out = df["out"].values
    train_mask = df["year"].isin([2002, 2006, 2010, 2014]).values
    test_mask = df["year"].isin([2018, 2022]).values

    # scegli il peso sul train (2002-2014). Cap conservativo a 0.40: l'Elo e' forte
    # e il campione WC e' piccolo, evitiamo che il GB domini per overfit.
    weights = np.linspace(0, 0.40, 9)  # cap conservativo
    best_w, best_ll = 0.0, 1e9
    for w in weights:
        b = blend(po[train_mask], xg[train_mask], w)
        ll = M.log_loss(b, out[train_mask])
        if ll < best_ll:
            best_ll, best_w = ll, w

    def metr(mask, probs):
        return (M.log_loss(probs, out[mask]), M.rps(probs, out[mask]), M.accuracy(probs, out[mask]))

    print("\n=== FASE 2: confronto (peso ensemble scelto su 2002-2014) ===")
    print(f"peso ensemble ottimale (train): w={best_w:.2f}  (0=solo Elo-Poisson, 1=solo XGBoost)\n")
    print(f"{'modello':24}{'logloss':>10}{'rps':>9}{'acc':>8}   set")
    for name, mask, label in [("baseline Elo-Poisson", test_mask, "TEST 2018-2022"),
                              ("XGBoost", test_mask, "TEST 2018-2022"),
                              (f"ensemble (w={best_w:.2f})", test_mask, "TEST 2018-2022")]:
        if name.startswith("baseline"): p = po[mask]
        elif name.startswith("XGBoost"): p = xg[mask]
        else: p = blend(po[mask], xg[mask], best_w)
        ll, rps, acc = metr(mask, p)
        print(f"{name:24}{ll:10.4f}{rps:9.4f}{acc:8.3f}   {label}")

    print()
    result = {"weight": round(float(best_w), 2), "rows": []}
    pooled = {}
    for name, key in [("baseline Elo-Poisson", "baseline"), ("XGBoost", "xgboost"),
                      (f"ensemble (w={best_w:.2f})", "ensemble")]:
        if key == "baseline": p_all, p_te = po, po[test_mask]
        elif key == "xgboost": p_all, p_te = xg, xg[test_mask]
        else: p_all, p_te = blend(po, xg, best_w), blend(po[test_mask], xg[test_mask], best_w)
        ll, rps, acc = metr(np.ones(len(out), bool), p_all)
        llt, rpst, acct = metr(test_mask, p_te)
        print(f"{name:24}{ll:10.4f}{rps:9.4f}{acc:8.3f}   POOLED 2002-2022")
        result["rows"].append({"model": name, "key": key,
                               "pooled": {"logloss": round(ll, 4), "rps": round(rps, 4), "acc": round(acc, 4)},
                               "test": {"logloss": round(llt, 4), "rps": round(rpst, 4), "acc": round(acct, 4)}})
    import json
    with open(DATA_PROC / "phase2.json", "w") as f:
        json.dump(result, f, indent=2)
    return best_w


if __name__ == "__main__":
    run()
