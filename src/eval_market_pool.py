"""Il mercato come 6o modello nel pool + re-selezione.

Trasforma le probabilita-titolo di consenso bookmaker in un modello a livello
partita (forza per squadra dalle prob-titolo -> 1X2), lo aggiunge al pool degli
altri modelli e RIFA la forward-selection OOS. Cosi' si verifica:
  (a) il mercato si guadagna un posto nel consenso?
  (b) con il mercato dentro, uno dei modelli scartati prima (Results-Elo,
      XGBoost) torna utile?

Solo 4 Mondiali (2010-2022) hanno dati di quote: leave-one-Mondiale-out = 4 fold.
"""
import csv
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from config import DATA_PROC, WC_START_DATES
from elo import compute_elo
from poisson import fit, match_probs, score_matrix, outcome_probs
from features import build_match_features, FEATURE_COLS
from eval_attdef import fit_attack_defense, ad_probs
from eval_npool import fit_massey, massey_probs, softmax, opt_weights, LOGIT_COLS
import metrics as M

NAMES = ["Elo-Poisson", "Attacco/Difesa", "Forma", "Massey", "Logistica",
         "Results-Elo", "XGBoost", "Mercato"]
YEARS = [2010, 2014, 2018, 2022]


def load_market():
    M_ = {}
    for row in csv.DictReader(open(DATA_PROC.parent / "market" / "market_winprob.csv")):
        M_.setdefault(int(row["year"]), {})[row["team"]] = float(row["market_prob_pct"]) / 100.0
    return M_


def market_ratings(probs):
    """Prob-titolo -> forza per squadra (log-prob standardizzato a scala-gol)."""
    teams = list(probs); lp = np.array([np.log(max(probs[t], 1e-5)) for t in teams])
    lp = lp - lp.mean()
    sc = 1.2 / (lp.std() + 1e-9)            # std ~ scala Massey (unita gol)
    return {t: float(lp[i] * sc) for i, t in enumerate(teams)}


def collect():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, _ = compute_elo(m)
    mm_ro, _, _ = compute_elo(m, ignore_margin=True)
    print("[mktpool] feature engineering...", file=sys.stderr)
    fm = build_match_features(mm)
    fm["pe_h_ro"] = mm_ro["pre_elo_home"].values
    fm["pe_a_ro"] = mm_ro["pre_elo_away"].values
    MK = load_market()

    rows = []
    for year in YEARS:
        cutoff = pd.Timestamp(WC_START_DATES[year])
        tr = fm[(fm.date < cutoff) & (fm.date >= cutoff - pd.Timedelta(days=365 * 14))].copy()
        trc = tr.dropna(subset=FEATURE_COLS + LOGIT_COLS)
        params = fit(tr, cutoff); rho = params["rho"]
        admodel = fit_attack_defense(tr); mas = fit_massey(tr)
        mkr = market_ratings(MK[year])
        y = trc.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
        clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, max_depth=3,
                                n_estimators=300, learning_rate=0.04, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                                eval_metric="mlogloss", verbosity=0)
        clf.fit(trc[FEATURE_COLS].values, y, sample_weight=trc["weight"].values.astype(float))
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(trc[LOGIT_COLS].values, y, sample_weight=trc["weight"].values.astype(float))

        wc = fm[(fm.tournament == "FIFA World Cup") & (fm.date >= cutoff)
                & (fm.date < cutoff + pd.Timedelta(days=60))].dropna(subset=FEATURE_COLS + LOGIT_COLS)
        xgp = clf.predict_proba(wc[FEATURE_COLS].values); lgp = lr.predict_proba(wc[LOGIT_COLS].values)
        for k, (_, r) in enumerate(wc.iterrows()):
            ad = ad_probs(admodel, r.home_team, r.away_team, True)
            ma = massey_probs(mas, r.home_team, r.away_team, rho)
            if ad is None or ma is None:
                continue
            elo = list(match_probs(r.pre_elo_home, r.pre_elo_away, True, params)[0])
            ro = list(match_probs(r.pe_h_ro, r.pe_a_ro, True, params)[0])
            form = list(outcome_probs(score_matrix(min((r.gf_h + r.ga_a) / 2, 12),
                                                   min((r.gf_a + r.ga_h) / 2, 12), rho)))
            gh, ga = mkr.get(r.home_team), mkr.get(r.away_team)
            if gh is None or ga is None:
                mkt = elo                              # squadra senza quota: fallback Elo
            else:
                gd = gh - ga
                mkt = list(outcome_probs(score_matrix(min(max((2.6 + gd) / 2, .2), 12),
                                                      min(max((2.6 - gd) / 2, .2), 12), rho)))
            rows.append({"year": year, "o": M.outcome_index(r.home_score, r.away_score),
                         "P": [elo, list(ad), form, list(ma), list(lgp[k]), ro, list(xgp[k]), mkt]})
        print(f"[mktpool] {year} fatto", file=sys.stderr)
    df = pd.DataFrame(rows)
    return np.array(df["P"].tolist()), df["o"].values, df["year"].values


def oos(idx, P, O, Y):
    sub = P[:, idx, :]; probs = np.zeros((len(O), 3))
    for h in np.unique(Y):
        tr = Y != h; te = Y == h
        w = opt_weights(sub[tr], O[tr])
        probs[te] = (sub[te] * w[None, :, None]).sum(1)
    probs /= probs.sum(1, keepdims=True)
    return M.log_loss(probs, O), probs


def run():
    P, O, Y = collect()
    print(f"\nPartite (4 WC con quote): {len(O)} | modelli: {len(NAMES)}")
    sel, rem = [], list(range(len(NAMES)))
    print(f"\n{'N':>2}  {'modello aggiunto':16}{'logloss OOS':>12}")
    prev = None
    for _ in range(len(NAMES)):
        best = None
        for c in rem:
            ll, _ = oos(sel + [c], P, O, Y)
            if best is None or ll < best[0]: best = (ll, c)
        sel.append(best[1]); rem.remove(best[1])
        d = "" if prev is None else f"  (Δ {best[0]-prev:+.4f})"
        print(f"{len(sel):>2}  {NAMES[best[1]]:16}{best[0]:12.4f}{d}"); prev = best[0]
    # peso del consenso completo + dove sta il mercato
    w = opt_weights(P, O)
    print("\npesi consenso (tutti i modelli):")
    for n, wi in sorted(zip(NAMES, w), key=lambda x: -x[1]):
        print(f"   {n:16}{wi:.3f}")


if __name__ == "__main__":
    run()
