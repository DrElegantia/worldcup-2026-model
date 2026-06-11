"""Scaling del consenso: quanti modelli servono davvero?

Pool di 7 modelli con IPOTESI/STRATEGIE diverse (non riparametrizzazioni):
  1 Elo-Poisson    (rating scalare iterativo, il margine conta)
  2 Attacco/Difesa (abilita 2D separate per squadra)
  3 Forma          (memoria corta: solo ultime ~10 partite)
  4 Massey         (rating da minimi quadrati sugli scarti gol, non iterativo)
  5 Results-Elo    (ignora il margine: conta solo W/D/L)
  6 XGBoost        (ML non lineare sulle feature)
  7 Logistica      (modello d'esito lineare, link diverso dal Poisson)

I pesi del consenso si ottimizzano LEAVE-ONE-MONDIALE-OUT (tarati su 5 tornei,
valutati sul 6o). Forward-selection: si aggiunge il modello che piu' riduce il
log loss OOS, finche' aggiungere non migliora piu'. Cosi' si trova il numero di
modelli empiricamente, senza presumere e senza overfitting.
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from config import BACKTEST_WORLD_CUPS, WC_START_DATES, DATA_PROC
from elo import compute_elo
from poisson import fit, match_probs, score_matrix, outcome_probs
from features import build_match_features, FEATURE_COLS
from eval_attdef import fit_attack_defense, ad_probs
import metrics as M

NAMES = ["Elo-Poisson", "Attacco/Difesa", "Forma", "Massey", "Results-Elo", "XGBoost", "Logistica"]
LOGIT_COLS = ["d_elo", "home_flag", "d_ppg", "d_gf", "d_ga", "d_mom"]


def fit_massey(train, half_life=900.0):
    teams = sorted(set(train.home_team) | set(train.away_team))
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams)
    cut = train.date.values.max()
    age = (cut - train.date.values) / np.timedelta64(1, "D")
    w = np.sqrt(train.weight.values * 0.5 ** (age / half_life))
    rows = len(train)
    A = np.zeros((rows + n, n)); b = np.zeros(rows + n)
    H = train.home_team.map(idx).values; Aw = train.away_team.map(idx).values
    gd = (train.home_score.values - train.away_score.values).astype(float)
    A[np.arange(rows), H] = w; A[np.arange(rows), Aw] = -w; b[:rows] = w * gd
    A[np.arange(rows, rows + n), np.arange(n)] = 0.3        # ridge verso 0
    r, *_ = np.linalg.lstsq(A, b, rcond=None)
    return {"idx": idx, "r": r}


def massey_probs(mas, h, a, rho):
    idx = mas["idx"]
    if h not in idx or a not in idx:
        return None
    gd = mas["r"][idx[h]] - mas["r"][idx[a]]
    T = 2.6
    lh = float(np.clip((T + gd) / 2, 0.2, 12)); la = float(np.clip((T - gd) / 2, 0.2, 12))
    return outcome_probs(score_matrix(lh, la, rho))


def collect():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, _ = compute_elo(m)
    mm_ro, _, _ = compute_elo(m, ignore_margin=True)
    print("[npool] feature engineering...", file=sys.stderr)
    fm = build_match_features(mm)
    fm["pe_h_ro"] = mm_ro["pre_elo_home"].values
    fm["pe_a_ro"] = mm_ro["pre_elo_away"].values

    rows = []
    for year in BACKTEST_WORLD_CUPS:
        cutoff = pd.Timestamp(WC_START_DATES[year])
        tr = fm[(fm.date < cutoff) & (fm.date >= cutoff - pd.Timedelta(days=365 * 14))].copy()
        trc = tr.dropna(subset=FEATURE_COLS + LOGIT_COLS)
        params = fit(tr, cutoff); rho = params["rho"]
        admodel = fit_attack_defense(tr)
        mas = fit_massey(tr)
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
        xgp = clf.predict_proba(wc[FEATURE_COLS].values)
        lgp = lr.predict_proba(wc[LOGIT_COLS].values)
        for k, (_, r) in enumerate(wc.iterrows()):
            ad = ad_probs(admodel, r.home_team, r.away_team, True)
            ma = massey_probs(mas, r.home_team, r.away_team, rho)
            if ad is None or ma is None:
                continue
            elo = list(match_probs(r.pre_elo_home, r.pre_elo_away, True, params)[0])
            ro = list(match_probs(r.pe_h_ro, r.pe_a_ro, True, params)[0])
            form = list(outcome_probs(score_matrix(min((r.gf_h + r.ga_a) / 2, 12),
                                                   min((r.gf_a + r.ga_h) / 2, 12), rho)))
            rows.append({"year": year, "o": M.outcome_index(r.home_score, r.away_score),
                         "P": [elo, list(ad), form, list(ma), ro, list(xgp[k]), list(lgp[k])]})
        print(f"[npool] {year} fatto", file=sys.stderr)
    df = pd.DataFrame(rows)
    P = np.array(df["P"].tolist())        # (N, 7, 3)
    return P, df["o"].values, df["year"].values


def softmax(t):
    e = np.exp(t - t.max()); return e / e.sum()


def opt_weights(Psub, O):
    k = Psub.shape[1]
    def nll(theta):
        w = softmax(theta)
        mix = (Psub * w[None, :, None]).sum(1)
        mix = mix / mix.sum(1, keepdims=True)
        return M.log_loss(mix, O)
    res = minimize(nll, np.zeros(k), method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-5})
    return softmax(res.x)


def oos_probs(model_idx, P, O, Y):
    """Probabilita OOS leave-one-Mondiale-out (pesi tarati sugli altri tornei)."""
    sub = P[:, model_idx, :]
    probs = np.zeros((len(O), 3))
    for held in np.unique(Y):
        trm = Y != held; tem = Y == held
        w = opt_weights(sub[trm], O[trm])
        mix = (sub[tem] * w[None, :, None]).sum(1)
        probs[tem] = mix / mix.sum(1, keepdims=True)
    return probs


def oos_logloss(model_idx, P, O, Y):
    p = oos_probs(model_idx, P, O, Y)
    return M.log_loss(p, O), M.rps(p, O), M.accuracy(p, O)


def significance(idxA, idxB, P, O, Y, labelA, labelB):
    """Test appaiato: B e' meglio di A in modo statisticamente rilevante?"""
    pa = oos_probs(idxA, P, O, Y); pb = oos_probs(idxB, P, O, Y)
    ar = np.arange(len(O))
    lla = -np.log(np.clip(pa[ar, O], 1e-12, 1))
    llb = -np.log(np.clip(pb[ar, O], 1e-12, 1))
    d = lla - llb                                  # >0 = B migliore
    mean = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); t = mean / se if se else 0
    print(f"\n{labelB} vs {labelA}: differenza log loss media {mean:+.4f} ± {se:.4f} (t={t:.2f})")
    print("  -> " + ("STATISTICAMENTE RILEVANTE (|t|>2)" if abs(t) > 2 else
                      "NON statisticamente rilevante (|t|<2): la differenza e' rumore"))


def run():
    P, O, Y = collect()
    print(f"\nPartite WC totali: {len(O)} | modelli nel pool: {len(NAMES)}")
    # forward selection greedy su log loss OOS
    selected, remaining = [], list(range(len(NAMES)))
    print(f"\n{'N':>2}  {'modello aggiunto':18}{'logloss OOS':>12}{'rps':>8}{'acc':>7}")
    prev = None
    for step in range(len(NAMES)):
        best = None
        for c in remaining:
            ll, rps, acc = oos_logloss(selected + [c], P, O, Y)
            if best is None or ll < best[0]:
                best = (ll, rps, acc, c)
        selected.append(best[3]); remaining.remove(best[3])
        delta = "" if prev is None else f"  (Δ {best[0]-prev:+.4f})"
        print(f"{len(selected):>2}  {NAMES[best[3]]:18}{best[0]:12.4f}{best[1]:8.4f}{best[2]:7.3f}{delta}")
        prev = best[0]
    print("\n--- confronto set specifici (stesso protocollo OOS) ---")
    print(f"{'set':40}{'logloss':>9}{'rps':>8}{'acc':>7}")
    for label, idx in [("Elo-Poisson da solo", [0]),
                       ("Produzione attuale (Elo+AttDif+XGB)", [0, 1, 5]),
                       ("Logistica + Massey (best OOS)", [6, 3]),
                       ("Logistica + Massey + XGB", [6, 3, 5]),
                       ("tutti e 7", list(range(7)))]:
        ll, rps, acc = oos_logloss(idx, P, O, Y)
        print(f"{label:40}{ll:9.4f}{rps:8.4f}{acc:7.3f}")
    # i miglioramenti sono statisticamente rilevanti?
    significance([0], [6, 3], P, O, Y, "Elo-Poisson da solo", "Logistica+Massey")
    significance([0], [0, 1, 5], P, O, Y, "Elo-Poisson da solo", "Produzione attuale")


if __name__ == "__main__":
    run()
