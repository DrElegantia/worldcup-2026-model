"""Consenso di modelli: media le probabilita 1X2 di piu' modelli validi.

Modelli: (1) Elo-Poisson Dixon-Coles, (2) XGBoost su feature, (3) Attacco/Difesa
per squadra. Il consenso (media, o media pesata) e' la probabilita che guida il
Monte Carlo. Backtest WC: pooled 2002-2022 + test out-of-sample 2018-2022.

Si adotta il consenso solo se batte in modo robusto (pooled E test) il miglior
singolo modello.
"""
import sys
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb

from config import BACKTEST_WORLD_CUPS, WC_START_DATES, DATA_PROC
from elo import compute_elo
from poisson import fit, match_probs, score_matrix, outcome_probs
from features import build_match_features, FEATURE_COLS
from eval_attdef import fit_attack_defense, ad_probs
import metrics as M

TRAIN = [2002, 2006, 2010, 2014]; TEST = [2018, 2022]


def collect():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, _ = compute_elo(m)
    fm = build_match_features(mm)
    rows = []
    for year in BACKTEST_WORLD_CUPS:
        cutoff = pd.Timestamp(WC_START_DATES[year])
        train = mm[(mm.date < cutoff) & (mm.date >= cutoff - pd.Timedelta(days=365 * 12))]
        params = fit(train, cutoff)
        admodel = fit_attack_defense(train)
        # XGB
        ftr = fm[(fm.date < cutoff) & (fm.date >= cutoff - pd.Timedelta(days=365 * 16))].dropna(subset=FEATURE_COLS)
        y = ftr.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
        clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, max_depth=3,
                                n_estimators=300, learning_rate=0.04, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                                eval_metric="mlogloss", verbosity=0)
        clf.fit(ftr[FEATURE_COLS].values, y, sample_weight=ftr["weight"].values.astype(float))
        wc = fm[(fm.tournament == "FIFA World Cup") & (fm.date >= cutoff)
                & (fm.date < cutoff + pd.Timedelta(days=60))].dropna(subset=FEATURE_COLS)
        px = clf.predict_proba(wc[FEATURE_COLS].values)
        for k, (_, r) in enumerate(wc.iterrows()):
            (p1, pxd, p2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, params)
            adp = ad_probs(admodel, r.home_team, r.away_team, True)
            if adp is None: continue
            # 4o modello: Poisson di sola FORMA recente (memoria corta, diverso dall'Elo)
            lhf = max(0.2, (r.gf_h + r.ga_a) / 2.0); laf = max(0.2, (r.gf_a + r.ga_h) / 2.0)
            form = outcome_probs(score_matrix(min(lhf, 12), min(laf, 12), params["rho"]))
            rows.append({"year": year, "elo": [p1, pxd, p2], "xgb": list(px[k]),
                         "ad": list(adp), "form": list(form),
                         "o": M.outcome_index(r.home_score, r.away_score)})
        print(f"[consensus] {year} fatto", file=sys.stderr)
    return pd.DataFrame(rows)


def run():
    df = collect()
    elo = np.array(df["elo"].tolist()); xg = np.array(df["xgb"].tolist()); ad = np.array(df["ad"].tolist())
    fo = np.array(df["form"].tolist())
    O = df["o"].values; Y = df["year"].values
    tem = np.isin(Y, TEST); full = np.ones(len(O), bool)

    def norm(p): return p / p.sum(1, keepdims=True)
    cons_eq = norm(elo + xg + ad)                      # consenso a peso uguale

    # consenso a peso ottimizzato sul train (simplex grossolano)
    best_w, best_ll = (1, 0, 0), 1e9
    for w in itertools.product(np.arange(0, 11), repeat=3):
        if sum(w) == 0: continue
        ws = np.array(w) / sum(w)
        mix = norm(ws[0]*elo + ws[1]*xg + ws[2]*ad)
        ll = M.log_loss(mix[np.isin(Y, TRAIN)], O[np.isin(Y, TRAIN)])
        if ll < best_ll: best_ll, best_w = ll, ws
    cons_opt = norm(best_w[0]*elo + best_w[1]*xg + best_w[2]*ad)

    blends = {
        "3 modelli (Elo.6/XGB.2/AD.2)": norm(0.6*elo + 0.2*xg + 0.2*ad),
        "4 modelli +forma (.5/.2/.15/.15)": norm(0.5*elo + 0.2*xg + 0.15*ad + 0.15*fo),
        "4 modelli peso uguale": norm(elo + xg + ad + fo),
        "Forma da sola": norm(fo),
    }

    def show(label, P, mask):
        print(f"{label:30}{M.log_loss(P[mask],O[mask]):9.4f}{M.rps(P[mask],O[mask]):8.4f}{M.accuracy(P[mask],O[mask]):7.3f}")
    print(f"\npeso consenso ottimizzato (Elo/XGB/AttDif) = {np.round(best_w,2)}")
    print(f"\n{'modello':30}{'logloss':>9}{'rps':>8}{'acc':>7}")
    print("--- TEST 2018-2022 ---")
    show("Elo-Poisson (baseline)", elo, tem)
    show("CONSENSO peso uguale", cons_eq, tem)
    for n, P in blends.items(): show(n, P, tem)
    print("--- POOLED 2002-2022 ---")
    show("Elo-Poisson (baseline)", elo, full)
    show("CONSENSO peso uguale", cons_eq, full)
    for n, P in blends.items(): show(n, P, full)


if __name__ == "__main__":
    run()
