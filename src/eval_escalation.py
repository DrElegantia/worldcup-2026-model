"""Escalation rigorosa del consenso: 1..11 modelli con IPOTESI diverse.

Per ogni N: pesi ottimizzati LEAVE-ONE-MONDIALE-OUT. Due metriche:
  A) log loss OOS a livello partita (6 Mondiali, 384 partite) -> rilevanza statistica
  B) RICOSTRUZIONE del campione per ogni Mondiale (rank del campione reale),
     simulando il torneo col consenso a N modelli (4 WC ricostruibili).

Si scala 1->11 e ci si ferma dove aggiungere modelli non migliora in modo
statisticamente rilevante (test appaiato). Niente assunzioni.

11 modelli (ipotesi/strategie diverse):
  Elo-Poisson, Attacco/Difesa, Forma, Massey, Results-Elo, Colley,
  Short-Elo (recency), Long-Elo (memoria lunga), XGBoost, RandomForest, Logistica
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from config import BACKTEST_WORLD_CUPS, WC_START_DATES, DATA_PROC, ELO_HOME_ADV
from elo import compute_elo, ratings_as_of
from poisson import fit, match_probs, score_matrix, outcome_probs
from features import build_match_features, FEATURE_COLS, team_snapshots
from eval_attdef import fit_attack_defense
import retro as R
import metrics as M

NAMES = ["Elo-Poisson", "Attacco/Difesa", "Forma", "Massey", "Results-Elo", "Colley",
         "Short-Elo", "Long-Elo", "XGBoost", "RandomForest", "Logistica"]
NM = len(NAMES)
LOGIT_COLS = ["d_elo", "home_flag", "d_ppg", "d_gf", "d_ga", "d_mom"]


def fit_rating_lsq(train, by_goals=True, half_life=900.0):
    """Massey (by_goals) o Colley-like (W/L) via minimi quadrati pesati."""
    teams = sorted(set(train.home_team) | set(train.away_team))
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams)
    cut = train.date.values.max()
    age = (cut - train.date.values) / np.timedelta64(1, "D")
    w = np.sqrt(train.weight.values * 0.5 ** (age / half_life))
    rows = len(train)
    A = np.zeros((rows + n, n)); b = np.zeros(rows + n)
    H = train.home_team.map(idx).values; Aw = train.away_team.map(idx).values
    gd = (train.home_score.values - train.away_score.values).astype(float)
    if not by_goals:
        gd = np.sign(gd)
    A[np.arange(rows), H] = w; A[np.arange(rows), Aw] = -w; b[:rows] = w * gd
    A[np.arange(rows, rows + n), np.arange(n)] = 0.3
    r, *_ = np.linalg.lstsq(A, b, rcond=None)
    return idx, r


def fit_models(m, mm, fm, cutoff):
    """Fitta tutti i modelli point-in-time e ritorna predittori snapshot-based."""
    cutoff = pd.Timestamp(cutoff)
    elo = ratings_as_of(mm_tl[id(mm)], cutoff)
    snap = team_snapshots(mm, cutoff, elo)
    tr = fm[(fm.date < cutoff) & (fm.date >= cutoff - pd.Timedelta(days=365 * 14))].copy()
    trc = tr.dropna(subset=FEATURE_COLS + LOGIT_COLS)
    params = fit(tr, cutoff); rho = params["rho"]
    ad = fit_attack_defense(tr)
    mas_idx, mas_r = fit_rating_lsq(tr, by_goals=True)
    col_idx, col_r = fit_rating_lsq(tr, by_goals=False)
    elo_ro = ratings_as_of(elo_ro_tl, cutoff)
    elo_sh = ratings_as_of(elo_sh_tl, cutoff)
    elo_lg = ratings_as_of(elo_lg_tl, cutoff)
    y = trc.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, max_depth=3,
                            n_estimators=300, learning_rate=0.04, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                            eval_metric="mlogloss", verbosity=0)
    clf.fit(trc[FEATURE_COLS].values, y, sample_weight=trc["weight"].values.astype(float))
    rf = RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=20,
                                n_jobs=-1, random_state=0)
    rf.fit(trc[FEATURE_COLS].values, y, sample_weight=trc["weight"].values.astype(float))
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(trc[LOGIT_COLS].values, y, sample_weight=trc["weight"].values.astype(float))

    def feat(h, a, neutral):
        sh, sa = snap[h], snap[a]; adv = 0.0 if neutral else ELO_HOME_ADV
        return {"d_elo": sh["elo"] - sa["elo"] + adv, "home_flag": 0.0 if neutral else 1.0,
                "d_ppg": sh["ppg"] - sa["ppg"], "d_gf": sh["gf"] - sa["gf"],
                "d_ga": sh["ga"] - sa["ga"], "d_mom": sh["mom"] - sa["mom"],
                "d_rest": sh["rest"] - sa["rest"], "d_m90": sh["m90"] - sa["m90"], "is_wc": 1.0,
                "gf_h": sh["gf"], "ga_h": sh["ga"], "gf_a": sa["gf"], "ga_a": sa["ga"]}

    def fc(d): return [d[c] for c in FEATURE_COLS]
    def lc(d): return [d[c] for c in LOGIT_COLS]

    def poi(lh, la):
        return list(outcome_probs(score_matrix(min(max(lh, .2), 12), min(max(la, .2), 12), rho)))

    def predict(h, a, neutral):
        if h not in snap or a not in snap:
            return None
        d = feat(h, a, neutral)
        out = []
        out.append(list(match_probs(snap[h]["elo"], snap[a]["elo"], neutral, params)[0]))   # Elo-Poisson
        # Att/Def
        if h in ad["idx"] and a in ad["idx"]:
            lh = np.exp(ad["b0"] + ad["att"][ad["idx"][h]] + ad["dfn"][ad["idx"][a]] + (0 if neutral else ad["hadv"]))
            la = np.exp(ad["b0"] + ad["att"][ad["idx"][a]] + ad["dfn"][ad["idx"][h]])
            out.append(poi(lh, la))
        else:
            out.append(out[0])
        out.append(poi((d["gf_h"] + d["ga_a"]) / 2, (d["gf_a"] + d["ga_h"]) / 2))            # Forma
        # Massey
        gd = mas_r[mas_idx[h]] - mas_r[mas_idx[a]] if h in mas_idx and a in mas_idx else 0
        out.append(poi((2.6 + gd) / 2, (2.6 - gd) / 2))
        out.append(list(match_probs(elo_ro.get(h, 1500), elo_ro.get(a, 1500), neutral, params)[0]))  # Results-Elo
        # Colley (W/L rating) -> logistica semplice sulla differenza
        cd = (col_r[col_idx[h]] - col_r[col_idx[a]]) if h in col_idx and a in col_idx else 0
        pw = 1 / (1 + np.exp(-2.2 * cd)); pdr = 0.26
        out.append([pw * (1 - pdr), pdr, (1 - pw) * (1 - pdr)])
        out.append(list(match_probs(elo_sh.get(h, 1500), elo_sh.get(a, 1500), neutral, params)[0]))  # Short-Elo
        out.append(list(match_probs(elo_lg.get(h, 1500), elo_lg.get(a, 1500), neutral, params)[0]))  # Long-Elo
        out.append(list(clf.predict_proba(np.array([fc(d)]))[0]))                            # XGBoost
        out.append(list(rf.predict_proba(np.array([fc(d)]))[0]))                             # RandomForest
        out.append(list(lr.predict_proba(np.array([lc(d)]))[0]))                             # Logistica
        return np.array(out)
    return predict, snap, params


# Elo timelines globali (calcolati una volta)
mm_tl = {}; elo_ro_tl = None; elo_sh_tl = None; elo_lg_tl = None


def softmax(t):
    e = np.exp(t - t.max()); return e / e.sum()


def opt_weights(Psub, O):
    k = Psub.shape[1]
    def nll(th):
        w = softmax(th); mix = (Psub * w[None, :, None]).sum(1); mix /= mix.sum(1, keepdims=True)
        return M.log_loss(mix, O)
    return softmax(minimize(nll, np.zeros(k), method="Nelder-Mead",
                            options={"maxiter": 1500, "fatol": 1e-5}).x)


def main():
    global mm_tl, elo_ro_tl, elo_sh_tl, elo_lg_tl
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, tl = compute_elo(m); mm_tl[id(mm)] = tl
    _, _, elo_ro_tl = compute_elo(m, ignore_margin=True)
    _, _, elo_sh_tl = compute_elo(m, k_scale=1.8)
    _, _, elo_lg_tl = compute_elo(m, k_scale=0.5)
    print("[esc] feature engineering...", file=sys.stderr)
    fm = build_match_features(mm)

    # ---- A) raccolta probabilita per-partita (tutti i WC) ----
    rows = []
    champ_data = {}
    for year in BACKTEST_WORLD_CUPS:
        cutoff = WC_START_DATES[year]
        predict, snap, params = fit_models(m, mm, fm, cutoff)
        wc = mm[(mm.tournament == "FIFA World Cup") & (mm.date >= pd.Timestamp(cutoff))
                & (mm.date < pd.Timestamp(cutoff) + pd.Timedelta(days=60))]
        for _, r in wc.iterrows():
            P = predict(r.home_team, r.away_team, True)
            if P is None or P.shape[0] != NM:
                continue
            rows.append({"year": year, "o": M.outcome_index(r.home_score, r.away_score), "P": P.tolist()})
        # dati per ricostruzione campione (solo WC ricostruibili)
        rec = R.reconstruct(mm, year) if year in R.WC_WINDOWS else None
        if rec is not None:
            champ_data[year] = (rec, predict, snap, params, ratings_as_of(tl, pd.Timestamp(cutoff)))
        print(f"[esc] {year} fatto", file=sys.stderr)

    df = pd.DataFrame(rows); P = np.array(df["P"].tolist()); O = df["o"].values; Y = df["year"].values

    # ---- forward selection OOS, curva N=1..11 ----
    sel, rem = [], list(range(NM)); curve = []
    for _ in range(NM):
        best = None
        for c in rem:
            idx = sel + [c]; sub = P[:, idx, :]
            probs = np.zeros((len(O), 3))
            for hld in np.unique(Y):
                trm = Y != hld; tem = Y == hld
                w = opt_weights(sub[trm], O[trm])
                probs[tem] = (sub[tem] * w[None, :, None]).sum(1)
            probs /= probs.sum(1, keepdims=True)
            ll = M.log_loss(probs, O)
            if best is None or ll < best[0]: best = (ll, c, probs)
        sel.append(best[1]); rem.remove(best[1]); curve.append((len(sel), best[0], best[2]))
    print("\n=== A) escalation OOS a livello partita (forward selection) ===")
    print(f"{'N':>2}  {'modello aggiunto':18}{'logloss OOS':>12}")
    prev = None
    for (nn, ll, _), midx in zip(curve, sel):
        d = "" if prev is None else f"  (Δ {ll-prev:+.4f})"
        print(f"{nn:>2}  {NAMES[midx]:18}{ll:12.4f}{d}"); prev = ll
    # significativita: 9 vs 11, e best vs Elo-solo
    def paired(p_a, p_b, la, lb):
        ar = np.arange(len(O)); da = -np.log(np.clip(p_a[ar, O], 1e-12, 1)); db = -np.log(np.clip(p_b[ar, O], 1e-12, 1))
        diff = da - db; mu = diff.mean(); se = diff.std(ddof=1) / np.sqrt(len(diff)); t = mu / se if se else 0
        print(f"  {lb} vs {la}: Δlogloss {mu:+.4f} ± {se:.4f} (t={t:.2f}) -> "
              + ("RILEVANTE" if abs(t) > 2 else "rumore"))
    probs_n = {nn: pr for nn, _, pr in curve}
    print("\nsignificativita (test appaiato):")
    paired(probs_n[9], probs_n[11], "consenso 9 modelli", "consenso 11 modelli")
    paired(probs_n[1], probs_n[min(2, NM)], "1 modello", "2 modelli")
    paired(probs_n[1], probs_n[11], "1 modello", "11 modelli")

    # ---- B) ricostruzione campione per Mondiale, a N crescente ----
    print("\n=== B) ricostruzione del CAMPIONE per Mondiale (rank del campione reale) ===")
    Ns = [1, 3, 5, 7, 9, 11]
    # pesi globali per ciascun N (ottimizzati su TUTTE le partite, ordine = forward selection)
    weights_for_N = {}
    for nn in Ns:
        idx = sel[:nn]; sub = P[:, idx, :]
        weights_for_N[nn] = (idx, opt_weights(sub, O))
    header = "Mondiale (campione)      " + "".join(f"N={n:<5}" for n in Ns)
    print(header)
    for year in sorted(champ_data):
        rec, predict, snap, params, elo = champ_data[year]
        teams32 = [t for g in rec["groups"] for t in g]
        # tabella 32x32 per ogni modello (P(home win) neutro), una volta
        nT = len(teams32); permod = np.zeros((NM, nT, nT, 3))
        for i in range(nT):
            for j in range(nT):
                if i == j: continue
                pr = predict(teams32[i], teams32[j], True)
                permod[:, i, j, :] = pr if pr is not None else 1 / 3
        line = f"{year} {rec['actual']['champion'][:14]:16}"
        for nn in Ns:
            idx, w = weights_for_N[nn]
            cons = (permod[idx] * w[:, None, None, None]).sum(0)        # nT x nT x 3
            cons /= cons.sum(-1, keepdims=True)
            ko = cons[:, :, 0] + cons[:, :, 1] * (cons[:, :, 0] /
                  np.clip(cons[:, :, 0] + cons[:, :, 2], 1e-9, None))   # P(avanza)
            probs = R.simulate_past(rec, elo, params, n=20000, cons_ko=ko)
            rank = next((k + 1 for k, r in enumerate(probs) if r["team"] == rec["actual"]["champion"]), None)
            line += f"#{str(rank):<5}"
        print(line)
    print("\nLettura: se aumentando N il rank del campione non scende, piu' modelli non aiutano a 'azzeccare il vincitore'.")


if __name__ == "__main__":
    main()
