"""Backtest point-in-time sui Mondiali 2002-2022.

Per ogni torneo:
  - Elo calcolato online: il pre_elo di ogni match e' leakage-free per costruzione
    (rating prima di quella partita, include solo match precedenti).
  - Poisson rifittato sui soli dati ANTERIORI al cutoff (giorno prima del torneo).
  - Si predice ogni match del torneo e si confronta con l'esito reale.

Benchmark:
  - uniform: 1/3 a tutti (log loss = ln 3 = 1.0986)
  - elo_static: Poisson fittato una volta su tutta la storia (no refit annuale)
  - model: Poisson rifittato per anno (production)
"""
import json
import sys

import numpy as np
import pandas as pd

from config import BACKTEST_WORLD_CUPS, WC_START_DATES, DATA_PROC
from elo import compute_elo
from poisson import fit, match_probs
import metrics as M

# ordine fasi nel formato classico 32 squadre (64 match): 48 gironi, 8 ottavi,
# 4 quarti, 2 semi, 2 fra finale e 3/4 posto
STAGE_ORDER = ["Gironi", "Ottavi", "Quarti", "Semifinali", "Finale"]


def _stage_of(i, total):
    if total >= 64:
        if i < 48: return "Gironi"
        if i < 56: return "Ottavi"
        if i < 60: return "Quarti"
        if i < 62: return "Semifinali"
        return "Finale"
    return "Gironi"


def run():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, _ = compute_elo(m)

    # modello statico: fit unico su tutta la storia fino al 2002 (per confronto)
    static_train = mm[(mm.date < pd.Timestamp("2002-05-31")) & (mm.date >= pd.Timestamp("1990-01-01"))]
    static_params = fit(static_train, "2002-05-31")

    rows = []
    per_match = []
    for year in BACKTEST_WORLD_CUPS:
        cutoff = pd.Timestamp(WC_START_DATES[year])
        wc = mm[(mm.tournament == "FIFA World Cup")
                & (mm.date >= cutoff)
                & (mm.date < cutoff + pd.Timedelta(days=60))].copy()
        # solo match tra le due squadre con Elo definito
        train = mm[(mm.date < cutoff) & (mm.date >= cutoff - pd.Timedelta(days=365 * 18))]
        params = fit(train, cutoff)

        wc = wc.sort_values("date").reset_index(drop=True)
        probs_model, probs_static, outs = [], [], []
        for i, r in wc.iterrows():
            (p1, px, p2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, params)
            (s1, sx, s2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, static_params)
            o = M.outcome_index(r.home_score, r.away_score)
            probs_model.append([p1, px, p2])
            probs_static.append([s1, sx, s2])
            outs.append(o)
            pred = int(np.argmax([p1, px, p2]))
            per_match.append({"year": year, "home": r.home_team, "away": r.away_team,
                              "p_home": round(p1, 3), "p_draw": round(px, 3),
                              "p_away": round(p2, 3), "outcome": o,
                              "pred": pred, "stage": _stage_of(i, len(wc)),
                              "elo_gap": round(abs(r.pre_elo_home - r.pre_elo_away), 1)})
        pm = np.array(probs_model)
        ps = np.array(probs_static)
        uni = np.full_like(pm, 1 / 3)
        rows.append({
            "year": year, "n_matches": len(outs),
            "logloss_model": M.log_loss(pm, outs),
            "logloss_static": M.log_loss(ps, outs),
            "logloss_uniform": M.log_loss(uni, outs),
            "brier_model": M.brier(pm, outs),
            "rps_model": M.rps(pm, outs),
            "acc_model": M.accuracy(pm, outs),
            "ece_model": M.calibration(pm, outs),
        })

    df = pd.DataFrame(rows)
    # aggregato pesato per n match
    w = df.n_matches
    agg = {
        "year": "POOLED", "n_matches": int(w.sum()),
        "logloss_model": float(np.average(df.logloss_model, weights=w)),
        "logloss_static": float(np.average(df.logloss_static, weights=w)),
        "logloss_uniform": float(np.average(df.logloss_uniform, weights=w)),
        "brier_model": float(np.average(df.brier_model, weights=w)),
        "rps_model": float(np.average(df.rps_model, weights=w)),
        "acc_model": float(np.average(df.acc_model, weights=w)),
        "ece_model": float(np.average(df.ece_model, weights=w)),
    }
    df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.round(4).to_string(index=False))

    # accuratezza per fase (pooled sui 6 Mondiali): quante partite azzeccate
    pmdf = pd.DataFrame(per_match)
    pmdf["correct"] = pmdf["pred"] == pmdf["outcome"]
    by_stage = []
    for st in STAGE_ORDER:
        sub = pmdf[pmdf.stage == st]
        if len(sub):
            by_stage.append({"stage": st, "n": int(len(sub)),
                             "correct": int(sub.correct.sum()),
                             "accuracy": round(float(sub.correct.mean()), 4)})
    overall = {"stage": "Tutte", "n": int(len(pmdf)), "correct": int(pmdf.correct.sum()),
               "accuracy": round(float(pmdf.correct.mean()), 4)}
    by_stage.append(overall)
    print("\nAccuratezza per fase (pooled 2002-2022):")
    for r in by_stage:
        print(f"  {r['stage']:14} {r['correct']:3}/{r['n']:3}  {100*r['accuracy']:.1f}%")

    # --- metriche aggiuntive: affidabilita, fascia di favorito, pareggi ---
    pmdf["conf"] = pmdf[["p_home", "p_draw", "p_away"]].max(axis=1)
    # 1) reliability: la confidenza dichiarata corrisponde all'accuratezza osservata?
    reliability = []
    edges = [0.34, 0.45, 0.55, 0.65, 0.75, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = pmdf[(pmdf.conf >= lo) & (pmdf.conf < hi)]
        if len(sub):
            reliability.append({"bin": f"{int(lo*100)}-{min(int(hi*100),100)}%",
                                "n": int(len(sub)),
                                "avg_conf": round(float(sub.conf.mean()), 3),
                                "obs_acc": round(float(sub.correct.mean()), 3)})
    # 2) per fascia di favorito (gap Elo): quanto e' affidabile su gare nette vs equilibrate
    by_favorite = []
    bands = [(0, 75, "equilibrata"), (75, 150, "lieve favorita"),
             (150, 300, "netta favorita"), (300, 9999, "nettissima")]
    for lo, hi, lab in bands:
        sub = pmdf[(pmdf.elo_gap >= lo) & (pmdf.elo_gap < hi)]
        if len(sub):
            P = sub[["p_home", "p_draw", "p_away"]].values; O = sub["outcome"].values
            by_favorite.append({"band": lab, "n": int(len(sub)),
                                 "accuracy": round(float(sub.correct.mean()), 3),
                                 "logloss": round(M.log_loss(P, O), 3)})
    # 3) pareggi: difficili. recall e prob media assegnata
    draws = pmdf[pmdf.outcome == 1]; nondraws = pmdf[pmdf.outcome != 1]
    draw_stats = {"share_real": round(float((pmdf.outcome == 1).mean()), 3),
                  "recall_argmax": round(float((draws.pred == 1).mean()), 3) if len(draws) else 0,
                  "avg_p_draw_on_draws": round(float(draws.p_draw.mean()), 3) if len(draws) else 0,
                  "avg_p_draw_on_nondraws": round(float(nondraws.p_draw.mean()), 3) if len(nondraws) else 0}
    # 4) sharpness e Brier skill score
    uni_ll = float(np.log(3))
    pooled_ll = M.log_loss(pmdf[["p_home", "p_draw", "p_away"]].values, pmdf["outcome"].values)
    extra = {"reliability": reliability, "by_favorite": by_favorite,
             "draws": draw_stats,
             "sharpness_avg_conf": round(float(pmdf.conf.mean()), 3),
             "logloss_skill_vs_uniform": round(1 - pooled_ll / uni_ll, 3)}
    print("\nAffidabilita per fascia di favorito (gap Elo):")
    for r in by_favorite:
        print(f"  {r['band']:16} n={r['n']:3}  acc {100*r['accuracy']:.0f}%  logloss {r['logloss']}")

    out = {"summary": df.round(5).to_dict(orient="records"),
           "by_stage": by_stage,
           "extra": extra,
           "static_params": static_params}
    with open(DATA_PROC / "backtest.json", "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(per_match).to_csv(DATA_PROC / "backtest_matches.csv", index=False)
    return df


if __name__ == "__main__":
    run()
