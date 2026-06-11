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

        probs_model, probs_static, outs = [], [], []
        for _, r in wc.iterrows():
            (p1, px, p2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, params)
            (s1, sx, s2), _, _ = match_probs(r.pre_elo_home, r.pre_elo_away, True, static_params)
            o = M.outcome_index(r.home_score, r.away_score)
            probs_model.append([p1, px, p2])
            probs_static.append([s1, sx, s2])
            outs.append(o)
            per_match.append({"year": year, "home": r.home_team, "away": r.away_team,
                              "p_home": round(p1, 3), "p_draw": round(px, 3),
                              "p_away": round(p2, 3), "outcome": o})
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

    out = {"summary": df.round(5).to_dict(orient="records"),
           "static_params": static_params}
    with open(DATA_PROC / "backtest.json", "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(per_match).to_csv(DATA_PROC / "backtest_matches.csv", index=False)
    return df


if __name__ == "__main__":
    run()
