"""Feature engineering leakage-free dal dataset dei risultati.

Per ogni partita calcola feature derivate SOLO da match precedenti (point-in-time):
  - forma recente (punti per gara, ultime 10)
  - attacco/difesa recenti (gol fatti/subiti per gara, ultime 10)
  - giorni di riposo dall'ultima gara (logistica/fatica)
  - partite negli ultimi 30/90 giorni (carico)
  - momentum Elo (variazione rating ultimi ~365 giorni)

Tutto in un'unica passata cronologica con storia per squadra.
"""
import bisect
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from config import ELO_HOME_ADV


def build_match_features(mm: pd.DataFrame) -> pd.DataFrame:
    """mm deve avere pre_elo_home/away (da compute_elo). Aggiunge colonne feature."""
    mm = mm.sort_values("date").reset_index(drop=True)
    dates = mm["date"].values
    homes = mm["home_team"].values
    aways = mm["away_team"].values
    hs = mm["home_score"].values
    as_ = mm["away_score"].values

    last10 = defaultdict(lambda: deque(maxlen=10))   # (pts, gf, ga)
    match_dates = defaultdict(list)                    # date di ogni match giocato
    elo_hist = defaultdict(list)                       # (date, pre_elo) per momentum

    n = len(mm)
    feats = {k: np.zeros(n) for k in
             ["ppg_h", "ppg_a", "gf_h", "gf_a", "ga_h", "ga_a",
              "rest_h", "rest_a", "m90_h", "m90_a", "mom_h", "mom_a", "n_h", "n_a"]}

    def team_stats(t, idx, side):
        hist = last10[t]
        if hist:
            arr = np.array(hist)
            ppg = arr[:, 0].mean(); gf = arr[:, 1].mean(); ga = arr[:, 2].mean()
        else:
            ppg, gf, ga = 1.0, 1.0, 1.0
        feats["ppg_" + side][idx] = ppg
        feats["gf_" + side][idx] = gf
        feats["ga_" + side][idx] = ga
        feats["n_" + side][idx] = len(hist)
        # riposo
        md = match_dates[t]
        cur = dates[idx]
        if md:
            rest = (cur - md[-1]) / np.timedelta64(1, "D")
            feats["rest_" + side][idx] = min(rest, 365)
            cutoff90 = cur - np.timedelta64(90, "D")
            feats["m90_" + side][idx] = sum(1 for d in md if d >= cutoff90)
        else:
            feats["rest_" + side][idx] = 120
        # momentum Elo (rating ~365gg fa)
        eh = elo_hist[t]
        if eh:
            ds = [e[0] for e in eh]
            target = cur - np.timedelta64(365, "D")
            j = bisect.bisect_right(ds, target) - 1
            past_elo = eh[j][1] if j >= 0 else eh[0][1]
        else:
            past_elo = None
        cur_elo = mm["pre_elo_home"].values[idx] if side == "h" else mm["pre_elo_away"].values[idx]
        feats["mom_" + side][idx] = (cur_elo - past_elo) if past_elo is not None else 0.0

    for i in range(n):
        h, a = homes[i], aways[i]
        team_stats(h, i, "h")
        team_stats(a, i, "a")
        # aggiorna storia DOPO aver calcolato le feature (no leakage)
        gd = hs[i] - as_[i]
        ph = 3 if gd > 0 else (1 if gd == 0 else 0)
        pa = 3 if gd < 0 else (1 if gd == 0 else 0)
        last10[h].append((ph, hs[i], as_[i]))
        last10[a].append((pa, as_[i], hs[i]))
        match_dates[h].append(dates[i]); match_dates[a].append(dates[i])
        elo_hist[h].append((dates[i], mm["pre_elo_home"].values[i]))
        elo_hist[a].append((dates[i], mm["pre_elo_away"].values[i]))

    out = mm.copy()
    for k, v in feats.items():
        out[k] = v
    # differenziali (la prospettiva del modello)
    adv = np.where(out["neutral"].values, 0.0, ELO_HOME_ADV)
    out["d_elo"] = out["pre_elo_home"] - out["pre_elo_away"] + adv
    out["d_ppg"] = out["ppg_h"] - out["ppg_a"]
    out["d_gf"] = out["gf_h"] - out["gf_a"]
    out["d_ga"] = out["ga_h"] - out["ga_a"]
    out["d_rest"] = out["rest_h"] - out["rest_a"]
    out["d_m90"] = out["m90_h"] - out["m90_a"]
    out["d_mom"] = out["mom_h"] - out["mom_a"]
    out["home_flag"] = (~out["neutral"]).astype(float)
    out["is_wc"] = out["is_wc"].astype(float)
    return out


FEATURE_COLS = ["d_elo", "d_ppg", "d_gf", "d_ga", "d_rest", "d_m90", "d_mom",
                "home_flag", "is_wc", "gf_h", "ga_h", "gf_a", "ga_a"]


def team_snapshots(mm: pd.DataFrame, asof, elo_as_of: dict) -> dict:
    """Stato feature di ogni squadra alla data asof (solo match anteriori).
    Ritorna team -> {ppg, gf, ga, rest, m90, mom, elo}."""
    asof = np.datetime64(pd.Timestamp(asof))
    hist = mm[mm["date"].values < asof].sort_values("date")
    last10 = defaultdict(lambda: deque(maxlen=10))
    last_date = {}; mdates = defaultdict(list); elo_hist = defaultdict(list)
    H = hist["home_team"].values; A = hist["away_team"].values
    HS = hist["home_score"].values; AS = hist["away_score"].values
    D = hist["date"].values
    PEH = hist["pre_elo_home"].values; PEA = hist["pre_elo_away"].values
    for i in range(len(hist)):
        h, a = H[i], A[i]; gd = HS[i] - AS[i]
        ph = 3 if gd > 0 else (1 if gd == 0 else 0)
        pa = 3 if gd < 0 else (1 if gd == 0 else 0)
        last10[h].append((ph, HS[i], AS[i])); last10[a].append((pa, AS[i], HS[i]))
        last_date[h] = D[i]; last_date[a] = D[i]
        mdates[h].append(D[i]); mdates[a].append(D[i])
        elo_hist[h].append((D[i], PEH[i])); elo_hist[a].append((D[i], PEA[i]))
    out = {}
    for t in set(list(H) + list(A)):
        hh = last10[t]
        if hh:
            arr = np.array(hh); ppg, gf, ga = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
        else:
            ppg, gf, ga = 1.0, 1.0, 1.0
        rest = min((asof - last_date[t]) / np.timedelta64(1, "D"), 365) if t in last_date else 120
        cut90 = asof - np.timedelta64(90, "D")
        m90 = sum(1 for d in mdates[t] if d >= cut90)
        eh = elo_hist[t]; cur = elo_as_of.get(t, 1500.0)
        if eh:
            ds = [e[0] for e in eh]; tgt = asof - np.timedelta64(365, "D")
            j = bisect.bisect_right(ds, tgt) - 1
            mom = cur - (eh[j][1] if j >= 0 else eh[0][1])
        else:
            mom = 0.0
        out[t] = {"ppg": ppg, "gf": gf, "ga": ga, "rest": float(rest),
                  "m90": float(m90), "mom": float(mom), "elo": cur}
    return out
