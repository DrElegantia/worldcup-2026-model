"""Elo point-in-time stile World Football Elo.

Processa la storia in ordine cronologico. Per ogni match registra l'Elo PRE-match
di entrambe le squadre (feature leakage-free per costruzione). Mantiene una timeline
per lookup "as-of" a qualunque data di cutoff.

Formula:
  E_home = 1 / (1 + 10^((R_away - (R_home + home_adv)) / 400))
  K = weight * G,  G = moltiplicatore scarto gol (World Football Elo)
  R' = R + K * (W - E)
"""
import numpy as np
import pandas as pd

from config import ELO_BASE, ELO_HOME_ADV


def goal_diff_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def expected(r_home, r_away, neutral):
    adv = 0.0 if neutral else ELO_HOME_ADV
    return 1.0 / (1.0 + 10 ** ((r_away - (r_home + adv)) / 400.0))


def compute_elo(matches: pd.DataFrame, k_scale: float = 1.0):
    """Ritorna (matches_con_pre_elo, ratings_finali, timeline).

    timeline: dict team -> lista (timestamp, rating_after) ordinata, per as-of.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    ratings = {}
    timeline = {}
    pre_h = np.empty(len(matches))
    pre_a = np.empty(len(matches))

    homes = matches["home_team"].values
    aways = matches["away_team"].values
    hs = matches["home_score"].values
    as_ = matches["away_score"].values
    neu = matches["neutral"].values
    wts = matches["weight"].values
    dates = matches["date"].values

    for i in range(len(matches)):
        h, a = homes[i], aways[i]
        rh = ratings.get(h, ELO_BASE)
        ra = ratings.get(a, ELO_BASE)
        pre_h[i] = rh
        pre_a[i] = ra
        eh = expected(rh, ra, neu[i])
        gd = hs[i] - as_[i]
        if gd > 0:
            wh = 1.0
        elif gd < 0:
            wh = 0.0
        else:
            wh = 0.5
        k = wts[i] * goal_diff_multiplier(gd) * k_scale
        delta = k * (wh - eh)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
        timeline.setdefault(h, []).append((dates[i], ratings[h]))
        timeline.setdefault(a, []).append((dates[i], ratings[a]))

    matches = matches.copy()
    matches["pre_elo_home"] = pre_h
    matches["pre_elo_away"] = pre_a
    return matches, ratings, timeline


def ratings_as_of(timeline: dict, cutoff) -> dict:
    """Rating di ogni squadra all'ultima partita STRETTAMENTE prima del cutoff."""
    cutoff = np.datetime64(pd.Timestamp(cutoff))
    out = {}
    for team, hist in timeline.items():
        r = None
        for ts, rating in hist:
            if ts < cutoff:
                r = rating
            else:
                break
        if r is not None:
            out[team] = r
    return out


def elo_win_prob(r_home, r_away, neutral=True):
    """P(home non perde) approssimata via Elo; usata come baseline e prior."""
    return expected(r_home, r_away, neutral)
