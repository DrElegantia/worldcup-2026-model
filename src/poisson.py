"""Modello Poisson bivariato con correzione Dixon-Coles, guidato dall'Elo.

Goal attesi come funzione della differenza Elo pre-match (leakage-free):
  d        = (R_home_eff - R_away_eff) / 400      (R_home_eff include vantaggio campo se non neutro)
  lambda_h = exp(c0 + c1 * d + h_goal * home_flag)
  lambda_a = exp(c0 - c1 * d)

Correzione Dixon-Coles (tau) per la dipendenza nei risultati a basso punteggio.
Stima MLE con pesatura temporale (decadimento esponenziale) e peso torneo.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist

from config import ELO_HOME_ADV, POISSON_HALF_LIFE


def _tau(h, a, lh, la, rho):
    # correzione Dixon-Coles sui 4 risultati a basso punteggio
    out = np.ones_like(lh, dtype=float)
    m00 = (h == 0) & (a == 0)
    m01 = (h == 0) & (a == 1)
    m10 = (h == 1) & (a == 0)
    m11 = (h == 1) & (a == 1)
    out[m00] = 1.0 - lh[m00] * la[m00] * rho
    out[m01] = 1.0 + lh[m01] * rho
    out[m10] = 1.0 + la[m10] * rho
    out[m11] = 1.0 - rho
    return np.clip(out, 1e-9, None)


def _prep(matches):
    d = ((matches["pre_elo_home"].values
          + np.where(matches["neutral"].values, 0.0, ELO_HOME_ADV)
          - matches["pre_elo_away"].values) / 400.0)
    home_flag = np.where(matches["neutral"].values, 0.0, 1.0)
    h = matches["home_score"].values.astype(int)
    a = matches["away_score"].values.astype(int)
    return d, home_flag, h, a


def _weights(matches, cutoff, half_life_days=900.0):
    age = (np.datetime64(cutoff) - matches["date"].values) / np.timedelta64(1, "D")
    age = np.clip(age, 0, None)
    decay = 0.5 ** (age / half_life_days)
    return matches["weight"].values * decay


def fit(matches, cutoff, half_life_days=POISSON_HALF_LIFE):
    """Stima MLE di (c0, c1, h_goal, rho) e, se disponibili le colonne di
    altitudine, del coefficiente di penalita altitudine k_alt."""
    d, home_flag, h, a = _prep(matches)
    w = _weights(matches, cutoff, half_life_days)
    has_alt = "alt_pen_home" in matches.columns and "alt_pen_away" in matches.columns
    if has_alt:
        aph = matches["alt_pen_home"].values / 1000.0
        apa = matches["alt_pen_away"].values / 1000.0
    else:
        aph = apa = np.zeros(len(matches))

    def nll(params):
        c0, c1, hg, rho, k = params
        rho = np.clip(rho, -0.2, 0.2)
        lh = np.clip(np.exp(c0 + c1 * d + hg * home_flag - k * aph), 1e-3, 12)
        la = np.clip(np.exp(c0 - c1 * d - k * apa), 1e-3, 12)
        ll = (poisson_dist.logpmf(h, lh) + poisson_dist.logpmf(a, la)
              + np.log(_tau(h, a, lh, la, rho)))
        return -np.sum(w * ll)

    res = minimize(nll, x0=[0.1, 1.0, 0.15, -0.05, 0.1], method="Nelder-Mead",
                   options={"maxiter": 6000, "xatol": 1e-5, "fatol": 1e-5})
    c0, c1, hg, rho, k = res.x
    return {"c0": float(c0), "c1": float(c1), "h_goal": float(hg),
            "rho": float(np.clip(rho, -0.2, 0.2)),
            "k_alt": float(k) if has_alt else 0.0}


def lambdas(elo_home, elo_away, neutral, params, alt_pen_h=0.0, alt_pen_a=0.0):
    d = (elo_home + (0.0 if neutral else ELO_HOME_ADV) - elo_away) / 400.0
    home_flag = 0.0 if neutral else 1.0
    k = params.get("k_alt", 0.0)
    lh = np.exp(params["c0"] + params["c1"] * d + params["h_goal"] * home_flag - k * alt_pen_h / 1000.0)
    la = np.exp(params["c0"] - params["c1"] * d - k * alt_pen_a / 1000.0)
    return float(np.clip(lh, 1e-3, 12)), float(np.clip(la, 1e-3, 12))


def score_matrix(lh, la, rho, max_goals=10):
    ph = poisson_dist.pmf(np.arange(max_goals + 1), lh)
    pa = poisson_dist.pmf(np.arange(max_goals + 1), la)
    mat = np.outer(ph, pa)
    # correzione DC sui 4 angoli bassi
    mat[0, 0] *= 1.0 - lh * la * rho
    mat[0, 1] *= 1.0 + lh * rho
    mat[1, 0] *= 1.0 + la * rho
    mat[1, 1] *= 1.0 - rho
    mat = np.clip(mat, 0, None)
    return mat / mat.sum()


def outcome_probs(mat):
    p_home = np.tril(mat, -1).sum()
    p_draw = np.trace(mat)
    p_away = np.triu(mat, 1).sum()
    return float(p_home), float(p_draw), float(p_away)


def match_probs(elo_home, elo_away, neutral, params, max_goals=10,
                alt_pen_h=0.0, alt_pen_a=0.0):
    lh, la = lambdas(elo_home, elo_away, neutral, params, alt_pen_h, alt_pen_a)
    mat = score_matrix(lh, la, params["rho"], max_goals)
    return outcome_probs(mat), (lh, la), mat
