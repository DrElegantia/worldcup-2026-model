"""Consenso a 6 modelli con IPOTESI diverse, pesi ottimizzati out-of-sample.

Modelli (ipotesi diverse, non riparametrizzazioni) e pesi:
  - Mercato     (consenso bookmaker: prob-titolo -> forza per squadra)  0.24
  - Massey      (rating da minimi quadrati sugli scarti gol)            0.17
  - Logistica   (modello d'esito lineare, link diverso dal Poisson)     0.16
  - Elo-Poisson (rating scalare iterativo + Poisson Dixon-Coles)         0.16
  - Forma       (memoria corta: solo ultime ~10 partite)                0.145
  - Attacco/Difesa (abilita 2D separate per squadra)                    0.13

Pesi da ottimizzazione leave-one-Mondiale-out con REGOLARIZZAZIONE (alpha=0.3).
Il mercato (Leitner/Zeileis/Hornik) e' il singolo modello piu' informativo: aggiunto
come 6o, ha il peso piu' alto e migliora l'OOS (4 WC con quote: 0.969 -> 0.958).
Quando le quote non sono disponibili per un torneo, il termine mercato ricade
sull'Elo e il consenso resta a 5 modelli. Guida il Monte Carlo (gironi via lambda
di consenso, knockout via tabella di P(avanza)).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

from config import (ELO_HOME_ADV, CONSENSUS_LOGPOOL, CONSENSUS_TEMPERATURE,
                    CONSENSUS_FLOOR)
from poisson import match_probs, score_matrix, outcome_probs
from features import build_match_features, FEATURE_COLS
from eval_attdef import fit_attack_defense, ad_probs
import metrics as M

W = {"market": 0.24, "massey": 0.17, "logit": 0.16, "elo": 0.16, "form": 0.145, "ad": 0.13}


def load_market_2026():
    import csv
    from config import DATA_PROC
    p = DATA_PROC.parent / "market" / "market_2026.csv"
    if not p.exists():
        return None
    return {r["team"]: float(r["market_prob_pct"]) / 100 for r in csv.DictReader(open(p))}


def load_market_year(year):
    import csv
    from config import DATA_PROC
    p = DATA_PROC.parent / "market" / "market_winprob.csv"
    if not p.exists():
        return None
    out = {r["team"]: float(r["market_prob_pct"]) / 100
           for r in csv.DictReader(open(p)) if int(r["year"]) == year}
    return out or None


def market_ratings(probs):
    """Prob-titolo di consenso bookmaker -> forza per squadra (scala-gol)."""
    teams = list(probs)
    lp = np.array([np.log(max(probs[t], 1e-5)) for t in teams])
    lp = lp - lp.mean()
    sc = 1.2 / (lp.std() + 1e-9)
    return {t: float(lp[i] * sc) for i, t in enumerate(teams)}
LOGIT_COLS = ["d_elo", "home_flag", "d_ppg", "d_gf", "d_ga", "d_mom"]
UNIF = np.array([1 / 3, 1 / 3, 1 / 3])


def _cl(x):
    return float(np.clip(x, 0.2, 12))


def fit_massey(train, half_life=900.0):
    teams = sorted(set(train.home_team) | set(train.away_team))
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams)
    cut = train.date.values.max()
    age = (cut - train.date.values) / np.timedelta64(1, "D")
    w = np.sqrt(train.weight.values * 0.5 ** (age / half_life)); rows = len(train)
    A = np.zeros((rows + n, n)); b = np.zeros(rows + n)
    H = train.home_team.map(idx).values; Aw = train.away_team.map(idx).values
    gd = (train.home_score.values - train.away_score.values).astype(float)
    A[np.arange(rows), H] = w; A[np.arange(rows), Aw] = -w; b[:rows] = w * gd
    A[np.arange(rows, rows + n), np.arange(n)] = 0.3
    r, *_ = np.linalg.lstsq(A, b, rcond=None)
    return {"idx": idx, "r": r}


def train(mm, cutoff, market_probs=None):
    cutoff = pd.Timestamp(cutoff)
    fm = build_match_features(mm)
    tr = fm[(fm.date < cutoff) & (fm.date >= cutoff - pd.Timedelta(days=365 * 14))].dropna(subset=FEATURE_COLS + LOGIT_COLS)
    y = tr.apply(lambda r: M.outcome_index(r.home_score, r.away_score), axis=1).values
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(tr[LOGIT_COLS].values, y, sample_weight=tr["weight"].values.astype(float))
    ad = fit_attack_defense(mm[(mm.date < cutoff) & (mm.date >= cutoff - pd.Timedelta(days=365 * 12))])
    mas = fit_massey(mm[(mm.date < cutoff) & (mm.date >= cutoff - pd.Timedelta(days=365 * 14))])
    market = market_ratings(market_probs) if market_probs else None
    return {"lr": lr, "ad": ad, "massey": mas, "market": market}


def _feat(snap, h, a, neutral):
    sh, sa = snap[h], snap[a]; adv = 0.0 if neutral else ELO_HOME_ADV
    return {"d_elo": sh["elo"] - sa["elo"] + adv, "home_flag": 0.0 if neutral else 1.0,
            "d_ppg": sh["ppg"] - sa["ppg"], "d_gf": sh["gf"] - sa["gf"],
            "d_ga": sh["ga"] - sa["ga"], "d_mom": sh["mom"] - sa["mom"],
            "gf_h": sh["gf"], "ga_h": sh["ga"], "gf_a": sa["gf"], "ga_a": sa["ga"]}


def _pool(components):
    """Combina i modelli del consenso. Log-opinion pooling (media geometrica pesata
    delle odds) + temperature scaling, validati out-of-sample (eval_pooling_calib.py:
    log loss e ECE migliori della media aritmetica, guadagni che si sommano).
    components: lista di (peso, prob_vec 1X2)."""
    comps = [(w, np.clip(np.asarray(p, float), CONSENSUS_FLOOR, 1.0)) for w, p in components]
    wsum = sum(w for w, _ in comps) or 1.0
    if CONSENSUS_LOGPOOL:
        logmix = sum(w * np.log(p) for w, p in comps) / wsum
    else:
        logmix = np.log(sum(w * p for w, p in comps) / wsum)
    logmix = logmix / CONSENSUS_TEMPERATURE
    logmix = logmix - logmix.max()
    out = np.exp(logmix)
    return out / out.sum()


def _blend(h, a, neutral, params, models, snap, logit_probs, alt_h=0.0, alt_a=0.0):
    (e1, ex, e2), (lh, la), _ = match_probs(snap[h]["elo"], snap[a]["elo"], neutral, params,
                                            alt_pen_h=alt_h, alt_pen_a=alt_a)
    elo = np.array([e1, ex, e2]); rho = params["rho"]
    sh, sa = snap[h], snap[a]
    comps = [(W["elo"], elo)]
    comps.append((W["form"], np.array(outcome_probs(score_matrix(
        _cl((sh["gf"] + sa["ga"]) / 2), _cl((sa["gf"] + sh["ga"]) / 2), rho)))))
    mi, mr = models["massey"]["idx"], models["massey"]["r"]
    if h in mi and a in mi:
        gd = mr[mi[h]] - mr[mi[a]]
        comps.append((W["massey"], np.array(outcome_probs(score_matrix(_cl((2.6 + gd) / 2), _cl((2.6 - gd) / 2), rho)))))
    else:
        comps.append((W["massey"], elo))
    comps.append((W["logit"], logit_probs if logit_probs is not None else elo))
    adp = ad_probs(models["ad"], h, a, neutral)
    comps.append((W["ad"], np.array(adp) if adp else elo))
    mk = models.get("market")
    if mk and h in mk and a in mk:
        gd = mk[h] - mk[a]
        comps.append((W["market"], np.array(outcome_probs(score_matrix(_cl((2.6 + gd) / 2), _cl((2.6 - gd) / 2), rho)))))
    else:
        comps.append((W["market"], elo))
    return _pool(comps), (lh, la)


def consensus_1x2(h, a, neutral, params, models, snap, alt_pen_h=0.0, alt_pen_a=0.0):
    d = _feat(snap, h, a, neutral)
    try:
        lg = models["lr"].predict_proba(np.array([[d[c] for c in LOGIT_COLS]]))[0]
    except Exception:
        lg = None
    return _blend(h, a, neutral, params, models, snap, lg, alt_pen_h, alt_pen_a)


def consensus_lambda(h, a, neutral, params, models, snap, alt_pen_h=0.0, alt_pen_a=0.0):
    """Lambda di gol che riproducono il consenso 1X2 (per il girone)."""
    c, (lh0, la0) = consensus_1x2(h, a, neutral, params, models, snap, alt_pen_h, alt_pen_a)
    rho = params["rho"]

    def loss(t):
        lh = _cl(lh0 * np.exp(t)); la = _cl(la0 * np.exp(-t))
        p = outcome_probs(score_matrix(lh, la, rho))
        return (p[0] - c[0]) ** 2 + (p[2] - c[2]) ** 2

    t = minimize_scalar(loss, bounds=(-0.8, 0.8), method="bounded").x
    return _cl(lh0 * np.exp(t)), _cl(la0 * np.exp(-t))


def knockout_table(teams, params, models, snap):
    """48x48: P(i avanza su j) di consenso a campo neutro (pareggi -> rigori split)."""
    n = len(teams)
    vecs, pairs = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = _feat(snap, teams[i], teams[j], True)
            vecs.append([d[c] for c in LOGIT_COLS]); pairs.append((i, j))
    try:
        lg = models["lr"].predict_proba(np.array(vecs))
    except Exception:
        lg = np.tile(UNIF, (len(vecs), 1))
    adv = np.full((n, n), 0.5)
    for k, (i, j) in enumerate(pairs):
        c, _ = _blend(teams[i], teams[j], True, params, models, snap, lg[k])
        denom = c[0] + c[2] if (c[0] + c[2]) > 0 else 1.0
        adv[i, j] = c[0] + c[1] * (c[0] / denom)
    return adv
