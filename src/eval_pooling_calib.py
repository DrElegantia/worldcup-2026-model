"""Test empirico di due migliorie a basso costo dalla letteratura:

  (A) POOLING: media aritmetica delle probabilita (attuale) vs LOG-POOLING
      (media geometrica delle odds, rinormalizzata). Satopaa 2014 / evidenza
      forecasting: il log-pooling batte spesso l'aritmetico ~5% sulle code.
  (B) CALIBRAZIONE: temperature scaling (Platt multiclasse) post-hoc, fit sul
      train. Su campioni piccoli da torneo dimezza l'ECE (Platt > isotonic).

Protocollo: leave-one-Mondiale-out sui 6 WC (2002-2022), 7 modelli del pool
(eval_npool.collect). I pesi e la temperatura si stimano SOLO sui 5 tornei di
train, si valuta sul 6o. Metriche: log loss, RPS, ECE. Nessun leakage.

Si adotta in produzione solo cio' che migliora in modo robusto (OOS, su piu' fold).
"""
import sys
import numpy as np
from scipy.optimize import minimize, minimize_scalar

import metrics as M
from eval_npool import collect, softmax

FLOOR = 1e-4


def arith_pool(P, w):
    mix = (P * w[None, :, None]).sum(1)
    return mix / mix.sum(1, keepdims=True)


def log_pool(P, w):
    Pc = np.clip(P, FLOOR, 1.0)
    logmix = (np.log(Pc) * w[None, :, None]).sum(1)
    logmix -= logmix.max(axis=1, keepdims=True)
    e = np.exp(logmix)
    return e / e.sum(1, keepdims=True)


def opt_w(P, O, pool):
    k = P.shape[1]
    def nll(theta):
        return M.log_loss(pool(P, softmax(theta)), O)
    res = minimize(nll, np.zeros(k), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-5})
    return softmax(res.x)


def fit_temperature(probs, O):
    """Temperature scaling: p_T = softmax(log(p)/T), T>0 minimizza log loss train."""
    z = np.log(np.clip(probs, 1e-9, 1.0))
    def nll(logT):
        T = np.exp(logT)
        zz = z / T
        zz -= zz.max(axis=1, keepdims=True)
        e = np.exp(zz)
        return M.log_loss(e / e.sum(1, keepdims=True), O)
    res = minimize_scalar(nll, bounds=(np.log(0.3), np.log(5.0)), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(probs, T):
    z = np.log(np.clip(probs, 1e-9, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def oos(P, O, Y, pool, calibrate):
    """Leave-one-WC-out. Ritorna probabilita OOS concatenate + temperatura media."""
    out = np.zeros((len(O), 3)); Ts = []
    for h in np.unique(Y):
        tr, te = Y != h, Y == h
        w = opt_w(P[tr], O[tr], pool)
        ptr, pte = pool(P[tr], w), pool(P[te], w)
        if calibrate:
            T = fit_temperature(ptr, O[tr]); Ts.append(T)
            pte = apply_temperature(pte, T)
        out[te] = pte
    return out, (float(np.mean(Ts)) if Ts else 1.0)


def run():
    P, O, Y = collect()
    print(f"\nPartite (6 WC): {len(O)} | modelli pool: {P.shape[1]}")
    print(f"\n{'configurazione':38}{'logloss':>9}{'rps':>8}{'ece':>8}{'  T':>6}")
    configs = [
        ("Aritmetico (attuale)", arith_pool, False),
        ("Aritmetico + calibrazione", arith_pool, True),
        ("Log-pooling", log_pool, False),
        ("Log-pooling + calibrazione", log_pool, True),
    ]
    base = None
    for name, pool, calib in configs:
        p, T = oos(P, O, Y, pool, calib)
        ll, rp, ec = M.log_loss(p, O), M.rps(p, O), M.calibration(p, O)
        if base is None:
            base = ll
        dd = "" if base is None else f"  (Δll {ll-base:+.4f})"
        print(f"{name:38}{ll:9.4f}{rp:8.4f}{ec:8.4f}{T:6.2f}{dd}")
    print("\nNota: baseline = pooling aritmetico con pesi opt OOS (equivalente al"
          " consenso di produzione). Adottare un cambio solo se migliora robustamente.")


if __name__ == "__main__":
    run()
