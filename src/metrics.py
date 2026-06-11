"""Metriche di valutazione probabilistica per esiti 1X2 (ordinati home/draw/away)."""
import numpy as np


def log_loss(probs, outcomes, eps=1e-15):
    """probs: (n,3) [p_home,p_draw,p_away]; outcomes: 0/1/2."""
    probs = np.clip(np.asarray(probs), eps, 1)
    idx = np.asarray(outcomes)
    return float(-np.mean(np.log(probs[np.arange(len(idx)), idx])))


def brier(probs, outcomes):
    probs = np.asarray(probs)
    y = np.zeros_like(probs)
    y[np.arange(len(outcomes)), np.asarray(outcomes)] = 1
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def rps(probs, outcomes):
    """Ranked Probability Score per esiti ordinati (home<draw<away)."""
    probs = np.asarray(probs)
    y = np.zeros_like(probs)
    y[np.arange(len(outcomes)), np.asarray(outcomes)] = 1
    cp = np.cumsum(probs, axis=1)
    cy = np.cumsum(y, axis=1)
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / (probs.shape[1] - 1)))


def accuracy(probs, outcomes):
    return float(np.mean(np.argmax(np.asarray(probs), axis=1) == np.asarray(outcomes)))


def calibration(probs, outcomes, bins=10):
    """Expected Calibration Error sulla classe predetta + reliability per la classe 'home win'."""
    probs = np.asarray(probs)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == np.asarray(outcomes)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def outcome_index(home_score, away_score):
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2
