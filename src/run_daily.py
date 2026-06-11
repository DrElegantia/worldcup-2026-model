"""Orchestratore giornaliero.

1. Aggiorna i dati grezzi (download results.csv)
2. Ricostruisce matches + struttura 2026 (assorbe i risultati reali gia' giocati)
3. Elo as-of oggi, fit Poisson su dati anteriori a oggi
4. Monte Carlo 100k
5. Predizioni 1X2 per le prossime partite
6. Scrive snapshot datato immutabile + aggiorna indice + serie storica

Gli snapshot passati non vengono mai sovrascritti: e' la base della vista archivio
e della vista day-by-day.
"""
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from config import DATA_PROC, SIMS, DOCS, WC_START_DATES
from elo import compute_elo, ratings_as_of
from poisson import fit, match_probs
from simulate import simulate
import ingest
import metrics as M

MODEL_VERSION = "1.0.0-core"   # Elo + Dixon-Coles Poisson


def today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def upcoming_predictions(mm, elo, params, wc, asof):
    """Probabilita' 1X2 per le partite non ancora giocate (prossimi 10 giorni)."""
    preds = []
    asof_ts = pd.Timestamp(asof)
    for f in wc["fixtures"]:
        if f["played"]:
            continue
        fdate = pd.Timestamp(f["date"])
        if fdate < asof_ts or fdate > asof_ts + pd.Timedelta(days=10):
            continue
        h, a = f["home"], f["away"]
        if h not in elo or a not in elo:
            continue
        from config import HOSTS, ELO_HOME_ADV
        neutral = not ((h in HOSTS and f["country"] == h) or (a in HOSTS and f["country"] == a))
        (p1, px, p2), (lh, la), _ = match_probs(elo[h], elo[a], neutral, params)
        preds.append({"date": f["date"], "stage": f["stage"], "group": f["group"],
                      "home": h, "away": a, "city": f["city"],
                      "p_home": round(p1, 3), "p_draw": round(px, 3),
                      "p_away": round(p2, 3),
                      "xg_home": round(lh, 2), "xg_away": round(la, 2)})
    return preds


def played_results(wc):
    return [{"date": f["date"], "group": f["group"], "home": f["home"],
             "away": f["away"], "home_score": f["home_score"],
             "away_score": f["away_score"]}
            for f in wc["fixtures"] if f["played"] and f["stage"] == "group"]


def build_snapshot(asof=None, n=100_000, refresh=True):
    asof = asof or today_str()
    if refresh:
        ingest.build_matches()
        ingest.build_wc2026()
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, tl = compute_elo(m)
    elo = ratings_as_of(tl, pd.Timestamp(asof) + pd.Timedelta(days=1))
    train = mm[(mm.date < pd.Timestamp(asof) + pd.Timedelta(days=1))
               & (mm.date >= pd.Timestamp(asof) - pd.Timedelta(days=365 * 18))]
    params = fit(train, asof)
    wc = json.load(open(DATA_PROC / "wc2026.json"))

    sim = simulate(elo, params, wc, n=n, seed=42)
    preds = upcoming_predictions(mm, elo, params, wc, asof)

    n_played = sum(1 for f in wc["fixtures"] if f["played"])
    snap = {
        "as_of": asof,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "n_sims": sim["n"],
        "poisson_params": {k: round(v, 4) for k, v in params.items()},
        "group_matches_played": n_played,
        "teams": sim["teams"],
        "groups": wc["groups"],
        "upcoming": preds,
        "results": played_results(wc),
    }
    return snap


def write_snapshot(snap):
    date = snap["as_of"]
    path = SIMS / f"{date}.json"
    with open(path, "w") as f:
        json.dump(snap, f, ensure_ascii=False)
    # indice
    dates = sorted(p.stem for p in SIMS.glob("20*.json"))
    index = {
        "tournament": "FIFA World Cup 2026",
        "latest": dates[-1] if dates else date,
        "dates": dates,
        "updated_utc": snap["generated_utc"],
        "model_version": snap["model_version"],
    }
    with open(SIMS / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    # latest pointer
    with open(SIMS / "latest.json", "w") as f:
        json.dump(snap, f, ensure_ascii=False)
    build_history()
    mirror_to_docs()
    print(f"[daily] snapshot {date}: {snap['n_sims']} sim, "
          f"{snap['group_matches_played']} match giocati", file=sys.stderr)


def build_history():
    """Serie storica per la vista day-by-day: per squadra, prob chiave per data."""
    keys = ["p_champion", "p_final", "p_sf", "p_qf", "p_advance", "p_group_winner"]
    series = {}
    dates = sorted(p.stem for p in SIMS.glob("20*.json"))
    for d in dates:
        snap = json.load(open(SIMS / f"{d}.json"))
        for t in snap["teams"]:
            s = series.setdefault(t["team"], {"team": t["team"], "group": t["group"],
                                              "dates": [], **{k: [] for k in keys}})
            s["dates"].append(d)
            for k in keys:
                s[k].append(t.get(k, 0))
    hist = {"dates": dates, "keys": keys, "teams": list(series.values())}
    with open(SIMS / "history.json", "w") as f:
        json.dump(hist, f)
    return hist


def mirror_to_docs():
    """Copia i JSON in docs/ per GitHub Pages (CORS) e per servizio statico."""
    import shutil
    ddir = DOCS / "data"
    ddir.mkdir(parents=True, exist_ok=True)
    for p in SIMS.glob("*.json"):
        shutil.copy(p, ddir / p.name)


if __name__ == "__main__":
    asof = sys.argv[1] if len(sys.argv) > 1 else None
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    snap = build_snapshot(asof=asof, n=n)
    write_snapshot(snap)
