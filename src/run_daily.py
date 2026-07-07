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

MODEL_VERSION = "2.2.0-consensus6"   # 6 modelli incl. mercato (bookmaker consensus)


def today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def all_match_predictions(tl, params, wc, asof, cons_models=None, snap=None):
    """Per ogni partita 2026: predizione del modello (leakage-free) e, se gia'
    giocata, il risultato reale. Le future usano il CONSENSO di 3 modelli."""
    import numpy as np
    from config import HOSTS
    from elo import ratings_as_of
    from consensus import consensus_1x2
    from poisson import score_matrix
    asof_ts = pd.Timestamp(asof)
    # penalita altitudine per ogni fixture (sede nota)
    alt_by_fix = {}
    try:
        import geo
        dfx = pd.DataFrame([{"home_team": f["home"], "away_team": f["away"],
                             "city": f["city"], "country": f["country"]} for f in wc["fixtures"]])
        dfx, _ = geo.add_elevation(dfx)
        for f, (_, r) in zip(wc["fixtures"], dfx.iterrows()):
            alt_by_fix[(f["home"], f["away"], f["date"])] = (r["alt_pen_home"], r["alt_pen_away"])
    except Exception:
        pass
    out = []
    for f in wc["fixtures"]:
        h, a = f["home"], f["away"]
        fdate = pd.Timestamp(f["date"])
        elo = ratings_as_of(tl, fdate)              # rating prima della partita
        if h not in elo or a not in elo:
            continue
        neutral = not ((h in HOSTS and f["country"] == h) or (a in HOSTS and f["country"] == a))
        ap_h, ap_a = alt_by_fix.get((h, a, f["date"]), (0.0, 0.0))
        if (not f["played"]) and cons_models is not None and snap is not None and h in snap and a in snap:
            (p1, px, p2), (lh, la) = consensus_1x2(h, a, neutral, params, cons_models, snap, ap_h, ap_a)
            mat = score_matrix(lh, la, params["rho"])
        else:
            (p1, px, p2), (lh, la), mat = match_probs(elo[h], elo[a], neutral, params,
                                                      alt_pen_h=ap_h, alt_pen_a=ap_a)
        gh, ga = np.unravel_index(int(np.argmax(mat)), mat.shape)
        if p1 >= px and p1 >= p2:
            pred = h
        elif p2 >= px and p2 >= p1:
            pred = a
        else:
            pred = "Pareggio"
        hours = (fdate - asof_ts) / pd.Timedelta(hours=1)
        rec = {"date": f["date"], "stage": f["stage"], "group": f["group"],
               "home": h, "away": a, "city": f["city"],
               "played": bool(f["played"]),
               "within_48h": bool(0 <= hours <= 48),
               "p_home": round(p1, 3), "p_draw": round(px, 3), "p_away": round(p2, 3),
               "xg_home": round(lh, 2), "xg_away": round(la, 2),
               "pred_winner": pred, "pred_score": f"{gh}-{ga}"}
        if f["played"] and f["home_score"] is not None:
            rec["home_score"] = f["home_score"]
            rec["away_score"] = f["away_score"]
            if f["home_score"] > f["away_score"]:
                actual = h
            elif f["home_score"] < f["away_score"]:
                actual = a
            else:
                actual = "Pareggio"
            rec["actual_winner"] = actual
            rec["hit"] = bool(actual == pred)
        out.append(rec)
    return out


def build_snapshot(asof=None, n=100_000, refresh=True):
    asof = asof or today_str()
    if refresh:
        ingest.build_matches()
        ingest.build_wc2026()
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, tl = compute_elo(m)
    elo = ratings_as_of(tl, pd.Timestamp(asof) + pd.Timedelta(days=1))
    _tl = tl
    train = mm[(mm.date < pd.Timestamp(asof) + pd.Timedelta(days=1))
               & (mm.date >= pd.Timestamp(asof) - pd.Timedelta(days=365 * 18))]
    params = fit(train, asof)
    wc = json.load(open(DATA_PROC / "wc2026.json"))

    # CONSENSO di 3 modelli (Elo-Poisson + XGBoost + Attacco/Difesa) nel Monte Carlo
    cons_models = snap_feat = cons_ko = None
    cons_group = {}
    try:
        import consensus as C
        from features import team_snapshots
        from bracket import GROUPS as _GR
        from config import HOSTS
        cutoff1 = pd.Timestamp(asof) + pd.Timedelta(days=1)
        cons_models = C.train(mm, cutoff1, market_probs=C.load_market_2026())
        snap_feat = team_snapshots(mm, cutoff1, elo)
        # altitudine per fixture dei gironi
        alt_fix = {}
        try:
            import geo
            gdf = pd.DataFrame([{"home_team": f["home"], "away_team": f["away"],
                                 "city": f["city"], "country": f["country"]}
                                for f in wc["fixtures"] if f["stage"] == "group"])
            gdf, _ = geo.add_elevation(gdf)
            for f, (_, r) in zip([x for x in wc["fixtures"] if x["stage"] == "group"], gdf.iterrows()):
                alt_fix[(f["home"], f["away"])] = (r["alt_pen_home"], r["alt_pen_away"])
        except Exception:
            pass
        for f in wc["fixtures"]:
            if f["stage"] != "group" or f["played"]:
                continue
            h, a = f["home"], f["away"]
            if h in snap_feat and a in snap_feat:
                neu = not ((h in HOSTS and f["country"] == h) or (a in HOSTS and f["country"] == a))
                aph, apa = alt_fix.get((h, a), (0.0, 0.0))
                cons_group[(h, a)] = C.consensus_lambda(h, a, neu, params, cons_models, snap_feat, aph, apa)
        teams_order = [t for g in _GR for t in wc["groups"][g]]
        if all(t in snap_feat for t in teams_order):
            cons_ko = C.knockout_table(teams_order, params, cons_models, snap_feat)
    except Exception as e:
        print(f"[daily] consenso non disponibile ({e}), uso Elo-Poisson", file=sys.stderr)

    sim = simulate(elo, params, wc, n=n, seed=42, cons_group=cons_group, cons_ko=cons_ko)
    matches = all_match_predictions(_tl, params, wc, asof, cons_models, snap_feat)
    from predicted import predicted_bracket, actual_bracket
    pred = predicted_bracket(sim["teams"], elo, params)
    # Tabellone REALE della fase a eliminazione: vincitori dai risultati veri,
    # dedotti come nel Monte Carlo (vedi simulate.py):
    #  1) partecipazione: chi gioca un match di turno successivo ha vinto quello
    #     precedente (risolve i pareggi ai rigori, dove il punteggio non basta);
    #  2) fallback: punteggio decisivo.
    # Un match senza vincitore desumibile (ultimo turno non ancora giocato) resta None.
    ko_fx = [f for f in wc["fixtures"] if f["stage"] != "group"]

    def _ko_advances(team, fdate):
        return any(g["date"] > fdate and (g.get("home") == team or g.get("away") == team)
                   for g in ko_fx)

    ko_results = {}
    for f in ko_fx:
        h, a = f.get("home"), f.get("away")
        if not h or not a:
            continue
        win = None
        h_adv, a_adv = _ko_advances(h, f["date"]), _ko_advances(a, f["date"])
        if h_adv and not a_adv:
            win = h
        elif a_adv and not h_adv:
            win = a
        elif (f["played"] and f.get("home_score") is not None
              and f["home_score"] != f["away_score"]):
            win = h if f["home_score"] > f["away_score"] else a
        if win is not None:
            ko_results[frozenset((h, a))] = win
    actual_ko = actual_bracket(sim["teams"], elo, ko_results)

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
        "predicted_standings": pred["standings"],
        "predicted_bracket": pred["bracket"],
        "predicted_champion": pred["champion"],
        "actual_knockout": actual_ko["bracket"],
        "actual_champion": actual_ko["champion"],
        "matches": matches,
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
    bt = DATA_PROC / "backtest.json"
    if bt.exists():
        shutil.copy(bt, ddir / "backtest.json")
    for extra in ("retro.json", "phase2.json"):
        p = DATA_PROC / extra
        if p.exists():
            shutil.copy(p, ddir / extra)


if __name__ == "__main__":
    asof = sys.argv[1] if len(sys.argv) > 1 else None
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    snap = build_snapshot(asof=asof, n=n)
    write_snapshot(snap)
