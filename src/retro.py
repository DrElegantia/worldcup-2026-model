"""Verifica storica: il modello, con i soli dati disponibili PRIMA del torneo,
quanto azzeccava i Mondiali passati?

Per ogni Mondiale (formato classico 32 squadre) ricostruisce dai dati:
  - gironi (co-occorrenza), classifiche reali, qualificate
  - il tabellone REALE (template R16->finale) dagli accoppiamenti effettivi
  - il risultato reale (campione, finalista, semifinaliste)
Poi gira una simulazione Monte Carlo PRE-torneo (Elo al cutoff + Poisson fittato
solo su dati anteriori) e confronta le probabilita previste con l'esito reale.

Tutto data-driven: nessun dato inventato, nessun hardcoding di risultati.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from config import DATA_PROC, WC_START_DATES
from elo import compute_elo, ratings_as_of
from poisson import fit, lambdas

WC_WINDOWS = {
    2010: ("2010-06-11", "2010-07-11"),
    2014: ("2014-06-12", "2014-07-13"),
    2018: ("2018-06-14", "2018-07-15"),
    2022: ("2022-11-20", "2022-12-18"),
}

# Esiti reali delle finali (fatti verificabili; il dataset non registra il vincitore
# ai rigori, quindi campione/finalista vengono da questi fatti ufficiali FIFA).
ACTUAL_FINALS = {
    2010: ("Spain", "Netherlands"),
    2014: ("Germany", "Argentina"),
    2018: ("France", "Croatia"),
    2022: ("Argentina", "France"),
}


def _standings(teams, results):
    pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
    for h, a, hs, as_ in results:
        gf[h] += hs; gf[a] += as_; gd[h] += hs - as_; gd[a] += as_ - hs
        if hs > as_: pts[h] += 3
        elif hs < as_: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
    return sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t]))


def reconstruct(mm, year):
    s, e = WC_WINDOWS[year]
    wc = mm[(mm.tournament == "FIFA World Cup") & (mm.date >= s) & (mm.date <= e)].copy()
    wc = wc.sort_values("date").reset_index(drop=True)
    teams = sorted(set(wc.home_team) | set(wc.away_team))
    if len(teams) != 32:
        return None

    group_matches = wc.iloc[:48]
    ko = wc.iloc[48:].reset_index(drop=True)        # 16 match knockout

    # gironi via co-occorrenza sui 48 match di girone
    adj = defaultdict(set)
    res_by_pair = {}
    for _, r in group_matches.iterrows():
        adj[r.home_team].add(r.away_team); adj[r.away_team].add(r.home_team)
    seen = set(); groups = []
    for t in teams:
        if t in seen: continue
        comp = set(); stack = [t]
        while stack:
            x = stack.pop()
            if x in seen: continue
            seen.add(x); comp.add(x); stack += list(adj[x] - seen)
        groups.append(sorted(comp))
    if any(len(g) != 4 for g in groups) or len(groups) != 8:
        return None

    # classifiche reali per girone -> 1/2 qualificate, posizione di ogni squadra
    group_of = {}; pos_of = {}; group_fixtures = []
    glabels = {}
    for gi, g in enumerate(groups):
        lab = chr(ord("A") + gi)
        res = []
        for _, r in group_matches[group_matches.home_team.isin(g)].iterrows():
            if r.away_team in g:
                res.append((r.home_team, r.away_team, int(r.home_score), int(r.away_score)))
        rank = _standings(g, res)
        for p, t in enumerate(rank):
            group_of[t] = lab; pos_of[t] = p + 1
        glabels[lab] = rank
        for (h, a, _, _2) in res:
            group_fixtures.append((h, a))

    # tabellone reale: template dagli accoppiamenti effettivi.
    # chi avanza si deduce da "chi gioca il turno successivo" (robusto ai rigori).
    r16 = ko.iloc[:8]; qf = ko.iloc[8:12]; sf = ko.iloc[12:14]
    qf_teams = set(qf.home_team) | set(qf.away_team)
    sf_teams = set(sf.home_team) | set(sf.away_team)
    final_teams = set(ACTUAL_FINALS[year])

    def advancer(row, next_teams):
        if row.home_team in next_teams and row.away_team not in next_teams:
            return row.home_team
        if row.away_team in next_teams and row.home_team not in next_teams:
            return row.away_team
        # fallback: punteggio nei 90/120 minuti
        return row.home_team if row.home_score >= row.away_score else row.away_team

    r16_tmpl = []; r16_winners_real = []
    for _, r in r16.iterrows():
        t1, t2 = r.home_team, r.away_team
        r16_tmpl.append(((group_of[t1], pos_of[t1]), (group_of[t2], pos_of[t2])))
        r16_winners_real.append(advancer(r, qf_teams))

    def slot_of(team, winners):
        return next((i for i, w in enumerate(winners) if w == team), None)

    qf_tmpl = []; qf_winners_real = []
    for _, r in qf.iterrows():
        i = slot_of(r.home_team, r16_winners_real); j = slot_of(r.away_team, r16_winners_real)
        if i is None or j is None: return None
        qf_tmpl.append((i, j)); qf_winners_real.append(advancer(r, sf_teams))

    sf_tmpl = []
    for _, r in sf.iterrows():
        i = slot_of(r.home_team, qf_winners_real); j = slot_of(r.away_team, qf_winners_real)
        if i is None or j is None: return None
        sf_tmpl.append((i, j))

    champ, fin = ACTUAL_FINALS[year]
    # finalina 3o/4o posto: l'unico match knockout tra i due perdenti delle semifinali.
    # bronzo = vincitore, legno = 4o (perdente). Storici tutti con esito netto.
    losers = sf_teams - set(ACTUAL_FINALS[year])
    third = fourth = None
    for _, r in ko.iterrows():
        if {r.home_team, r.away_team} == losers:
            if r.home_score >= r.away_score:
                third, fourth = r.home_team, r.away_team
            else:
                third, fourth = r.away_team, r.home_team
            break
    actual = {"champion": champ, "finalist": fin, "third": third, "fourth": fourth,
              "semifinalists": sorted(sf_teams)}

    return {"year": year, "groups": groups, "group_of": group_of, "pos_of": pos_of,
            "group_fixtures": group_fixtures,
            "r16_tmpl": r16_tmpl, "qf_tmpl": qf_tmpl, "sf_tmpl": sf_tmpl,
            "actual": actual}


def simulate_past(rec, elo, params, n=40000, seed=7, cons_ko=None, cons_group=None):
    rng = np.random.default_rng(seed)
    teams = [t for g in rec["groups"] for t in g]
    tidx = {t: i for i, t in enumerate(teams)}
    elo_arr = np.array([elo.get(t, 1500.0) for t in teams])
    labels = sorted({rec["group_of"][t] for t in teams})

    # standings vettoriali (8 gironi, 4 squadre)
    gteams = {lab: [t for t in teams if rec["group_of"][t] == lab] for lab in labels}
    pts = {lab: np.zeros((n, 4)) for lab in labels}
    gf = {lab: np.zeros((n, 4)) for lab in labels}
    ga = {lab: np.zeros((n, 4)) for lab in labels}
    lidx = {t: gteams[rec["group_of"][t]].index(t) for t in teams}

    for (h, a) in rec["group_fixtures"]:
        lab = rec["group_of"][h]
        lh, la = lidx[h], lidx[a]
        if cons_group is not None and (h, a) in cons_group:
            lam_h, lam_a = cons_group[(h, a)]
        else:
            lam_h, lam_a = lambdas(elo_arr[tidx[h]], elo_arr[tidx[a]], True, params)
        hs = rng.poisson(lam_h, n); as_ = rng.poisson(lam_a, n)
        pts[lab][:, lh] += np.where(hs > as_, 3, np.where(hs == as_, 1, 0))
        pts[lab][:, la] += np.where(as_ > hs, 3, np.where(hs == as_, 1, 0))
        gf[lab][:, lh] += hs; gf[lab][:, la] += as_
        ga[lab][:, lh] += as_; ga[lab][:, la] += hs

    # rank entro girone
    qualif = {}   # (lab,pos1based) -> team index array (N,)
    for lab in labels:
        d = gf[lab] - ga[lab]
        key = pts[lab] * 1e6 + (d + 50) * 1e3 + gf[lab] + rng.random((n, 4)) * 1e-3
        order = np.argsort(-key, axis=1)
        base = np.array([tidx[t] for t in gteams[lab]])
        qualif[(lab, 1)] = base[order[:, 0]]
        qualif[(lab, 2)] = base[order[:, 1]]

    def ko(a, b):
        if cons_ko is not None:
            pa = cons_ko[a, b]
            return np.where(rng.random(len(a)) < pa, a, b)
        ea, eb = elo_arr[a], elo_arr[b]
        d = (ea - eb) / 400.0
        la_ = np.exp(params["c0"] + params["c1"] * d)
        lb = np.exp(params["c0"] - params["c1"] * d)
        ga_ = rng.poisson(np.clip(la_, 1e-3, 12)); gb = rng.poisson(np.clip(lb, 1e-3, 12))
        tie = ga_ == gb
        pa = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
        coin = rng.random(len(a)) < pa
        return np.where((ga_ > gb) | (tie & coin), a, b)

    r16w = [ko(qualif[s1], qualif[s2]) for (s1, s2) in rec["r16_tmpl"]]
    qfw = [ko(r16w[i], r16w[j]) for (i, j) in rec["qf_tmpl"]]
    sfw = [ko(qfw[i], qfw[j]) for (i, j) in rec["sf_tmpl"]]
    champ = ko(sfw[0], sfw[1])

    counts = np.zeros(32)
    np.add.at(counts, champ, 1)
    finalists = np.zeros(32)
    for w in sfw: np.add.at(finalists, w, 1)
    semis = np.zeros(32)
    for w in qfw: np.add.at(semis, w, 1)

    probs = []
    for i, t in enumerate(teams):
        probs.append({"team": t, "elo": round(float(elo_arr[i]), 0),
                      "p_champion": round(counts[i] / n, 4),
                      "p_final": round(finalists[i] / n, 4),
                      "p_semi": round(semis[i] / n, 4)})
    probs.sort(key=lambda r: -r["p_champion"])
    return probs


def _rank_of(probs, team):
    return next((i + 1 for i, r in enumerate(probs) if r["team"] == team), None)


def _build_consensus(mm, rec, elo, params, cutoff):
    """Costruisce, point-in-time, il consenso (modelli + tabelle) per un WC passato."""
    import consensus as C
    from features import team_snapshots
    cutoff = pd.Timestamp(cutoff)
    models = C.train(mm, cutoff, market_probs=C.load_market_year(rec["year"]))
    snap = team_snapshots(mm, cutoff, elo)
    teams = [t for g in rec["groups"] for t in g]
    if not all(t in snap for t in teams):
        return None, None
    cons_ko = C.knockout_table(teams, params, models, snap)   # 32x32
    cons_group = {}
    for (h, a) in rec["group_fixtures"]:
        cons_group[(h, a)] = C.consensus_lambda(h, a, True, params, models, snap)
    return cons_ko, cons_group


def run():
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, tl = compute_elo(m)

    out = []
    for year in sorted(WC_WINDOWS):
        rec = reconstruct(mm, year)
        if rec is None:
            print(f"[retro] {year}: ricostruzione fallita, skip", file=sys.stderr)
            continue
        cutoff = WC_START_DATES[year]
        elo = ratings_as_of(tl, pd.Timestamp(cutoff))
        train = mm[(mm.date < pd.Timestamp(cutoff)) & (mm.date >= pd.Timestamp(cutoff) - pd.Timedelta(days=365 * 16))]
        params = fit(train, cutoff)
        actual = rec["actual"]

        # CONSENSO point-in-time (gli stessi 3 modelli del 2026), con fallback a Elo
        cons_ko = cons_group = None
        try:
            cons_ko, cons_group = _build_consensus(mm, rec, elo, params, cutoff)
        except Exception as e:
            print(f"[retro] {year}: consenso non disponibile ({e}), uso Elo", file=sys.stderr)
        probs = simulate_past(rec, elo, params, n=40000, cons_ko=cons_ko, cons_group=cons_group)

        rank = _rank_of(probs, actual["champion"])
        out.append({
            "year": year, "actual": actual,
            "model": "consenso" if cons_ko is not None else "elo-poisson",
            "champion_pred_rank": rank,
            "champion_pred_prob": next((r["p_champion"] for r in probs if r["team"] == actual["champion"]), None),
            "top": probs[:8],
        })
        print(f"[retro] {year}: campione reale {actual['champion']} -> consenso lo dava #{rank} "
              f"({round(100*(out[-1]['champion_pred_prob'] or 0),1)}%); favorito {probs[0]['team']} "
              f"{round(100*probs[0]['p_champion'],1)}%", file=sys.stderr)

    with open(DATA_PROC / "retro.json", "w") as f:
        json.dump({"tournaments": out, "model": "consenso"}, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    run()
