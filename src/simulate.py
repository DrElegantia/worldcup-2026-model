"""Monte Carlo vettorizzato del Mondiale 2026 (48 squadre, nuovo formato).

Tutto il torneo e' simulato con array NumPy di indici-squadra di forma (N,):
gironi vettorizzati, selezione delle migliori 8 terze, allocazione Annex C,
tabellone R32 -> finale. Default N = 100.000 iterazioni.

I match gia' giocati usano il risultato reale; solo i restanti sono simulati.
Cosi' giorno dopo giorno la simulazione si aggiorna sui risultati veri.
"""
import json
from pathlib import Path

import numpy as np

from config import DATA_PROC, HOSTS, ELO_HOME_ADV
from bracket import CFG, GROUPS, SLOT_ORDER, build_third_allocation_table

LETTER_IDX = {g: i for i, g in enumerate(GROUPS)}


def _build_third_lookup():
    table, _ = build_third_allocation_table()
    lut = np.full((1 << 12, len(SLOT_ORDER)), -1, dtype=np.int8)
    for combo, assign in table.items():
        mask = 0
        for g in combo:
            mask |= (1 << LETTER_IDX[g])
        lut[mask] = [LETTER_IDX[assign[s]] for s in SLOT_ORDER]
    return lut


def simulate(elo: dict, params: dict, wc: dict, n: int = 100_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    c0, c1, hg, rho = params["c0"], params["c1"], params["h_goal"], params["rho"]

    groups = wc["groups"]
    teams = [t for g in GROUPS for t in groups[g]]   # 48, in ordine di girone
    tidx = {t: i for i, t in enumerate(teams)}
    elo_arr = np.array([elo[t] for t in teams], dtype=float)
    group_of = {t: g for g, ts in groups.items() for t in ts}
    # indice locale 0..3 dentro il girone
    local_idx = {t: groups[group_of[t]].index(t) for t in teams}

    k_alt = params.get("k_alt", 0.0)

    # penalita altitudine per ogni fixture dei gironi 2026 (sede nota)
    alt_by_fix = {}
    try:
        import geo
        import pandas as _pd
        gfx = [f for f in wc["fixtures"] if f["stage"] == "group"]
        dfx = _pd.DataFrame([{"home_team": f["home"], "away_team": f["away"],
                              "city": f["city"], "country": f["country"]} for f in gfx])
        dfx, _ = geo.add_elevation(dfx)
        for f, (_, r) in zip(gfx, dfx.iterrows()):
            alt_by_fix[(f["home"], f["away"])] = (r["alt_pen_home"], r["alt_pen_away"])
    except Exception:
        pass

    def lam(i, j, home_i=False, home_j=False, ap_i=0.0, ap_j=0.0):
        adv = (ELO_HOME_ADV if home_i else 0) - (ELO_HOME_ADV if home_j else 0)
        d = (elo_arr[i] + adv - elo_arr[j]) / 400.0
        li = np.exp(c0 + c1 * d + (hg if home_i else 0) - k_alt * ap_i / 1000.0)
        lj = np.exp(c0 - c1 * d + (hg if home_j else 0) - k_alt * ap_j / 1000.0)
        return float(np.clip(li, 1e-3, 12)), float(np.clip(lj, 1e-3, 12))

    # standings per girone: pts/gd/gf shape (N,4)
    pts = {g: np.zeros((n, 4)) for g in GROUPS}
    gf = {g: np.zeros((n, 4)) for g in GROUPS}
    ga = {g: np.zeros((n, 4)) for g in GROUPS}

    fixtures = [f for f in wc["fixtures"] if f["stage"] == "group"]
    for f in fixtures:
        g = f["group"]
        hi, ai = tidx[f["home"]], tidx[f["away"]]
        lh = local_idx[f["home"]]
        la_ = local_idx[f["away"]]
        if f["played"] and f["home_score"] is not None:
            hsc = np.full(n, f["home_score"])
            asc = np.full(n, f["away_score"])
        else:
            home_h = f["home"] in HOSTS and f["country"] == f["home"]
            home_a = f["away"] in HOSTS and f["country"] == f["away"]
            ap_h, ap_a = alt_by_fix.get((f["home"], f["away"]), (0.0, 0.0))
            lhl, lal = lam(hi, ai, home_h, home_a, ap_h, ap_a)
            hsc = rng.poisson(lhl, n)
            asc = rng.poisson(lal, n)
        # punti
        hw = hsc > asc
        aw = asc > hsc
        dr = hsc == asc
        pts[g][:, lh] += np.where(hw, 3, np.where(dr, 1, 0))
        pts[g][:, la_] += np.where(aw, 3, np.where(dr, 1, 0))
        gf[g][:, lh] += hsc
        gf[g][:, la_] += asc
        ga[g][:, lh] += asc
        ga[g][:, la_] += hsc

    # ranking entro girone: key = pts*1e6 + (gd+50)*1e3 + gf + noise
    pos = {}      # g -> (N,4) ranking 0=primo
    for g in GROUPS:
        gd = gf[g] - ga[g]
        key = pts[g] * 1e6 + (gd + 50) * 1e3 + gf[g] + rng.random((n, 4)) * 1e-3
        order = np.argsort(-key, axis=1)         # indici locali ordinati per rank
        rank = np.empty((n, 4), dtype=int)
        for r in range(4):
            rank[np.arange(n), order[:, r]] = r
        pos[g] = rank

    # team-index globali per primo/secondo/terzo di ogni girone
    def slot_team(g, place):
        local = np.where(pos[g] == place)[1].reshape(n)  # local idx con quel rank
        base = [tidx[t] for t in groups[g]]
        return np.array(base)[local]

    winners = {g: slot_team(g, 0) for g in GROUPS}
    runners = {g: slot_team(g, 1) for g in GROUPS}
    thirds = {g: slot_team(g, 2) for g in GROUPS}

    # distribuzione posizione finale nel girone (1/2/3/4) e punti attesi per squadra
    place = np.zeros((48, 4))
    exp_pts = np.zeros(48)
    for g in GROUPS:
        base = np.array([tidx[t] for t in groups[g]])
        for r in range(4):
            local = np.where(pos[g] == r)[1].reshape(n)
            np.add.at(place[:, r], base[local], 1)
        for li in range(4):
            exp_pts[base[li]] = pts[g][:, li].mean()

    # chiave delle terze per selezionare le migliori 8
    third_key = np.zeros((n, 12))
    thirds_mat = np.zeros((n, 12), dtype=int)
    for gi, g in enumerate(GROUPS):
        local = np.where(pos[g] == 2)[1].reshape(n)
        gd = (gf[g] - ga[g])[np.arange(n), local]
        p = pts[g][np.arange(n), local]
        f_ = gf[g][np.arange(n), local]
        third_key[:, gi] = p * 1e6 + (gd + 50) * 1e3 + f_ + rng.random(n) * 1e-3
        thirds_mat[:, gi] = thirds[g]

    # top-8 terze qualificate -> maschera 12 bit
    top8 = np.argsort(-third_key, axis=1)[:, :8]
    qual = np.zeros((n, 12), dtype=bool)
    qual[np.arange(n)[:, None], top8] = True
    third_qualifies = qual.copy()
    bits = (1 << np.arange(12))
    mask = (qual * bits).sum(axis=1)

    lut = _build_third_lookup()
    alloc = lut[mask]                      # (N,8) group-letter-index per slot

    # risolvi le squadre dei 16 match R32
    def resolve(spec, slot_pos=None):
        t = spec["type"]
        if t == "winner":
            return winners[spec["group"]]
        if t == "runner":
            return runners[spec["group"]]
        if t == "third":
            j = SLOT_ORDER.index(spec["slot"])
            gli = alloc[:, j]
            return thirds_mat[np.arange(n), gli]
        raise ValueError(spec)

    def sim_ko(a, b):
        ea, eb = elo_arr[a], elo_arr[b]
        d = (ea - eb) / 400.0
        la_ = np.exp(c0 + c1 * d)
        lb = np.exp(c0 - c1 * d)
        gha = rng.poisson(np.clip(la_, 1e-3, 12))
        ghb = rng.poisson(np.clip(lb, 1e-3, 12))
        aw = gha > ghb
        bw = ghb > gha
        tie = gha == ghb
        pa = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))   # rigori pesati per forza
        coin = rng.random(len(a)) < pa
        a_wins = aw | (tie & coin)
        return np.where(a_wins, a, b)

    # contatori esiti
    reach = {stage: np.zeros(48) for stage in
             ["r32", "r16", "qf", "sf", "final", "champion"]}
    gw = np.zeros(48); ru = np.zeros(48); t3 = np.zeros(48)

    for g in GROUPS:
        np.add.at(gw, winners[g], 1)
        np.add.at(ru, runners[g], 1)
    # terze qualificate
    for gi, g in enumerate(GROUPS):
        q = third_qualifies[:, gi]
        np.add.at(t3, thirds[g][q], 1)

    # R32 competitors
    r32 = {}
    for mid, spec in CFG["r32"].items():
        a = resolve(spec["t1"])
        b = resolve(spec["t2"])
        for x in (a, b):
            np.add.at(reach["r32"], x, 1)
        r32[mid] = (a, b)

    # gioca R32
    win_r32 = {mid: sim_ko(a, b) for mid, (a, b) in r32.items()}
    for w in win_r32.values():
        np.add.at(reach["r16"], w, 1)

    # R16
    win_r16 = {}
    for mid, (m1, m2) in CFG["r16"].items():
        w = sim_ko(win_r32[m1], win_r32[m2])
        win_r16[mid] = w
        np.add.at(reach["qf"], w, 1)

    # QF
    win_qf = {}
    for mid, (m1, m2) in CFG["qf"].items():
        w = sim_ko(win_r16[m1], win_r16[m2])
        win_qf[mid] = w
        np.add.at(reach["sf"], w, 1)

    # SF
    win_sf = {}
    for mid, (m1, m2) in CFG["sf"].items():
        w = sim_ko(win_qf[m1], win_qf[m2])
        win_sf[mid] = w
        np.add.at(reach["final"], w, 1)

    # Final
    (fa, fb) = CFG["final"]["104"]
    champ = sim_ko(win_sf[fa], win_sf[fb])
    np.add.at(reach["champion"], champ, 1)

    # output per squadra
    out = []
    for i, t in enumerate(teams):
        out.append({
            "team": t, "group": group_of[t], "elo": round(float(elo_arr[i]), 1),
            "exp_points": round(float(exp_pts[i]), 2),
            "p_first": round(place[i, 0] / n, 4),
            "p_second": round(place[i, 1] / n, 4),
            "p_third": round(place[i, 2] / n, 4),
            "p_fourth": round(place[i, 3] / n, 4),
            "p_group_winner": round(gw[i] / n, 4),
            "p_runner_up": round(ru[i] / n, 4),
            "p_third_qualify": round(t3[i] / n, 4),
            "p_advance": round(reach["r32"][i] / n, 4),
            "p_r16": round(reach["r16"][i] / n, 4),
            "p_qf": round(reach["qf"][i] / n, 4),
            "p_sf": round(reach["sf"][i] / n, 4),
            "p_final": round(reach["final"][i] / n, 4),
            "p_champion": round(reach["champion"][i] / n, 4),
        })
    out.sort(key=lambda r: -r["p_champion"])
    return {"n": n, "teams": out}


if __name__ == "__main__":
    import pandas as pd
    from elo import compute_elo, ratings_as_of
    from poisson import fit
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    mm, _, tl = compute_elo(m)
    cutoff = "2026-06-11"
    elo = ratings_as_of(tl, cutoff)
    train = mm[(mm.date < pd.Timestamp(cutoff)) & (mm.date >= pd.Timestamp("2008-01-01"))]
    params = fit(train, cutoff)
    wc = json.load(open(DATA_PROC / "wc2026.json"))
    import time
    t0 = time.time()
    res = simulate(elo, params, wc, n=100_000, seed=1)
    print(f"sim 100k in {time.time()-t0:.1f}s")
    print(f"{'TEAM':22}{'grp':4}{'Elo':>7}{'win%':>8}{'final%':>8}{'sf%':>7}{'adv%':>7}")
    for r in res["teams"][:16]:
        print(f"{r['team']:22}{r['group']:4}{r['elo']:7.0f}"
              f"{100*r['p_champion']:8.1f}{100*r['p_final']:8.1f}"
              f"{100*r['p_sf']:7.1f}{100*r['p_advance']:7.1f}")
    print("sum champion% =", round(sum(r['p_champion'] for r in res['teams']), 4))
