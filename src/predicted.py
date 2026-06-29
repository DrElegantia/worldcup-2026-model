"""Classifica prevista per girone + tabellone previsto (modale/deterministico).

Dalla simulazione si ricava lo scenario piu' probabile:
  - ordine girone per punti attesi (tiebreak Elo)
  - 8 migliori terze per Elo -> allocazione Annex C
  - ogni match a eliminazione: vince il favorito (prob 1X2 senza pareggio)
Serve a mostrare un tabellone concreto, accanto alle probabilita complete.
"""
from bracket import CFG, GROUPS, SLOT_ORDER, _matching, official_allocation
from poisson import match_probs


def predicted_standings(teams, elo):
    """Per ogni girone, classifica prevista ordinata (1..4) con punti attesi."""
    by_group = {}
    for g in GROUPS:
        ts = [t for t in teams if t["group"] == g]
        ts = sorted(ts, key=lambda t: (-t.get("exp_points", 0), -t["elo"]))
        by_group[g] = [{"pos": i + 1, "team": t["team"], "exp_points": t.get("exp_points", 0),
                        "p_advance": t.get("p_advance", 0)} for i, t in enumerate(ts)]
    return by_group


def _ko_winner(a, b, elo, params):
    """Favorito e sua probabilita di passaggio (1X2 senza pareggio, campo neutro)."""
    (p1, px, p2), _, _ = match_probs(elo[a], elo[b], True, params)
    pa = p1 + px / 2.0
    pb = p2 + px / 2.0
    s = pa + pb
    pa, pb = pa / s, pb / s
    if pa >= pb:
        return a, round(pa, 3)
    return b, round(pb, 3)


def predicted_bracket(teams, elo, params):
    """Tabellone previsto deterministico R32 -> finale."""
    std = predicted_standings(teams, elo)
    winners = {g: std[g][0]["team"] for g in GROUPS}
    runners = {g: std[g][1]["team"] for g in GROUPS}
    thirds = {g: std[g][2]["team"] for g in GROUPS}

    # Migliori 8 terze per probabilita di qualificazione (p_advance dalla simulazione,
    # che applica i tiebreak FIFA punti -> diff.reti -> gol fatti; Elo solo come spareggio).
    # NB: usare p_advance e non l'Elo grezzo e' cio che, a gironi giocati (p_advance in {0,1}),
    # fa coincidere il tabellone deterministico con le terze REALMENTE qualificate; con l'Elo
    # entravano nel bracket terze gia' eliminate.
    third_p = {g: std[g][2].get("p_advance", 0.0) for g in GROUPS}
    third_rank = sorted(GROUPS, key=lambda g: (-third_p[g], -elo[thirds[g]]))
    qualified = set(third_rank[:8])
    # allocazione UFFICIALE Annexe C (fallback matching se risorsa assente)
    alloc = official_allocation(qualified) or _matching(qualified) or {}

    def resolve(spec):
        if spec["type"] == "winner":
            return winners[spec["group"]]
        if spec["type"] == "runner":
            return runners[spec["group"]]
        if spec["type"] == "third":
            g = alloc.get(spec["slot"])
            return thirds[g] if g else None
        return None

    # R32
    slots = {}        # match_id -> winner
    rounds = []
    r32 = []
    for mid, spec in CFG["r32"].items():
        a = resolve(spec["t1"]); b = resolve(spec["t2"])
        if a and b:
            w, p = _ko_winner(a, b, elo, params)
        else:
            w, p = (a or b), 0.0
        slots[mid] = w
        r32.append({"match": mid, "home": a, "away": b, "winner": w, "p": p})
    rounds.append({"round": "Sedicesimi", "matches": r32})

    def play(stage_name, mapping):
        out = []
        for mid, (m1, m2) in mapping.items():
            a, b = slots[m1], slots[m2]
            w, p = _ko_winner(a, b, elo, params)
            slots[mid] = w
            out.append({"match": mid, "home": a, "away": b, "winner": w, "p": p})
        rounds.append({"round": stage_name, "matches": out})

    play("Ottavi di finale", CFG["r16"])
    play("Quarti di finale", CFG["qf"])
    play("Semifinali", CFG["sf"])
    play("Finale", CFG["final"])

    champion = slots[list(CFG["final"].keys())[0]]
    return {"standings": std, "bracket": rounds, "champion": champion,
            "qualified_thirds": sorted(qualified)}


def actual_bracket(teams, elo, results_map):
    """Tabellone REALE: stessa topologia ufficiale, ma i vincitori vengono dai
    RISULTATI veri (results_map: {frozenset({a,b}): vincitore}); None dove la
    partita non e' ancora stata giocata. R32 seminato dalle qualificate reali
    (prime due per girone + terze con p_advance==1). Si estende automaticamente
    man mano che escono i risultati delle fasi successive."""
    std = predicted_standings(teams, elo)
    winners = {g: std[g][0]["team"] for g in GROUPS}
    runners = {g: std[g][1]["team"] for g in GROUPS}
    thirds = {g: std[g][2]["team"] for g in GROUPS}
    # terze REALMENTE qualificate (le 8 con p_advance==1), non per Elo
    qualified = set(g for g in GROUPS if std[g][2].get("p_advance", 0) == 1)
    alloc = official_allocation(qualified) or _matching(qualified) or {}

    def resolve(spec):
        if spec["type"] == "winner":
            return winners[spec["group"]]
        if spec["type"] == "runner":
            return runners[spec["group"]]
        if spec["type"] == "third":
            g = alloc.get(spec["slot"])
            return thirds[g] if g else None
        return None

    def real_winner(a, b):
        if not a or not b:
            return None
        return results_map.get(frozenset((a, b)))

    slots = {}
    rounds = []
    r32 = []
    for mid, spec in CFG["r32"].items():
        a = resolve(spec["t1"]); b = resolve(spec["t2"])
        w = real_winner(a, b)
        slots[mid] = w
        r32.append({"match": mid, "home": a, "away": b, "winner": w})
    rounds.append({"round": "Sedicesimi", "matches": r32})

    def play(stage_name, mapping):
        out = []
        for mid, (m1, m2) in mapping.items():
            a, b = slots.get(m1), slots.get(m2)
            w = real_winner(a, b)
            slots[mid] = w
            out.append({"match": mid, "home": a, "away": b, "winner": w})
        rounds.append({"round": stage_name, "matches": out})

    play("Ottavi di finale", CFG["r16"])
    play("Quarti di finale", CFG["qf"])
    play("Semifinali", CFG["sf"])
    play("Finale", CFG["final"])
    champion = slots.get(list(CFG["final"].keys())[0])
    return {"bracket": rounds, "champion": champion}
