"""Logica del tabellone 2026: allocazione Annex C delle migliori 8 terze e
mappatura degli slot R32 -> finale. Precalcola la tabella di assegnazione per
tutte le C(12,8)=495 combinazioni di terze qualificate.
"""
import json
from itertools import combinations
from pathlib import Path

CFG = json.load(open(Path(__file__).parent / "bracket_2026.json"))
GROUPS = CFG["groups"]                       # ["A".."L"]
THIRD_SLOTS = CFG["third_place_slots"]        # slot_id -> [gruppi ammessi]
SLOT_ORDER = ["74", "77", "79", "80", "81", "82", "85", "87"]


def _matching(qualified_groups):
    """Assegna ad ogni slot un gruppo distinto tra i qualificati, rispettando i
    set ammessi. Augmenting path con ordine deterministico. Ritorna {slot: group}
    o None se non esiste matching perfetto."""
    allowed = {s: [g for g in THIRD_SLOTS[s] if g in qualified_groups] for s in SLOT_ORDER}
    assign = {}
    used = set()

    def try_assign(slot, visiting):
        for g in allowed[slot]:
            if g in visiting:
                continue
            visiting.add(g)
            holder = next((s for s, gg in assign.items() if gg == g), None)
            if holder is None or try_assign(holder, visiting):
                assign[slot] = g
                return True
        return False

    for s in SLOT_ORDER:
        if not try_assign(s, set()):
            return None
    used = set(assign.values())
    if len(used) != len(SLOT_ORDER):
        return None
    return assign


def build_third_allocation_table():
    """{frozenset(8 gruppi terzi qualificati): {slot: group}} per i 495 casi."""
    table = {}
    unsolved = 0
    for combo in combinations(GROUPS, 8):
        m = _matching(set(combo))
        if m is None:
            unsolved += 1
            continue
        table[frozenset(combo)] = m
    return table, unsolved


if __name__ == "__main__":
    t, u = build_third_allocation_table()
    print(f"combinazioni risolte: {len(t)} / 495   non risolte: {u}")
    # esempio
    ex = list(t.items())[0]
    print("esempio:", sorted(ex[0]), "->", ex[1])
