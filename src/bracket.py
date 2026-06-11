"""Logica del tabellone 2026: allocazione UFFICIALE (Annexe C FIFA) delle migliori
8 terze e mappatura degli slot R32 -> finale.

L'allocazione delle terze NON e' un matching qualsiasi: FIFA pubblica la tabella
Annexe C con tutte le C(12,8)=495 combinazioni (regolamento FWC 2026, art. 12.5 +
Annexe C). Per ogni insieme di 8 gruppi che qualificano una terza, la tabella fissa
quale terza va in quale match R32. Usiamo quella tabella ufficiale, committata in
`data/bracket/annexe_c.json`. Il vecchio matching bipartito restava "valido" (slot
ammessi) ma divergeva dall'ufficiale nel 98% delle combinazioni: tenuto solo come
fallback CI-safe se la risorsa manca.
"""
import json
from itertools import combinations
from pathlib import Path

CFG = json.load(open(Path(__file__).parent / "bracket_2026.json"))
GROUPS = CFG["groups"]                       # ["A".."L"]
THIRD_SLOTS = CFG["third_place_slots"]        # slot_id -> [gruppi ammessi]
SLOT_ORDER = ["74", "77", "79", "80", "81", "82", "85", "87"]

_ANNEXE_C = Path(__file__).parent.parent / "data" / "bracket" / "annexe_c.json"


def _matching(qualified_groups):
    """Assegna ad ogni slot un gruppo distinto tra i qualificati, rispettando i
    set ammessi. Augmenting path con ordine deterministico. Ritorna {slot: group}
    o None se non esiste matching perfetto.

    NOTA: fallback non ufficiale. Diverge da Annexe C quando esistono piu' matching
    perfetti (quasi sempre). Usare la tabella ufficiale via build_third_allocation_table.
    """
    allowed = {s: [g for g in THIRD_SLOTS[s] if g in qualified_groups] for s in SLOT_ORDER}
    assign = {}

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
    if len(set(assign.values())) != len(SLOT_ORDER):
        return None
    return assign


def official_allocation(qualified_groups):
    """Allocazione ufficiale Annexe C per un insieme di 8 gruppi qualificati.
    Ritorna {slot: group} o None se la risorsa manca."""
    tab = _load_official()
    if tab is None:
        return None
    return tab.get(",".join(sorted(qualified_groups)))


_OFFICIAL_CACHE = "_unset"


def _load_official():
    global _OFFICIAL_CACHE
    if _OFFICIAL_CACHE == "_unset":
        try:
            _OFFICIAL_CACHE = json.load(open(_ANNEXE_C))["table"]
        except Exception:
            _OFFICIAL_CACHE = None
    return _OFFICIAL_CACHE


def build_third_allocation_table():
    """{frozenset(8 gruppi terzi qualificati): {slot: group}} per i 495 casi.

    Usa la tabella UFFICIALE Annexe C. Fallback al matching bipartito (non ufficiale)
    solo se la risorsa committata manca (es. CI senza il file)."""
    official = _load_official()
    table = {}
    unsolved = 0
    for combo in combinations(GROUPS, 8):
        if official is not None:
            m = official.get(",".join(sorted(combo)))
        else:
            m = _matching(set(combo))
        if m is None:
            unsolved += 1
            continue
        table[frozenset(combo)] = m
    return table, unsolved


if __name__ == "__main__":
    t, u = build_third_allocation_table()
    src = "UFFICIALE (Annexe C)" if _load_official() is not None else "fallback matching"
    print(f"fonte: {src}")
    print(f"combinazioni risolte: {len(t)} / 495   non risolte: {u}")
    ex = list(t.items())[0]
    print("esempio:", sorted(ex[0]), "->", ex[1])
