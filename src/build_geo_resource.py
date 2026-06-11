"""Genera risorse elevazione committabili dal GeoNames (one-off, locale).

Copre tutte le (citta, paese) presenti nel dataset partite + le sedi 2026 e
l'elevazione di casa per ogni nazionale. Output in data/geo/ (statico, no leakage),
cosi' la produzione/CI non deve scaricare GeoNames.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import geo
from config import DATA_PROC

OUT = Path(__file__).parent.parent / "data" / "geo"
OUT.mkdir(parents=True, exist_ok=True)


def run():
    by_city_cc, by_city, name_to_iso, iso_to_capital = geo.build_lookups()
    m = pd.read_parquet(DATA_PROC / "matches.parquet")
    wc = json.load(open(DATA_PROC / "wc2026.json"))
    # citta/paese da coprire
    pairs = set(zip(m["city"].astype(str), m["country"].astype(str)))
    for f in wc["fixtures"]:
        pairs.add((str(f["city"]), str(f["country"])))

    venue_iso = {c: geo.country_to_iso(c, name_to_iso) for c in {p[1] for p in pairs}}
    city_elev = {}
    for city, country in pairs:
        iso = venue_iso.get(country)
        e = by_city_cc.get((geo._norm(city), iso)) if iso else None
        if e is None:
            e = by_city.get(geo._norm(city))
        if e is not None and not (isinstance(e, float) and np.isnan(e)):
            city_elev[f"{city}|{country}"] = round(float(e), 1)

    teams = sorted(set(m["home_team"]) | set(m["away_team"]))
    home_elev = geo.home_elevation_map(teams, name_to_iso, iso_to_capital, by_city_cc, by_city)
    home_elev = {t: round(float(v), 1) for t, v in home_elev.items() if not np.isnan(v)}

    json.dump(city_elev, open(OUT / "city_elev.json", "w"))
    json.dump(home_elev, open(OUT / "home_elev.json", "w"))
    print(f"city_elev: {len(city_elev)} sedi | home_elev: {len(home_elev)} squadre")
    # spot check 2026
    for f in wc["fixtures"][:1]:
        pass
    for k in ["Mexico City|Mexico", "Guadalajara|Mexico", "Atlanta|United States"]:
        print(" ", k, "->", city_elev.get(k))
    for t in ["Mexico", "Brazil", "Bolivia", "Netherlands"]:
        print(" home", t, "->", home_elev.get(t))


if __name__ == "__main__":
    run()
