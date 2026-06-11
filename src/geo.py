"""Elevazione delle sedi e dei paesi via GeoNames (dato statico, no leakage).

Costruisce:
  - city_elev: (citta normalizzata, iso2) -> elevazione m, e fallback per nome
  - country_home_elev: paese squadra -> elevazione della capitale (m)
Da questi calcola, per ogni partita, l'altitudine della sede e la penalita di
altitudine per ciascuna squadra (sentire la quota solo SALENDO rispetto a casa).
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

GEO_DIR = Path("/tmp")
CITIES = GEO_DIR / "cities15000.txt"
COUNTRYINFO = GEO_DIR / "countryInfo.txt"

# nazionali che non sono stati GeoNames standard / nomi diversi dal dataset
HOME_NATION_CAPITAL = {
    "England": ("London", "GB"), "Scotland": ("Edinburgh", "GB"),
    "Wales": ("Cardiff", "GB"), "Northern Ireland": ("Belfast", "GB"),
}
# nazionali che giocano in casa ad alta quota (sede WCQ reale), m
HIGHLAND_HOME = {
    "Bolivia": 3640, "Ecuador": 2850, "Mexico": 2240, "Peru": 150,
    "Colombia": 18, "Guatemala": 1500, "Costa Rica": 1170, "Honduras": 1000,
    "Saudi Arabia": 600, "Iran": 1200, "Afghanistan": 1800, "Ethiopia": 2355,
}
DEFAULT_HOME_ELEV = 100.0   # squadre lowland senza dato: quota bassa di default
# normalizzazione nome paese dataset -> nome GeoNames countryInfo
COUNTRY_ALIAS = {
    "United States": "United States", "South Korea": "South Korea",
    "North Korea": "North Korea", "Ivory Coast": "Ivory Coast",
    "DR Congo": "Democratic Republic of the Congo", "Czech Republic": "Czech Republic",
    "Cape Verde": "Cabo Verde", "Iran": "Iran", "Russia": "Russia",
    "Bolivia": "Bolivia", "Venezuela": "Venezuela", "Tanzania": "Tanzania",
    "Syria": "Syria", "Moldova": "Moldova", "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Curaçao": "Curaçao", "Turkey": "Turkey",
}


def _norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()
    return s


def load_cities():
    cols = ["geonameid", "name", "asciiname", "alt", "lat", "lon", "fclass", "fcode",
            "cc", "cc2", "a1", "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
    df = pd.read_csv(CITIES, sep="\t", header=None, names=cols, low_memory=False)
    df["dem"] = pd.to_numeric(df["dem"], errors="coerce")
    df["pop"] = pd.to_numeric(df["pop"], errors="coerce").fillna(0)
    return df


def build_lookups():
    cities = load_cities()
    cities = cities[cities["dem"].notna()].sort_values("pop", ascending=False)
    cities["n_ascii"] = cities["asciiname"].map(_norm)
    cities["n_name"] = cities["name"].map(_norm)
    a = cities[["n_ascii", "cc", "dem"]].rename(columns={"n_ascii": "n"})
    b = cities[["n_name", "cc", "dem"]].rename(columns={"n_name": "n"})
    allc = pd.concat([a, b], ignore_index=True)          # ascii prima, gia ordinato per pop
    cc_unique = allc.drop_duplicates(["n", "cc"])
    by_city_cc = {(r.n, r.cc): r.dem for r in cc_unique.itertuples(index=False)}
    name_unique = allc.drop_duplicates("n")
    by_city = dict(zip(name_unique["n"], name_unique["dem"]))
    # countryInfo: nome paese -> iso2, capitale
    import csv
    ci = pd.read_csv(COUNTRYINFO, sep="\t", comment="#", header=None, low_memory=False,
                     quoting=csv.QUOTE_NONE, on_bad_lines="skip", dtype=str)
    name_to_iso = {}
    iso_to_capital = {}
    for _, r in ci.iterrows():
        iso2 = r[0]; cname = r[4]; capital = r[5]
        if isinstance(cname, str):
            name_to_iso[cname] = iso2
        iso_to_capital[iso2] = capital
    return by_city_cc, by_city, name_to_iso, iso_to_capital


def country_to_iso(country, name_to_iso):
    c = COUNTRY_ALIAS.get(country, country)
    if c in name_to_iso:
        return name_to_iso[c]
    # match per normalizzazione
    nc = _norm(c)
    for k, v in name_to_iso.items():
        if _norm(k) == nc:
            return v
    return None


def home_elevation_map(teams, name_to_iso, iso_to_capital, by_city_cc, by_city):
    """team (paese) -> elevazione capitale."""
    out = {}
    for t in teams:
        if t in HIGHLAND_HOME:
            out[t] = float(HIGHLAND_HOME[t]); continue
        if t in HOME_NATION_CAPITAL:
            cap, iso = HOME_NATION_CAPITAL[t]
        else:
            iso = country_to_iso(t, name_to_iso)
            cap = iso_to_capital.get(iso) if iso else None
        elev = None
        if cap and iso:
            elev = by_city_cc.get((_norm(cap), iso))
        if elev is None and cap:
            elev = by_city.get(_norm(cap))
        out[t] = float(elev) if elev is not None else DEFAULT_HOME_ELEV
    return out


RESOURCE_DIR = Path(__file__).parent.parent / "data" / "geo"


def load_resource():
    import json
    ce = json.load(open(RESOURCE_DIR / "city_elev.json"))
    he = json.load(open(RESOURCE_DIR / "home_elev.json"))
    return ce, he


def add_elevation(mm):
    """Aggiunge venue_alt, alt_pen_home, alt_pen_away, d_alt_pen a mm.
    Usa i lookup committati (data/geo/*.json); fallback a GeoNames se assenti."""
    try:
        city_elev, home_elev = load_resource()

        def venue_elev(city, country):
            return city_elev.get(f"{city}|{country}", np.nan)

        def home_of(t):
            return home_elev.get(t, DEFAULT_HOME_ELEV)
    except Exception:
        by_city_cc, by_city, name_to_iso, iso_to_capital = build_lookups()
        teams = sorted(set(mm["home_team"]) | set(mm["away_team"]))
        hmap = home_elevation_map(teams, name_to_iso, iso_to_capital, by_city_cc, by_city)
        venue_iso = {c: country_to_iso(c, name_to_iso) for c in mm["country"].dropna().unique()}

        def venue_elev(city, country):
            iso = venue_iso.get(country)
            e = by_city_cc.get((_norm(city), iso)) if iso else None
            if e is None:
                e = by_city.get(_norm(city))
            return float(e) if e is not None else np.nan

        def home_of(t):
            return hmap.get(t, DEFAULT_HOME_ELEV)

    cache = {}
    va = np.empty(len(mm))
    for i, (city, country) in enumerate(zip(mm["city"].values, mm["country"].values)):
        key = (city, country)
        if key not in cache:
            cache[key] = venue_elev(city, country)
        va[i] = cache[key]

    he = mm["home_team"].map(home_of).values
    ae = mm["away_team"].map(home_of).values
    mm = mm.copy()
    mm["venue_alt"] = va
    # penalita altitudine: si soffre salendo oltre ~500 m dalla quota di casa
    mm["alt_pen_home"] = np.clip((va - he) - 500, 0, None)
    mm["alt_pen_away"] = np.clip((va - ae) - 500, 0, None)
    mm["d_alt_pen"] = mm["alt_pen_home"] - mm["alt_pen_away"]
    # riempi mancanti con 0 (nessun effetto altitudine noto)
    for c in ["venue_alt", "alt_pen_home", "alt_pen_away", "d_alt_pen"]:
        mm[c] = mm[c].fillna(0.0)
    return mm, {}


if __name__ == "__main__":
    m = pd.read_parquet(Path(__file__).parent.parent / "data/processed/matches.parquet")
    mm, info = add_elevation(m)
    # copertura
    cov_city = (mm["venue_alt"] > 0).mean()
    print(f"match con venue_alt nota (>0): {cov_city:.1%}")
    print("home elev mancanti:", sum(1 for v in info["home_elev"].values() if np.isnan(v)))
    # citta alta quota note
    hi = mm[mm["venue_alt"] > 1500][["city", "country", "venue_alt"]].drop_duplicates().head(12)
    print(hi.to_string(index=False))
    # esempio: Bolivia in casa a La Paz
    bol = mm[(mm.home_team == "Bolivia") & (mm.venue_alt > 3000)]
    print("partite Bolivia a >3000m:", len(bol))
