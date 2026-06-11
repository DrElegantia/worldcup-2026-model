"""Forza-rosa dai rating EA FIFA (sofifa via dataset aperto eddwebster).

Per ogni nazionale, forza = media dei top-23 overall dell'edizione FIFA piu vicina.
Edizioni disponibili: FIFA 18 (-> WC 2018), FIFA 22 (-> WC 2022). Statico per
edizione, point-in-time (l'edizione esce prima del torneo).
"""
import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).parent.parent / "data" / "squad"
RES.mkdir(parents=True, exist_ok=True)

# nazionalita FIFA -> nome squadra dataset martj42
FIFA_TO_TEAM = {
    "Korea Republic": "South Korea", "Korea DPR": "North Korea",
    "China PR": "China", "United States": "United States",
    "Republic of Ireland": "Republic of Ireland", "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast", "Cape Verde Islands": "Cape Verde",
    "Czech Republic": "Czech Republic", "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao", "DR Congo": "DR Congo", "Congo DR": "DR Congo",
}


def build_strength(edition_csv):
    df = pd.read_csv(edition_csv, usecols=["nationality_name", "overall", "player_positions"],
                     low_memory=False)
    top = df.sort_values("overall", ascending=False).groupby("nationality_name").head(23)
    agg = top.groupby("nationality_name")["overall"].mean()
    cnt = df.groupby("nationality_name")["overall"].count()
    out = {}
    for nat, val in agg.items():
        if cnt.get(nat, 0) >= 15:               # serve un bacino minimo
            team = FIFA_TO_TEAM.get(nat, nat)
            out[team] = round(float(val), 2)
    return out


def build_and_save(fifa18_csv, fifa22_csv, fc24_csv=None):
    out = {"2018": build_strength(fifa18_csv), "2022": build_strength(fifa22_csv)}
    if fc24_csv:
        out["2026"] = build_strength(fc24_csv)     # EA FC24, copre i contender
    json.dump(out, open(RES / "squad_strength.json", "w"))
    return out


def load():
    p = RES / "squad_strength.json"
    if p.exists():
        return json.load(open(p))
    return {}


def edition_for_year(year):
    if year < 2020:
        return "2018"
    if year < 2024:
        return "2022"
    return "2026"


SQUAD_COLS = ["d_squad", "squad_h", "squad_a"]
NEUTRAL = 73.0   # forza-rosa di riferimento per squadre non coperte


def strength_of(strength, team, year):
    ed = edition_for_year(year)
    return strength.get(ed, {}).get(team)


def attach_squad(fm, strength):
    import numpy as np
    yrs = fm["date"].dt.year.values
    sh = np.array([strength_of(strength, t, y) for t, y in zip(fm["home_team"].values, yrs)], dtype=float)
    sa = np.array([strength_of(strength, t, y) for t, y in zip(fm["away_team"].values, yrs)], dtype=float)
    fm = fm.copy()
    # d_squad: 0 dove manca (neutro); livelli riempiti col riferimento
    d = sh - sa
    fm["d_squad"] = np.where(np.isnan(d), 0.0, d)
    fm["squad_h"] = np.where(np.isnan(sh), NEUTRAL, sh)
    fm["squad_a"] = np.where(np.isnan(sa), NEUTRAL, sa)
    return fm


if __name__ == "__main__":
    import sys
    s = build_and_save(sys.argv[1], sys.argv[2])
    print("squadre FIFA18:", len(s["2018"]), "| FIFA22:", len(s["2022"]))
